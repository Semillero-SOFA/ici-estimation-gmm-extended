from utils import *
import argparse

def run_regression_single_model(dist_powers, gaussians, covs, models, timestamp, logger, global_results_dir, datasets_dir):
    """
    Itera sobre todas las combinaciones de parámetros y entrena un modelo por fold
    usando train_test_regression_model().
    
    Args:
        dist_powers (list): Lista de tuplas (distancia, power)
        gaussians (list): Lista de números de gaussianas
        covs (list): Lista de tipos de covarianza
        models (list): Lista de nombres de modelos
        timestamp (str): Timestamp del experimento
        logger: Logger configurado
        global_results_dir (str): Directorio global de resultados
        datasets_dir (str): Directorio de datasets
    """
    total_runs = len(dist_powers) * len(gaussians) * len(covs) * len(models)
    ml_pbar = tqdm.tqdm(total=total_runs, desc="ML Model Training Progress")
    
    for distancia, power in dist_powers:
        for gaussian in gaussians:
            for cov in covs:
                logger.info(f"Cargando dataset: {distancia}km, {power}dBm, {gaussian} gaussians, {cov}")
                
                try:
                    database = extract_df(distancia, power, gaussian, cov, datasets_dir)
                    logger.info(f"Dataset cargado exitosamente. Shape: {database.shape}")
                except Exception as e:
                    logger.error(f"ERROR al cargar dataset: {e}")
                    continue
                
                for model_name in models:
                    output_dir = os.path.join(global_results_dir, 'results_regression', 
                                             f"{distancia}_{power}", f"run_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    try:
                        logger.info(f"\nEntrenando {model_name} CON OSNR...")
                        results = train_test_regression_model(database, model_name, logger, include_osnr=True)
                        
                        filename = os.path.join(output_dir, f'reg_results_w.csv')
                        save_regression_results(results, filename, gaussian, cov, model_name, logger)
                        
                        filename_detailed = os.path.join(output_dir, f'reg_results_w_detailed.json')
                        save_regression_results_detailed(results, filename_detailed, gaussian, cov, model_name, logger)
                        logger.info(f"Resultados CON OSNR guardados")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento CON OSNR: {e}")
                    
                    try:
                        logger.info(f"\nEntrenando {model_name} SIN OSNR...")
                        results_wo = train_test_regression_model(database, model_name, logger, include_osnr=False)
                        
                        filename = os.path.join(output_dir, f'reg_results_wo.csv')
                        save_regression_results(results_wo, filename, gaussian, cov, model_name, logger)
                        
                        filename_detailed = os.path.join(output_dir, f'reg_results_wo_detailed.json')
                        save_regression_results_detailed(results_wo, filename_detailed, gaussian, cov, model_name, logger)
                        logger.info(f"Resultados SIN OSNR guardados")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento SIN OSNR: {e}")
                    
                    logger.info(f"Completed {model_name} model for {gaussian} gaussians, {cov} covariance.")
                    
                    try:
                        ml_pbar.update(1)
                    except Exception:
                        pass
    
    ml_pbar.close()
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENTOS FINALIZADOS")


def run_regression_all_models(dist_powers, gaussians, covs, models, timestamp, logger, global_results_dir, datasets_dir):
    """
    Itera sobre todas las combinaciones de parámetros y entrena cada modelo
    usando train_test_regression_all_models() (mismo modelo optimizado en cada fold).
    Acumula todas las predicciones de todos los folds y calcula métricas globales.
    
    Args:
        dist_powers (list): Lista de tuplas (distancia, power)
        gaussians (list): Lista de números de gaussianas
        covs (list): Lista de tipos de covarianza
        models (list): Lista de nombres de modelos a entrenar individualmente
        timestamp (str): Timestamp del experimento
        logger: Logger configurado
        global_results_dir (str): Directorio global de resultados
        datasets_dir (str): Directorio de datasets
    """
    total_runs = len(dist_powers) * len(gaussians) * len(covs) * len(models)
    ml_pbar = tqdm.tqdm(total=total_runs, desc="ML Model Training Progress (all models)")
    
    for distancia, power in dist_powers:
        for gaussian in gaussians:
            for cov in covs:
                logger.info(f"Cargando dataset: {distancia}km, {power}dBm, {gaussian} gaussians, {cov}")
                
                try:
                    database = extract_df(distancia, power, gaussian, cov, datasets_dir)
                    logger.info(f"Dataset cargado exitosamente. Shape: {database.shape}")
                except Exception as e:
                    logger.error(f"ERROR al cargar dataset: {e}")
                    continue
                
                for model_name in models:
                    output_dir = os.path.join(global_results_dir, 'results_regression_all',
                                             f"{distancia}_{power}", f"run_{timestamp}")
                    os.makedirs(output_dir, exist_ok=True)
                    
                    try:
                        logger.info(f"\nEntrenando {model_name} CON OSNR - acumulando predicciones...")
                        results_w = train_test_regression_all_models(database, model_name, logger, include_osnr=True)
                        
                        r2_test = np.mean(results_w['r2']['test'])
                        rmse_test = np.mean(results_w['rmse']['test'])
                        mae_test = np.mean(results_w['mae']['test'])
                        
                        logger.info(f"Métricas globales CON OSNR - R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
                        
                        filename_w = os.path.join(output_dir, f'reg_results_all_w_{model_name}.json')
                        with open(filename_w, 'w') as f:
                            json.dump({
                                'gaussian': gaussian,
                                'covariance': cov,
                                'model': model_name,
                                'metrics': {
                                    'r2_test': r2_test,
                                    'rmse_test': rmse_test,
                                    'mae_test': mae_test
                                },
                                'model_params': results_w['model_params'],
                                'predictions': {
                                    'y_test': [float(y) for y in results_w['y_test']],
                                    'y_pred_test': [float(y) for y in results_w['y_pred_test']]
                                }
                            }, f, indent=4)
                        logger.info(f"Resultados CON OSNR guardados en: {filename_w}")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento CON OSNR: {e}")
                    
                    try:
                        logger.info(f"\nEntrenando {model_name} SIN OSNR - acumulando predicciones...")
                        results_wo = train_test_regression_all_models(database, model_name, logger, include_osnr=False)
                        
                        r2_test = np.mean(results_wo['r2']['test'])
                        rmse_test = np.mean(results_wo['rmse']['test'])
                        mae_test = np.mean(results_wo['mae']['test'])
                        
                        logger.info(f"Métricas globales SIN OSNR - R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
                        
                        filename_wo = os.path.join(output_dir, f'reg_results_all_wo_{model_name}.json')
                        with open(filename_wo, 'w') as f:
                            json.dump({
                                'gaussian': gaussian,
                                'covariance': cov,
                                'model': model_name,
                                'metrics': {
                                    'r2_test': r2_test,
                                    'rmse_test': rmse_test,
                                    'mae_test': mae_test
                                },
                                'model_params': results_wo['model_params'],
                                'predictions': {
                                    'y_test': [float(y) for y in results_wo['y_test']],
                                    'y_pred_test': [float(y) for y in results_wo['y_pred_test']]
                                }
                            }, f, indent=4)
                        logger.info(f"Resultados SIN OSNR guardados en: {filename_wo}")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento SIN OSNR: {e}")
                    
                    logger.info(f"Completed {model_name} for {gaussian} gaussians, {cov} covariance.")
                    
                    try:
                        ml_pbar.update(1)
                    except Exception:
                        pass
    
    ml_pbar.close()
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENTOS FINALIZADOS")


#=====================================================
# Parse arguments
#=====================================================
parser = argparse.ArgumentParser(description='Train regression models')
parser.add_argument('--mode', type=str, default='single', choices=['single', 'all'],
                   help='Training mode: single (one model per fold) or all (multiple models)')
parser.add_argument('--results_dir', type=str, default="D:/Semillero SOFA/gmm_32_definitivo",
                   help='Global results directory')
parser.add_argument('--datasets_dir', type=str, default="D:/Semillero SOFA/gmm_32_definitivo/new_models",
                   help='Datasets directory')
args = parser.parse_args()

GLOBAL_RESULTS_DIR = args.results_dir
DATASETS_DIR = args.datasets_dir

# Expandir ~ (home directory) y convertir a rutas absolutas
GLOBAL_RESULTS_DIR = os.path.abspath(os.path.expanduser(GLOBAL_RESULTS_DIR))
DATASETS_DIR = os.path.abspath(os.path.expanduser(DATASETS_DIR))

#=====================================================
# Crear Logger
#=====================================================
# Crear directorio base de resultados si no existe
os.makedirs(GLOBAL_RESULTS_DIR, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%m_%d_%H%M")
run_output_dir = os.path.join(GLOBAL_RESULTS_DIR, 'results_regression', f"run_{timestamp}")
#os.makedirs(run_output_dir, exist_ok=True)

logger = setup_logger(run_output_dir)

logger.info("INICIANDO EXPERIMENTOS DE REGRESIÓN")
logger.info(f"Modo: {args.mode}")
logger.info(f"Directorio de resultados: {GLOBAL_RESULTS_DIR}")

#=====================================================
# Parametros
#=====================================================
dist_powers = [(0,0), (270,0), (270,9)]
gaussians = [16, 24, 32, 40, 48, 56, 64]
covs = ["diag", "spherical"]
models = ["RandomForest"]

#=====================================================
# Ejecutar experimentos
#=====================================================
if args.mode == 'single':
    # Opción 1: Entrenar un modelo por fold (método tradicional)
    # Calcula métricas promedio de los K folds
    run_regression_single_model(dist_powers, gaussians, covs, models, timestamp, logger, GLOBAL_RESULTS_DIR, DATASETS_DIR)
else:
    # Opción 2: Entrenar cada modelo acumulando predicciones de todos los folds
    # Calcula métricas globales sobre todas las predicciones acumuladas
    models_list = ["DecisionTree", "SVM", "RandomForest", "XGBoost"]
    run_regression_all_models(dist_powers, gaussians, covs, models_list, timestamp, logger, GLOBAL_RESULTS_DIR, DATASETS_DIR)
