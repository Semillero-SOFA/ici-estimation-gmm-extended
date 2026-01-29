from utils import *
import argparse

def run_classification_single_model(dist_powers, gaussians, covs, models, timestamp, logger, global_results_dir):
    """
    Itera sobre todas las combinaciones de parámetros y entrena un modelo por fold
    usando train_test_classification_model().
    
    Args:
        dist_powers (list): Lista de tuplas (distancia, power)
        gaussians (list): Lista de números de gaussianas
        covs (list): Lista de tipos de covarianza
        models (list): Lista de nombres de modelos
        timestamp (str): Timestamp del experimento
        logger: Logger configurado
        global_results_dir (str): Directorio global de resultados
    """
    total_runs = len(dist_powers) * len(gaussians) * len(covs) * len(models) * 8
    experiment_count = 0
    ml_pbar = tqdm.tqdm(total=total_runs, desc='ML experiments', unit='run')
    
    for distancia, power in dist_powers:
        for gaussian in gaussians:
            for cov in covs:
                logger.info(f"Cargando dataset: {distancia}km, {power}dBm, {gaussian} gaussians, {cov}")
                # Cargar dataset
                try:
                    database = extract_df(distancia, power, gaussian, cov)
                    logger.info(f"Dataset cargado exitosamente. Shape: {database.shape}")
                except Exception as e:
                    logger.error(f"ERROR al cargar dataset: {e}")
                    continue
                    
                if (distancia, power) == (0, 0):
                    n_classes_list = list(INTERVAL_LIST_0_0.keys())
                elif (distancia, power) == (270, 0):
                    n_classes_list = list(INTERVAL_LIST_270_0.keys())
                elif (distancia, power) == (270, 9):
                    n_classes_list = list(INTERVAL_LIST_270_9.keys())
                
                for n_classes in n_classes_list:
                    for model_name in models:
                        experiment_count += 1
                        # Directorio de salida para este escenario
                        output_dir = os.path.join(global_results_dir,
                            'results_classification',
                            f"{distancia}_{power}",
                            f"run_{timestamp}"
                        )
                        os.makedirs(output_dir, exist_ok=True)
                        # Entrenar CON OSNR
                        try:
                            logger.info(f"\nEntrenando {model_name} CON OSNR ({n_classes} clases)...")
                            results_w = train_test_classification_model(
                                database,
                                model_name,
                                logger=logger,
                                n_classes=n_classes,
                                include_osnr=True,
                                BD=(distancia, power)
                            )
                            
                            filename_w = os.path.join(output_dir, f'class_results_w_{n_classes}classes.csv')
                            save_classification_results(results_w, filename_w,
                                                       gaussian, cov, model_name, n_classes)
                            save_classification_results_detailed(results_w, os.path.join(output_dir, f'class_results_detailed_w_{n_classes}classes.json'),
                                                                 gaussian, cov, model_name, n_classes, logger)
                            logger.info(f"Resultados CON OSNR guardados en: {filename_w}")
                            
                        except Exception as e:
                            logger.error(f"ERROR en entrenamiento CON OSNR: {e}")
                        # Entrenar SIN OSNR
                        try:
                            logger.info(f"\nEntrenando {model_name} SIN OSNR ({n_classes} clases)...")
                            results_wo = train_test_classification_model(
                                database,
                                model_name,
                                logger=logger,
                                n_classes=n_classes,
                                include_osnr=False,
                                BD=(distancia, power)
                            )
                            
                            filename_wo = os.path.join(output_dir, f'class_results_wo_{n_classes}classes.csv')
                            save_classification_results(results_wo, filename_wo,
                                                       gaussian, cov, model_name, n_classes)
                            save_classification_results_detailed(results_wo, os.path.join(output_dir, f'class_results_detailed_wo_{n_classes}classes.json'),
                                                                 gaussian, cov, model_name, n_classes, logger)
                            logger.info(f"Resultados SIN OSNR guardados en: {filename_wo}")
                            
                        except Exception as e:
                            logger.error(f"ERROR en entrenamiento SIN OSNR: {e}")
                            
                        logger.info(f"Completed {model_name} with {n_classes} classes for {gaussian} gaussians, {cov} covariance.")
                        
                        try:
                            ml_pbar.update(1)
                        except Exception:
                            pass
    
    ml_pbar.close()
    logger.info("\n" + "="*70)
    logger.info("EXPERIMENTOS FINALIZADOS")


def run_classification_all_predictions(dist_powers, gaussians, covs, models, timestamp, logger, global_results_dir):
    """
    Itera sobre todas las combinaciones de parámetros y entrena múltiples modelos
    usando train_test_classification_all_predictions() (un modelo diferente por fold).
    
    Args:
        dist_powers (list): Lista de tuplas (distancia, power)
        gaussians (list): Lista de números de gaussianas
        covs (list): Lista de tipos de covarianza
        models (list): Lista de nombres de modelos para usar en cada fold
        timestamp (str): Timestamp del experimento
        logger: Logger configurado
        global_results_dir (str): Directorio global de resultados
    """
    total_runs = len(dist_powers) * len(gaussians) * len(covs) * 8
    experiment_count = 0
    ml_pbar = tqdm.tqdm(total=total_runs, desc='ML experiments (all predictions)', unit='run')
    
    for distancia, power in dist_powers:
        for gaussian in gaussians:
            for cov in covs:
                logger.info(f"Cargando dataset: {distancia}km, {power}dBm, {gaussian} gaussians, {cov}")
                
                try:
                    database = extract_df(distancia, power, gaussian, cov)
                    logger.info(f"Dataset cargado exitosamente. Shape: {database.shape}")
                except Exception as e:
                    logger.error(f"ERROR al cargar dataset: {e}")
                    continue
                    
                if (distancia, power) == (0, 0):
                    n_classes_list = list(INTERVAL_LIST_0_0.keys())
                elif (distancia, power) == (270, 0):
                    n_classes_list = list(INTERVAL_LIST_270_0.keys())
                elif (distancia, power) == (270, 9):
                    n_classes_list = list(INTERVAL_LIST_270_9.keys())
                
                for n_classes in n_classes_list:
                    experiment_count += 1
                    
                    output_dir = os.path.join(global_results_dir,
                        'results_classification_all',
                        f"{distancia}_{power}",
                        f"run_{timestamp}"
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    
                    try:
                        logger.info(f"\nEntrenando múltiples modelos CON OSNR ({n_classes} clases)...")
                        results_w = train_test_classification_all_predictions(
                            database,
                            models,
                            logger=logger,
                            n_classes=n_classes,
                            include_osnr=True,
                            BD=(distancia, power)
                        )
                        
                        acc_test = np.mean(results_w['acc']['test'])
                        prec_test = np.mean(results_w['precision']['test'])
                        rec_test = np.mean(results_w['recall']['test'])
                        f1_test = np.mean(results_w['f1_score']['test'])
                        
                        logger.info(f"Métricas globales CON OSNR - Acc: {acc_test:.4f}, Prec: {prec_test:.4f}, Rec: {rec_test:.4f}, F1: {f1_test:.4f}")
                        
                        filename_w = os.path.join(output_dir, f'class_results_all_w_{n_classes}classes.json')
                        with open(filename_w, 'w') as f:
                            json.dump({
                                'gaussian': gaussian,
                                'covariance': cov,
                                'n_classes': n_classes,
                                'models': models,
                                'metrics': {
                                    'acc_test': acc_test,
                                    'precision_test': prec_test,
                                    'recall_test': rec_test,
                                    'f1_score_test': f1_test
                                },
                                'model_params': results_w['model_params'],
                                'predictions': {
                                    'y_test': [int(y) for y in results_w['y_test']],
                                    'y_pred_test': [int(y) for y in results_w['y_pred_test']]
                                }
                            }, f, indent=4)
                        logger.info(f"Resultados CON OSNR guardados en: {filename_w}")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento CON OSNR: {e}")
                    
                    try:
                        logger.info(f"\nEntrenando múltiples modelos SIN OSNR ({n_classes} clases)...")
                        results_wo = train_test_classification_all_predictions(
                            database,
                            models,
                            logger=logger,
                            n_classes=n_classes,
                            include_osnr=False,
                            BD=(distancia, power)
                        )
                        
                        acc_test = np.mean(results_wo['acc']['test'])
                        prec_test = np.mean(results_wo['precision']['test'])
                        rec_test = np.mean(results_wo['recall']['test'])
                        f1_test = np.mean(results_wo['f1_score']['test'])
                        
                        logger.info(f"Métricas globales SIN OSNR - Acc: {acc_test:.4f}, Prec: {prec_test:.4f}, Rec: {rec_test:.4f}, F1: {f1_test:.4f}")
                        
                        filename_wo = os.path.join(output_dir, f'class_results_all_wo_{n_classes}classes.json')
                        with open(filename_wo, 'w') as f:
                            json.dump({
                                'gaussian': gaussian,
                                'covariance': cov,
                                'n_classes': n_classes,
                                'models': models,
                                'metrics': {
                                    'acc_test': acc_test,
                                    'precision_test': prec_test,
                                    'recall_test': rec_test,
                                    'f1_score_test': f1_test
                                },
                                'model_params': results_wo['model_params'],
                                'predictions': {
                                    'y_test': [int(y) for y in results_wo['y_test']],
                                    'y_pred_test': [int(y) for y in results_wo['y_pred_test']]
                                }
                            }, f, indent=4)
                        logger.info(f"Resultados SIN OSNR guardados en: {filename_wo}")
                        
                    except Exception as e:
                        logger.error(f"ERROR en entrenamiento SIN OSNR: {e}")
                        
                    logger.info(f"Completed all models with {n_classes} classes for {gaussian} gaussians, {cov} covariance.")
                    
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
parser = argparse.ArgumentParser(description='Train classification models')
parser.add_argument('--mode', type=str, default='single', choices=['single', 'all'],
                   help='Training mode: single (one model per fold) or all (multiple models)')
parser.add_argument('--results_dir', type=str, default="D:/Semillero SOFA/gmm_32_definitivo",
                   help='Global results directory')
args = parser.parse_args()

# Update global variables based on arguments
GLOBAL_RESULTS_DIR = args.results_dir
DATASETS_DIR = f"{GLOBAL_RESULTS_DIR}/new_models"

# # Update the module-level variables in utils
# import utils
# utils.GLOBAL_RESULTS_DIR = GLOBAL_RESULTS_DIR
# utils.DATASETS_DIR = DATASETS_DIR

#=====================================================
# Configurar experimentos
#=====================================================
timestamp = datetime.datetime.now().strftime("%m_%d_%H%M")
run_output_dir = os.path.join(GLOBAL_RESULTS_DIR, 'results_classification', f"run_{timestamp}")
os.makedirs(run_output_dir, exist_ok=True)

logger = setup_logger(run_output_dir)
logger.info("="*70)
logger.info("INICIANDO EXPERIMENTOS DE CLASIFICACIÓN")
logger.info(f"Modo: {args.mode}")
logger.info(f"Directorio de resultados: {GLOBAL_RESULTS_DIR}")
logger.info("="*70)

#=====================================================
# Parámetros de experimentación
#=====================================================
dist_powers = [(0,0), (270,0), (270,9)]
gaussians = [16, 24, 32, 40, 48, 56, 64]
covs = ["diag", "spherical"]
models = ["DecisionTree", "SVM", "RandomForest", "XGBoost"]

#=====================================================
# Ejecutar experimentos
#=====================================================
if args.mode == 'single':
    # Opción 1: Entrenar un modelo por fold (método tradicional)
    run_classification_single_model(dist_powers, gaussians, covs, models, timestamp, logger, GLOBAL_RESULTS_DIR)
else:
    # Opción 2: Entrenar múltiples modelos diferentes (un modelo por fold)
    run_classification_all_predictions(dist_powers, gaussians, covs, models, timestamp, logger, GLOBAL_RESULTS_DIR)
