from utils import *
#=====================================================
# Configurar experimentos
#=====================================================
timestamp = datetime.datetime.now().strftime("%m_%d_%H%M")
run_output_dir = os.path.join(GLOBAL_RESULTS_DIR, 'results_classification', f"run_{timestamp}")
os.makedirs(run_output_dir, exist_ok=True)

#log, log_file = configurar_logs(run_output_dir, timestamp)
logger = setup_logger(run_output_dir)
logger.info("="*70)
logger.info("INICIANDO EXPERIMENTOS DE CLASIFICACIÓN")
logger.info("="*70)

#=====================================================
# Parámetros de experimentación
#=====================================================
dist_powers = [(0,0), (270,0), (270,9)]
gaussians = [16, 24, 32, 40, 48, 56, 64]
covs = ["diag", "spherical"]
#models = ["DecisionTree", "SVM", "RandomForest"]
models = ['XGBoost']
#n_classes_list = ["2", "3", "4", "5", "6", "8", "full"]  # Diferentes números de clases

#=====================================================
# Iterar sobre todos los escenarios
#=====================================================
total_runs = len(dist_powers) * len(gaussians) * len(covs) * len(models) * 8
experiment_count = 0
ml_pbar = tqdm.tqdm(total=total_runs, desc='ML experiments', unit='run')
for distancia,power in dist_powers:
    for gaussian in gaussians:
        #TODO: Run in parallel
        for cov in covs:

            logger.info(f"Cargando dataset: {distancia}km, {power}dBm, {gaussian} gaussians, {cov}")

            
            # Cargar dataset
            try:
                database = extract_df(distancia, power, gaussian, cov)
                logger.info(f"Dataset cargado exitosamente. Shape: {database.shape}")
            except Exception as e:
                logger.error(f"ERROR al cargar dataset: {e}")
                continue
            if (distancia, power) == (0,0):
                n_classes_list = list(INTERVAL_LIST_0_0.keys())
            elif (distancia, power) == (270,0):
                n_classes_list = list(INTERVAL_LIST_270_0.keys())
            elif (distancia, power) == (270,9):
                n_classes_list = list(INTERVAL_LIST_270_9.keys())

            for n_classes in n_classes_list:
                for model_name in models:
                    experiment_count += 1

                    # Directorio de salida para este escenario
                    output_dir = os.path.join(GLOBAL_RESULTS_DIR, 
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
                        # If progress update fails (rare), continue without stopping the experiment
                        pass
logger.info("\n" + "="*70)
logger.info("EXPERIMENTOS FINALIZADOS")

