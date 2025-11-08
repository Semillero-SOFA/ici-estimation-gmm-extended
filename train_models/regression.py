from utils import *
#=====================================================
# Crear Logger
#=====================================================
timestamp = datetime.datetime.now().strftime("%d_%H%M")
run_output_dir = os.path.join(GLOBAL_RESULTS_DIR, 'results', f"run_{timestamp}", "logs")
os.makedirs(run_output_dir, exist_ok=True)

logger = setup_logger(run_output_dir)

#=====================================================
# Parametros
#=====================================================

distancias = [0, 270]
powers = [0, 0, 9]
dist_powers = [(0,0), (270,0), (270,9)]
gaussians = [16,24,32]
covs = ["diag", "spherical"]
# TODO: Agregar más modelos si es necesario
# - Agregar MLP -> Modelo Multimodal. => informacion en Agregar optuna.
# - Agregar ejecucion segundo plano. tmux. No hup
# - Verificar que agrega sobre logger. Hacer copias de seguridad 
# - Agregar hiperparametros a el csv que se guarda
models = ["DecisionTree", "SVM", "RandomForest"]

#=====================================================
# Iterar sobre todos los escenarios (ML models only)
#=====================================================
# Compute total number of ML runs and create progress bar
total_runs = len(dist_powers) * len(gaussians) * len(covs) * len(models)
ml_pbar = tqdm(total=total_runs, desc='ML experiments', unit='run')
for distancia, power in dist_powers:
    for gaussian in gaussians:
        for cov in covs:
            database = extract_df(distancia, power, gaussian, cov)
            for model_name in models:
                #database = extract_df(distancia, power, gaussian, cov)
                    #model_name = model  # Cambiar a "DecisionTree" o "RandomForest" según se desee
                    output_dir = os.path.join(run_output_dir, f"{distancia}_{power}") # It can be saved in other specific folder
                    os.makedirs(output_dir, exist_ok=True)

                    #=====================================================
                    results= train_test_regression_model(database, model_name, logger, include_osnr=True)
                    filename = os.path.join(output_dir, f'reg_results_w.csv')

                    #This function will save the average results
                    save_regression_results(results, filename, gaussian, cov, model_name, logger)
                    # TODO: ¿Guardar parametros del modelo?
                    filename = os.path.join(output_dir, f'reg_results_w_detailed.json')
                    save_regression_results_detailed(results, filename, gaussian, cov, model_name, logger)

                    #=====================================================
                    # Guardar resultados sin OSNR
                    #=====================================================
                    results_wo =  train_test_regression_model(database, model_name, logger, include_osnr=False)
                    filename = os.path.join(output_dir, f'reg_results_wo.csv')
                    save_regression_results(results_wo, filename, gaussian, cov, model_name, logger)

                    filename = os.path.join(output_dir, f'reg_results_wo_detailed.json')
                    save_regression_results_detailed(results_wo, filename, gaussian, cov, model_name, logger)

                    log_msg = f"Completed {model_name} model for {gaussian} gaussians, {cov} covariance, distance {distancia} km and power {power} dBm."
                    logging.info(log_msg)
                    # Update progress bar after finishing this ML configuration
                    try:
                        ml_pbar.update(1)
                    except Exception:
                        # If progress update fails (rare), continue without stopping the experiment
                        pass
# Close the progress bar when done
try:
    ml_pbar.close()
except Exception:
    pass