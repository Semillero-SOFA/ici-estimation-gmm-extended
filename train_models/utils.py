import numpy as np
import pandas as pd


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
import datetime
import json
import logging
import tqdm
import shutil

INTERVAL_LIST = {"2": [35.2],
                 "3": [31.5, 35.2],
                 "4": [30.0, 32.5, 35.2],
                 "5": [30.0, 31.5, 33.5, 35.2]}

# Grid de parámetros para cada modelo
PARAMS_GRID_CLASSIFICATION = {
    'DecisionTree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}


PARAMS_GRID_REGRESSION = {
    'DecisionTree': {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['rbf']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
}


GLOBAL_RESULTS_DIR = "D:/Semillero SOFA/gmm_32_definitivo"
DATASETS_DIR = f"{GLOBAL_RESULTS_DIR}/new_models"

# Cargar datos
def extract_df(dis, power, gauss, cov):
    sub_dir = f"{dis}km{power}dBm/{gauss}_gaussians"
    df = pd.read_csv(f"{DATASETS_DIR}/{sub_dir}/models32_gmm_{cov}.csv")
    return df

def setup_logger(name: str) -> logging.Logger:
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler(f"{name}.log")
        ]
    )
    logger = logging.getLogger(name)
    return logger


# Convertir spacing continuo a clases discretas
def spacing_to_class(spacing_value, interval_list):
    """
    Convierte un valor de spacing continuo en una clase discreta
    basado en los intervalos definidos.
    
    Args:
        spacing_value (float): Valor de spacing a clasificar
        interval_list (list): Lista de límites de intervalos ordenados
    
    Returns:
        int: Clase asignada (0, 1, 2, ..., n_classes-1)
    """
    for i, threshold in enumerate(interval_list):
        if spacing_value <= threshold:
            return i
    return len(interval_list)  # Última clase

def transform_to_classification(data, n_classes="2"):
    """
    Transforma el dataset de regresión a clasificación
    
    Args:
        data (pd.DataFrame): Dataset original
        n_classes (str): Número de clases ("2", "3", "4", "5")
    
    Returns:
        pd.DataFrame: Dataset con clases en lugar de valores continuos
    """
    data = data.copy()
    
    # Convertir spacing a numérico si es string
    if data["spacing"].dtype == 'object':
        data["spacing"] = data["spacing"].str.replace('GHz', '').astype(float)
    
    # Obtener intervalos para el número de clases especificado
    interval_list = INTERVAL_LIST[n_classes]
    
    # Convertir spacing a clases
    data["spacing_class"] = data["spacing"].apply(
        lambda x: spacing_to_class(x, interval_list)
    )
    
    return data

# Calcular metricas de clasificación
def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    Calcula métricas de clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        average: Tipo de promedio para métricas multiclase
    
    Returns:
        tuple: (accuracy, precision, recall, f1_score)
    """
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, precision, recall, f1

# Acumular resultados
def accumulate_classification_results(results, tipo="train", acc=None, precision=None, recall=None, f1=None):
    results["acc"][tipo].append(acc)
    results["precision"][tipo].append(precision)
    results["recall"][tipo].append(recall)
    results["f1_score"][tipo].append(f1)

def extract_X_y_classification(data, include_osnr=True):
    """
    Extrae features y target para clasificación
    
    Args:
        data (pd.DataFrame): Dataset con columna 'spacing_class'
        include_osnr (bool): Si incluir OSNR como feature
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target (clases)
    """
    data = data.copy()
    
    # Convertir OSNR si es string
    if data["osnr"].dtype == 'object':
        data["osnr"] = data["osnr"].str.replace('dB', '').astype(float)
    
    # Seleccionar features
    if include_osnr:
        X = data.drop(["spacing", "spacing_class"], axis=1)
    else:
        X = data.drop(["spacing", "spacing_class", "osnr"], axis=1)
    
    # Target son las clases
    y = data["spacing_class"]
    
    return X, y

def initialize_classification_results():
    results = {}
    results["model_params"] = {}
    results["y_test"] = []
    results["y_pred_test"] = []
    results["acc"] = {"train": [], "test": []}
    results["precision"] = {"train": [], "test": []}
    results["recall"] = {"train": [], "test": []}
    results["f1_score"] = {"train": [], "test": []}
    return results

def choose_classification_model(model_name):
    """
    Selecciona el modelo de clasificación
    
    Args:
        model_name (str): Nombre del modelo
    
    Returns:
        Modelo de sklearn sin entrenar
    """
    if model_name == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(kernel='rbf', random_state=42)
    elif model_name == "RandomForest":
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError(f"Modelo {model_name} no soportado.")
    return model

def save_classification_results(results, path_file, gaussian, covariance, model, n_classes):
    """
    Guarda los resultados de clasificación en un CSV
    """

    if os.path.exists(path_file):
        # Save backup
        backup_path = path_file + ".bak"
        shutil.copy2(path_file, backup_path)
        current_results = pd.read_csv(path_file)
    else:
        current_results = pd.DataFrame()
        
    dict_results = {}
    metrics = ['acc', 'precision', 'recall', 'f1_score']
    for key, value in results.items():
        if key in metrics:
            dict_results[f"{key}_train"] = np.mean(value['train'])
            dict_results[f"{key}_test"] = np.mean(value['test'])
            dict_results[f"{key}_std_train"] = np.std(value['train'])
            dict_results[f"{key}_std_test"] = np.std(value['test'])
    
    dict_results['gaussian'] = gaussian
    dict_results['n_classes'] = n_classes
    dict_results['covariance'] = covariance
    dict_results['model_name'] = model

    df_results = pd.DataFrame([dict_results])
      
    current_results = pd.concat([current_results, df_results], ignore_index=True)
    current_results.to_csv(path_file, index=False)
def save_classification_results_detailed(results, path_file, gaussian, covariance, model, n_classes,logger):
    """
    Save raw results (without averaging) for each fold)
    It also include the best params for each fold.
    Save in a JSON file. 
    It will save in a specific folder according to the gaussians and covariance type.
    TODO: Save in a compressed format and backup if it's necessary

        Args:
            results (dict): The results dictionary containing metrics and model parameters.
            path_file (str): The file path to save the detailed results.
            gaussian (int): The number of gaussians used in the model.
            covariance (str): The type of covariance used in the model.
            model (str): The name of the regression model used.
            n_classes (str): The number of classes used in the classification.

    """


    # Check if the file exists
    if not os.path.exists(path_file):
        dict_results = {
            str(gaussian): {
                covariance:{
                    str(n_classes): {
                        model: {
                            'metrics': {
                                'acc': results['acc'],
                                'precision': results['precision'],
                                'recall': results['recall'],
                                'f1_score': results['f1_score']
                            },
                            'model_params': results['model_params'],
                            }
                    }
                }
            }
        }
    else:
        # If the file exists, save backup and load the current results
        backup_path = path_file + ".bak"
        shutil.copy2(path_file, backup_path)

        with open(path_file, "r") as f:
            dict_results = json.load(f)
        # Update the results for the specific fold
        # If gaussian type does not exist, create it
        if str(gaussian) not in dict_results:
            dict_results[str(gaussian)] = {}
        # If covariance type does not exist, create it
        if covariance not in dict_results[str(gaussian)]:
            dict_results[str(gaussian)][covariance] = {}
        # If n_classes type does not exist, create it
        if str(n_classes) not in dict_results[str(gaussian)][covariance]:
            dict_results[str(gaussian)][covariance][str(n_classes)] = {}
        
        # Update the results for the specific fold
        dict_results[str(gaussian)][covariance][str(n_classes)][model] = {
            'metrics': {
                    'acc': results['acc'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                },
            'model_params': results['model_params'],
        }

    # Save the updated results
    with open(path_file, "w") as f:
        json.dump(dict_results, f, indent=4)
    log_msg = f"Saved detailed classification {gaussian} gaussians, {covariance} covariance, model {model} results to {path_file}"
    logger.info(log_msg)


def train_test_classification_model(data, model_name, logger, n_classes="2", include_osnr=True):
    """
    Entrena un modelo de clasificación con validación cruzada estratificada
    
    Args:
        data (pd.DataFrame): Dataset original
        model_name (str): Nombre del modelo ('DecisionTree', 'SVM', 'RandomForest')
        n_classes (str): Número de clases ("2", "3", "4", "5")
        include_osnr (bool): Si incluir OSNR como feature
    
    Returns:
        dict: Resultados del entrenamiento y evaluación
    """
    results = initialize_classification_results()
    
    # 1. Transformar datos a problema de clasificación
    data_class = transform_to_classification(data, n_classes=n_classes)
    
    # 2. Extraer features y target
    X, y = extract_X_y_classification(data_class, include_osnr=include_osnr)
    
    # 3. Configurar validación cruzada estratificada
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 4. Extraer parámetros para el modelo
    params = PARAMS_GRID_CLASSIFICATION[model_name]
    
    # 5. Iterar sobre los folds
    fold = 1
    for index, (train_index, test_index) in enumerate(skf.split(X, y)):
        print(f"Procesando fold {fold}/{n_splits}...")
        
        # Dividir datos
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Escalar features (fit solo con train)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Crear modelo con GridSearchCV
        base_model = choose_classification_model(model_name)
        
        # Usar f1_weighted para GridSearch en problemas multiclase
        model = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=3,
            n_jobs=-1,
            scoring='f1_weighted',
            verbose=0
        )
        
        # Entrenar modelo
        model.fit(X_train_scaled, y_train)
        
        logger.info(f"Mejores parámetros fold {fold}: {model.best_params_}")
        
        # Predicciones
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calcular métricas
        train_acc, train_prec, train_rec, train_f1 = calculate_classification_metrics(
            y_train, y_pred_train
        )
        test_acc, test_prec, test_rec, test_f1 = calculate_classification_metrics(
            y_test, y_pred_test
        )
        
        # Acumular resultados
        accumulate_classification_results(
            results, "train", train_acc, train_prec, train_rec, train_f1
        )
        accumulate_classification_results(
            results, "test", test_acc, test_prec, test_rec, test_f1
        )
        
        # Guardar predicciones
        results["y_test"].extend(y_test)
        results["y_pred_test"].extend(y_pred_test)
        results["model_params"][index] = model.best_params_
        
        fold += 1
    
    # Imprimir resumen
    # print(f"\n{'='*50}")
    # print(f"Resultados promedio - {model_name} - {n_classes} clases")
    # print(f"{'='*50}")
    # print(f"Accuracy Train: {np.mean(results['acc']['train']):.4f} ± {np.std(results['acc']['train']):.4f}")
    # print(f"Accuracy Test:  {np.mean(results['acc']['test']):.4f} ± {np.std(results['acc']['test']):.4f}")
    # print(f"F1-Score Train: {np.mean(results['f1_score']['train']):.4f} ± {np.std(results['f1_score']['train']):.4f}")
    # print(f"F1-Score Test:  {np.mean(results['f1_score']['test']):.4f} ± {np.std(results['f1_score']['test']):.4f}")
    # print(f"Recall Train:   {np.mean(results['recall']['train']):.4f} ± {np.std(results['recall']['train']):.4f}")
    # print(f"Recall Test:    {np.mean(results['recall']['test']):.4f} ± {np.std(results['recall']['test']):.4f}")
    # print(f"{'='*50}\n")
    
    return results


#========================================================================================================================
# REGRESSION MODELS
#========================================================================================================================
# Calcular metricas
def calculate_regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae
# Acumular resultados
def accumulate_regression_results(results, tipo = "train", r2=None, rmse=None, mae=None):
    results["r2"][tipo].append(r2)
    results["rmse"][tipo].append(rmse)
    results["mae"][tipo].append(mae)

def extract_X_y_regression(data, include_osnr=True):
    """
    Extracst features and target variable from the dataset.

    Args:
        data (pd.DataFrame): The input dataset containing features and target.
        include_osnr (bool): Whether to include the 'osnr' feature in X.
    Returns:
        X (pd.DataFrame): The feature set.
        y (pd.Series): The target variable (spacing).
    """
    # 1. Preparar datos
    data = data.copy()
    if data["osnr"].dtype == 'object':
        data["osnr"] = data["osnr"].str.replace('dB', '').astype(float)
    # Preparar datos
    if include_osnr:
        X = data.drop(["spacing"], axis=1)
    else:
        # Excluir tanto spacing como osnr
        X = data.drop(["spacing", "osnr"], axis=1)
    # Si spacing es categórico (ej: "29GHz"), convertir a numérico
    y = data["spacing"].copy()
    if y.dtype == 'object':
        # Extraer números de strings como "29GHz"
        y = y.str.replace('GHz', '').astype(float)
    return X, y
def initialize_regression_results():
    results = {}
    # Modelo
    results["model_params"] = {}
    results["y_test"] = []
    results["y_pred_test"] = []
    results["mae"] = {"train": [], "test": []}
    results["r2"] = {"train": [], "test": []}
    results["rmse"] = {"train": [], "test": []}
    return results

# TODO: Agregar más modelos si es necesario
def choose_model_regression(model_name):
    """
    Choose and return a regression model.
    
    Note: For MLP models with Optuna optimization, use train_test_mlp_optuna()
    from train_test_optuna.py instead of this function.
    """
    if model_name == "DecisionTree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "SVM":
        model = SVR(kernel='rbf')
    elif model_name == "RandomForest":
        model = RandomForestRegressor(random_state=42)
    else:
        raise ValueError(f"Modelo {model_name} no soportado.")
    return model

def save_regression_results(results, path_file, gaussian, covariance, model, logger):
   
    dict_results = {}
    metrics = ['mae', 'r2', 'rmse']
    for key, value in results.items(): # Iterate over the metrics
        if key in metrics:
            dict_results[f"{key}_train"] = np.mean(value['train'])
            dict_results[f"{key}_test"] = np.mean(value['test'])
            dict_results[f"{key}_std_train"] = np.std(value['train'])
            dict_results[f"{key}_std_test"] = np.std(value['test'])
    dict_results['gaussian'] = gaussian
    dict_results['covariance'] = covariance
    dict_results['model_name'] = model

    df_results = pd.DataFrame([dict_results])
    if os.path.exists(path_file):
        current_results = pd.read_csv(path_file)
    else:
        current_results = pd.DataFrame()
        
    current_results = pd.concat([current_results, df_results], ignore_index=True)
    current_results.to_csv(path_file, index=False)
    log_msg = f"Saved regression {gaussian} gaussians, {covariance} covariance, model {model} results to {path_file}"
    logger.info(log_msg)

def save_regression_results_detailed(results, path_file, gaussian, covariance, model, logger):
    """
    Save raw results (without averaging) for each fold)
    It also include the best params for each fold.
    Save in a JSON file. 
    It will save in a specific folder according to the gaussians and covariance type.
    TODO: Save in a compressed format and backup if it's necessary

        Args:
            results (dict): The results dictionary containing metrics and model parameters.
            path_file (str): The file path to save the detailed results.
            gaussian (int): The number of gaussians used in the model.
            covariance (str): The type of covariance used in the model.
            model (str): The name of the regression model used.
    """
    

    # Check if the file exists
    if not os.path.exists(path_file):
        dict_results = {
            str(gaussian): {
                covariance:{
                    model: {
                    'metrics': {
                        'mae': results['mae'],
                        'r2': results['r2'],
                        'rmse': results['rmse'],
                    },
                    'model_params': results['model_params'],
                    }
                }
            }
        }
    else:
        # If the file exists, save back up and load the existing results
        backup_path = path_file + ".bak"
        shutil.copy2(path_file, backup_path)

        with open(path_file, "r") as f:
            dict_results = json.load(f)
        # Update the results for the specific fold
        # If gaussian type does not exist, create it
        if str(gaussian) not in dict_results:
            dict_results[str(gaussian)] = {}
        # If covariance type does not exist, create it
        if covariance not in dict_results[str(gaussian)]:
            dict_results[str(gaussian)][covariance] = {}
        # Update the results for the specific fold
        dict_results[str(gaussian)][covariance][model] = {
            'metrics': {
                'mae': results['mae'],
                'r2': results['r2'],
                'rmse': results['rmse'],
            },
            'model_params': results['model_params'],
        }

    # Save the updated results
    with open(path_file, "w") as f:
        json.dump(dict_results, f, indent=4)
    log_msg = f"Saved detailed regression {gaussian} gaussians, {covariance} covariance, model {model} results to {path_file}"
    logger.info(log_msg)



def train_test_regression_model(data, model_name, logger, include_osnr=True):
    """
    Entrena un modelo de regresión Decision Tree para predecir spacing
    
    Args:
        data: DataFrame con los datos
        model_name: Nombre del modelo a utilizar
        include_osnr: Si incluir OSNR como feature o no
    
    Returns:
        dict: métricas del modelo y modelo entrenado
    """
    results = initialize_regression_results()
    # 1. Preparar datos
    X, y = extract_X_y_regression(data, include_osnr=include_osnr)
    #X_scaled = scale_features(X, type="standard")
    n_splits = 5
    

    # TODO: Accoding to GPT: StratifiedKFold stratification: 
    # using LabelEncoder().fit_transform(y) on continuous spacing is not appropriate — 
    # it uses unique continuous values and will not produce useful strata (or will fail if few repeats).
    #  You should bin y (e.g. pd.qcut or pd.cut) before stratifying.
    
    # Convertir y a bins para estratificación
    y_bins = LabelEncoder().fit_transform(y)
    # Crear objeto para estratificación
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    #sin estratificar:
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    #for train_index, test_index in kf.split(X_scaled, y):

    # Extraer parametros para el modelo
    params = PARAMS_GRID_REGRESSION[model_name]
    for index, (train_index, test_index) in enumerate(skf.split(X, y_bins)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scaler = StandardScaler().fit(X_train) # Ajustar scaler solo con datos de entrenamiento
        # Escalar features utilizando solo datos de train
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        # 3. Definir modelo


        model = choose_model_regression(model_name)

        model = GridSearchCV(estimator=model,
                             param_grid=params, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
        
        #log(best_model.best_params_.values()) # Imprimir mejores parámetros
        # 2 crosvalidacion - 1.
        ###################### ttttttt
        ################ tttt
        # 4. Entrenar modelo
        model.fit(X_train, y_train)
        logger.info(str(model.best_params_))
        # 5. Evaluar modelo
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        # Métricas
        train_r2, train_rmse, train_mae = calculate_regression_metrics(y_train, y_pred_train)
        test_r2, test_rmse, test_mae = calculate_regression_metrics(y_test, y_pred_test)
        # Resultados
        # Acumular resultados
        accumulate_regression_results(results, "train", train_r2, train_rmse, train_mae)
        accumulate_regression_results(results, "test", test_r2, test_rmse, test_mae)
        results["y_test"].extend(y_test)
        results["y_pred_test"].extend(y_pred_test)
        results["model_params"][index] = model.best_params_  # Save best_params in each fold

    return results