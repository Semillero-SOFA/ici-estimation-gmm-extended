# Train Models

This folder contains scripts and notebooks for training classification and regression models for ICI estimation using GMM features.

## Files

- **`classification.py`**: Python script to train classification models (Decision Tree, SVM, Random Forest, XGBoost)
- **`regression.py`**: Python script to train regression models (Decision Tree, SVM, Random Forest, XGBoost)
- **`utils.py`**: Utility functions including model training, data extraction, logging, and result saving
- **`classification_model.ipynb`**: Jupyter notebook version for classification experiments
- **`regression_model.ipynb`**: Jupyter notebook version for regression experiments

## Recent Updates

### Training Modes
Both classification and regression scripts now support two training modes:

**Single Mode (default)**: Trains one model type across all K-folds with GridSearchCV optimization. Returns averaged metrics across folds.

**All Mode**: Trains different models in each K-fold (e.g., DecisionTree in fold 1, SVM in fold 2, etc.). Accumulates all predictions and calculates global metrics.

### Command-Line Arguments
- `--mode`: Choose training mode (`single` or `all`)
- `--results_dir`: Specify global results directory path
- `--datasets_dir`: Specify datasets directory path

### New Functions
- `train_test_classification_all_predictions()`: Classification with multiple models per run
- `train_test_regression_all_models()`: Regression with multiple models per run
- `run_classification_single_model()`: Iterator for single-model classification
- `run_classification_all_predictions()`: Iterator for multi-model classification
- `run_regression_single_model()`: Iterator for single-model regression
- `run_regression_all_models()`: Iterator for multi-model regression

## Data Configuration

Default dataset location: `D:/Semillero SOFA/gmm_32_definitivo/new_models` (configurable via `--datasets_dir`)

Features extracted from configurations:
- Distances: 0, 270 km
- Powers: 0, 9 dBm
- Gaussian components: 16, 24, 32, 40, 48, 56, 64
- Covariance types: diag, spherical

## Execution

### Classification

**Single model mode (default)**:
```bash
python classification.py --mode single --results_dir "D:/Semillero SOFA/gmm_32_definitivo" --datasets_dir "D:/Semillero SOFA/gmm_32_definitivo/new_models"
```

**Multiple models mode**:
```bash
python classification.py --mode all --results_dir "D:/Semillero SOFA/gmm_32_definitivo" --datasets_dir "D:/Semillero SOFA/gmm_32_definitivo/new_models"
```

### Regression

**Single model mode (default)**:
```bash
python regression.py --mode single --results_dir "D:/Semillero SOFA/gmm_32_definitivo" --datasets_dir "D:/Semillero SOFA/gmm_32_definitivo/new_models"
```

**Multiple models mode**:
```bash
python regression.py --mode all --results_dir "D:/Semillero SOFA/gmm_32_definitivo" --datasets_dir "D:/Semillero SOFA/gmm_32_definitivo/new_models"
```

### Using default paths
If using the default paths, you can omit the directory arguments:
```bash
python classification.py --mode single
python regression.py --mode all
```

### Custom dataset location
To use a different dataset directory:
```bash
python classification.py --datasets_dir "path/to/your/datasets"
python regression.py --datasets_dir "path/to/your/datasets"
```

## Output Structure

### Single Mode
- CSV: `{results_dir}/results_{classification|regression}/{dist}_{power}/run_{timestamp}/{task}_results_{w|wo}.csv`
- JSON: `{results_dir}/results_{classification|regression}/{dist}_{power}/run_{timestamp}/{task}_results_detailed_{w|wo}.json`

### All Mode
- JSON: `{results_dir}/results_{classification|regression}_all/{dist}_{power}/run_{timestamp}/{task}_results_all_{w|wo}.json`

Results include:
- `w`: With OSNR feature
- `wo`: Without OSNR feature
- Averaged metrics (single mode) or global metrics (all mode)
- Model parameters and predictions

