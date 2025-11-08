# Train Models

This folder contains scripts and notebooks for training classification and regression models for ICI estimation using GMM features.

## Files

- **`classification.py`**: Python script to train classification models (Decision Tree, SVM, Random Forest) for multiple GMM configurations
- **`regression.py`**: Python script to train regression models (Decision Tree, SVM, Random Forest) for ICI power estimation
- **`utils.py`**: Utility functions including model training, data extraction, logging, and result saving
- **`classification_model.ipynb`**: Jupyter notebook version for classification experiments
- **`regression_model.ipynb`**: Jupyter notebook version for regression experiments

## Data Used

The scripts use preprocessed GMM feature datasets located in the path defined in `utils.py`:
- Default location: `D:/Semillero SOFA/gmm_32_definitivo/new_models`
- Features extracted from different configurations: distances (0, 270 km), powers (0, 9 dBm), Gaussian components (16, 24, 32), and covariance types (diag, spherical)

## Folder to Change

**Before running, update in `utils.py`:**
```python
GLOBAL_RESULTS_DIR = "YOUR_PATH_HERE"  # Line 71
```
This path determines where:
- Input datasets are read from (`{GLOBAL_RESULTS_DIR}/new_models`)
- Output results are saved to (`{GLOBAL_RESULTS_DIR}/results/run_TIMESTAMP`)

## Quick Instructions

1. **Update path**: Edit `GLOBAL_RESULTS_DIR` in `utils.py` to point to your data directory
2. **Run classification**: `python classification.py`
3. **Run regression**: `python regression.py`
4. **Check results**: Look in `{GLOBAL_RESULTS_DIR}/results/run_TIMESTAMP/logs/`

Results include CSV files with averaged metrics and JSON files with detailed fold-by-fold results.
