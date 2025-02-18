from pathlib import Path

# Base directory dynamically determined relative to the package
BASE_DIR = Path(__file__).resolve().parent / "models"

# Binary quant model paths
BINARY_MODEL_PATH = BASE_DIR / "binary_TP_4000features_95to75missingness_2024.joblib"
BINARY_FEATURES_PATH = BASE_DIR / "binary_features_TP_4000features_95to75missingness_2024.txt"

# UPDATED NSAF quant model paths
MULTI_CLASS_MODEL_PATH = BASE_DIR / "TP_full_quant_update_20241216.joblib"
MULTI_CLASS_FEATURES_PATH = BASE_DIR / "TP_features_full_quant_update_20241216.txt"

UPDATED_MULTI_CLASS_MODEL_PATH = BASE_DIR / "TP_full_quant_update_20250113.joblib"
UPDATED_MULTI_CLASS_FEATURES_PATH = BASE_DIR / "TP_features_full_quant_update_20250113.txt"
