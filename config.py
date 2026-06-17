# ======================================================================
# SENTINEL-BOT
# config.py for training and evaluation configuration
# Central Configuration File
# ======================================================================

import os
import random
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# GLOBAL RANDOM SEED
# ==========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# ==========================================================
# PATHS (Portable Version)
# ==========================================================

# This finds the folder where config.py is sitting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# If your datasets are in a folder called 'processed' next to your code:
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

# If your outputs are in 'ensemble outputs' next to your code:
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "ensemble outputs")

MODEL_DIR = os.path.join(OUTPUT_BASE_DIR, "ensemble_models")
REPORT_DIR = os.path.join(OUTPUT_BASE_DIR, "ensemble_reports")
PLOT_DIR = os.path.join(OUTPUT_BASE_DIR, "ensemble_plots")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================================
# DATASETS
# ==========================================================

DATASETS = {
    "ablation_blind": "sentinel_ablation_blind.csv",
    "full": "sentinel_full.csv",
    "biometrics_only": "sentinel_biometrics_only.csv",
    "network_only": "sentinel_network_only.csv",
    "biometrics_device": "sentinel_biometrics_device.csv"
}

LABEL = "label"

# ==========================================================
# SAFE BIOMETRIC FEATURES
# ==========================================================

SAFE_FEATURES = [
    "straightness_ratio",
    "mean_velocity",
    "std_velocity",
    "direction_changes",
    "mouse_entropy",
    "acceleration_entropy",
    "avg_dwell_time",
    "std_dwell_time",
    "avg_flight_time",
    "std_flight_time",
    "key_overlap_count",
    "backspace_ratio",
    "unique_dwell_ratio",
    "unique_key_ratio"
]

# ==========================================================
# TRAIN / TEST / CALIBRATION SPLITS
# ==========================================================

TEST_SIZE = 0.15
CALIB_SIZE = 0.17

# ==========================================================
# CROSS VALIDATION
# ==========================================================

CV_FOLDS = 3

# ==========================================================
# BOOTSTRAP SETTINGS
# ==========================================================

BOOTSTRAP_ITERATIONS = 2000

# ==========================================================
# CALIBRATION SETTINGS
# ==========================================================

ECE_BINS = 10
CALIBRATION_BINS = 5

# ==========================================================
# LATENCY SETTINGS
# ==========================================================

LATENCY_WARMUP_RUNS = 20
LATENCY_MAX_SAMPLES = 100

# ==========================================================
# XGBOOST HYPERPARAMETERS
# ==========================================================

XGB_PARAMS = {
    "n_estimators": 120,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_alpha": 1.0,
    "reg_lambda": 2.0,
    "min_child_weight": 3,
    "gamma": 0.2,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
    "use_label_encoder": False,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ==========================================================
# RANDOM FOREST HYPERPARAMETERS
# ==========================================================

RF_PARAMS = {
    "n_estimators": 250,
    "max_depth": 8,
    "min_samples_split": 8,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# ==========================================================
# SVM HYPERPARAMETERS
# ==========================================================

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.5,
    "gamma": "scale",
    "probability": False,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

# ==========================================================
# META LEARNER HYPERPARAMETERS
# ==========================================================

META_LR_PARAMS = {
    "penalty": "elasticnet",
    "solver": "saga",       # 'saga' is required to use elasticnet
    "l1_ratio": 0.5,        # 50% L1, 50% L2
    "C": 0.1,
    "max_iter": 5000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE
}

# ==========================================================
# THRESHOLD OPTIMIZATION
# ==========================================================

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.80

# ==========================================================
# PLOTTING SETTINGS
# ==========================================================

PLOT_DPI = 300

ROC_FIGSIZE = (8, 6)
PR_FIGSIZE = (8, 6)
CM_FIGSIZE = (6, 5)
CALIBRATION_FIGSIZE = (6, 6)
DISTRIBUTION_FIGSIZE = (8, 5)
FEATURE_IMPORTANCE_FIGSIZE = (10, 6)
THRESHOLD_FIGSIZE = (8, 5)
DIVERSITY_FIGSIZE = (5, 4)

# ==========================================================
# PROBABILITY DISTRIBUTION HISTOGRAM
# ==========================================================

DISTRIBUTION_BINS = np.linspace(0, 1, 21)

# ==========================================================
# EXPERIMENT CONFIG EXPORT
# ==========================================================

EXPERIMENT_CONFIG_EXPORT = {
    "random_state": RANDOM_STATE,
    "cv_folds": CV_FOLDS,
    "test_size": TEST_SIZE,
    "calib_size": CALIB_SIZE,
    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
    "xgb_estimators": XGB_PARAMS["n_estimators"],
    "xgb_max_depth": XGB_PARAMS["max_depth"],
    "xgb_learning_rate": XGB_PARAMS["learning_rate"],
    "rf_estimators": RF_PARAMS["n_estimators"],
    "rf_max_depth": RF_PARAMS["max_depth"],
    "svm_C": SVM_PARAMS["C"],
    "meta_lr_C": META_LR_PARAMS["C"]
}