# ==========================================================
# SENTINEL-BOT TRAIN_TOURNAMENT.PY v3
# ==========================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ----------------------------------------------------------
# OPTIONAL LIBRARIES
# ----------------------------------------------------------
HAS_XGB = True
HAS_LGBM = True
HAS_CAT = True

try:
    from xgboost import XGBClassifier
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
except:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
except:
    HAS_CAT = False

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = r"C:\Users\aamin\Professional\VS Projects\Sentinel-Bot-Project\Sentinel-Bot-Project\train tournament outputs"

RESULT_DIR = os.path.join(BASE_DIR, "training results")
MODEL_DIR = os.path.join(BASE_DIR, "training models")
PLOT_DIR = os.path.join(BASE_DIR, "training plots")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ==========================================================
# DATASETS (NEW - MULTI-RUN SUPPORT)
# ==========================================================
DATASETS = [
    "sentinel_full.csv",
    "sentinel_biometrics_only.csv",
    "sentinel_network_only.csv",
    "sentinel_ablation_blind.csv",
    "sentinel_biometrics_device.csv"
]

DATASET_BASE_PATH = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\processed"

LABEL = "label"

# ==========================================================
# CV SETUP 
# ==========================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==========================================================
# HELPERS (MOVED OUTSIDE LOOP - FIXED)
# ==========================================================
def evaluate_predictions(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "false_positive_rate": fpr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }

# ==========================================================
# MAIN LOOP 
# ==========================================================
for dataset_name in DATASETS:

    print("\n" + "="*60)
    print(f" TRAINING ON: {dataset_name}")
    print("="*60)

    # ==========================================
    # RESET PER DATASET (CRITICAL FIX)
    # ==========================================
    results = []
    plt.figure(figsize=(10, 8))

    # ==========================================
    # LOAD DATA
    # ==========================================
    dataset_tag = dataset_name.replace(".csv", "")

    DATA_PATH = os.path.join(DATASET_BASE_PATH, dataset_name)
    df = pd.read_csv(DATA_PATH)

    if LABEL not in df.columns:
        raise Exception(f"label column missing in {dataset_name}")

    X = df.drop(columns=[LABEL])
    y = df[LABEL].astype(int)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Class distribution:", np.bincount(y))

    # ==========================================
    # TRAIN TEST SPLIT 
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==========================================
    # COLUMN TYPES 
    # ==========================================
    num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    # ==========================================
    # PREPROCESSORS 
    # ==========================================
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    numeric_only = ColumnTransformer([
        ("num", numeric_pipe, num_cols)
    ], remainder="drop")

    tree_preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols)
    ], remainder="drop")

    # ==========================================
    # MODELS (RESET EACH LOOP - CRITICAL)
    # ==========================================
    models = {}

    models["LogisticRegression"] = Pipeline([
        ("prep", numeric_only),
        ("clf", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    models["SVM"] = Pipeline([
        ("prep", numeric_only),
        ("clf", SVC(
            probability=True,
            kernel="rbf",
            class_weight="balanced",
            random_state=42
        ))
    ])

    models["RandomForest"] = Pipeline([
        ("prep", tree_preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42
        ))
    ])

    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("prep", tree_preprocessor),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42
            ))
        ])

    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("prep", tree_preprocessor),
            ("clf", LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                random_state=42
            ))
        ])

    if HAS_CAT:
        models["CatBoost"] = Pipeline([
            ("prep", tree_preprocessor),
            ("clf", CatBoostClassifier(
                verbose=0,
                iterations=400,
                learning_rate=0.05,
                depth=6,
                random_state=42
            ))
        ])

    # ==========================================
    # TOURNAMENT LOOP 
    # ==========================================
    for name, model in models.items():

        print(f"\nTraining: {name}")

        # -------------------------------
        # Cross-validated predictions
        # -------------------------------
        cv_prob = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        # Optimal threshold
        fpr_cv, tpr_cv, thresholds = roc_curve(y_train, cv_prob)
        best_idx = np.argmax(tpr_cv - fpr_cv)
        best_thresh = thresholds[best_idx]

        cv_pred = (cv_prob >= best_thresh).astype(int)
        cv_metrics = evaluate_predictions(y_train, cv_pred, cv_prob)

        # -------------------------------
        # Train full model
        # -------------------------------
        model.fit(X_train, y_train)

        # SAVE MODEL (NOW DATASET-SPECIFIC)
        joblib.dump(
            model,
            os.path.join(MODEL_DIR, f"{dataset_tag}_{name}.pkl")
        )

        # -------------------------------
        # Test evaluation
        # -------------------------------
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= best_thresh).astype(int)

        test_metrics = evaluate_predictions(y_test, test_pred, test_prob)

        # -------------------------------
        # ROC Curve
        # -------------------------------
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={test_metrics['roc_auc']:.3f})")

        # -------------------------------
        # Feature Importance (FIXED)
        # -------------------------------
        try:
            clf = model.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_

                imp_df = pd.DataFrame({
                    "feature": num_cols,
                    "importance": importances
                }).sort_values(by="importance", ascending=False)

                imp_df.to_csv(
                    os.path.join(RESULT_DIR, f"{dataset_tag}_{name}_feature_importance.csv"),
                    index=False
                )
        except:
            pass

        # -------------------------------
        # Store results
        # -------------------------------
        results.append({
            "model": name,
            "cv_accuracy": cv_metrics["accuracy"],
            "cv_precision": cv_metrics["precision"],
            "cv_recall": cv_metrics["recall"],
            "cv_f1": cv_metrics["f1"],
            "cv_auc": cv_metrics["roc_auc"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auc": test_metrics["roc_auc"],
            "test_fpr": test_metrics["false_positive_rate"],
            "tp": test_metrics["tp"],
            "fp": test_metrics["fp"],
            "tn": test_metrics["tn"],
            "fn": test_metrics["fn"]
        })

    # ==========================================
    # SAVE ROC (DATASET-SPECIFIC)
    # ==========================================
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve - {dataset_tag}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"roc_{dataset_tag}.png"), dpi=300)
    plt.close()

    # ==========================================
    # LEADERBOARD (DATASET-SPECIFIC)
    # ==========================================
    res = pd.DataFrame(results)
    res = res.sort_values(
        by=["test_f1", "test_auc", "test_recall"],
        ascending=False
    )

    res.to_csv(
        os.path.join(RESULT_DIR, f"leaderboard_{dataset_tag}.csv"),
        index=False
    )

    # ==========================================
    # DISPLAY
    # ==========================================
    print("\n=================================================")
    print(f" COMPLETED: {dataset_tag}")
    print("=================================================")
    print(res[[
        "model",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "test_fpr"
    ]].to_string(index=False))

print("\nALL DATASETS COMPLETED")
print("Check training results folder.")