# ==========================================================
# SENTINEL-BOT
# TRAIN_ENSEMBLE_v0.6.py
# HYBRID STACKING VERSION (without optimization)
# ==========================================================
#
# PURPOSE:
# Hybrid stacking ensemble for behavioral bot detection.
#
# ==========================================================
#
# BASE MODELS:
# - XGBoost (Lightly Regularized)
# - Random Forest (Moderately Stabilized)
# - SVM RBF (Calibrated)
#
# META LEARNER:
# - Logistic Regression (L2-Regularized)
#
# HYBRID STACKING:
# Meta learner receives:
#   1. Base model probabilities
#   2. Safe behavioral biometric features
#
# KEY FEATURES:
# - Strict leakage-free stacking
# - OOF predictions
# - Meta-column alignment lock
# - Threshold optimization
# - Threshold clipping
# - Safe feature injection
# - Double-scaling protection
# - Training-only imbalance handling
# - Thesis-grade evaluation pipeline
#
# ==========================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# ==========================================================
# PATHS
# ==========================================================

PROCESSED_DIR = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\processed"

OUTPUT_BASE_DIR = r"C:\Users\aamin\Professional\VS Projects\Sentinel-Bot-Project\Sentinel-Bot-Project\ensemble outputs"

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
# RESULTS
# ==========================================================

leaderboard = []

overall_start_time = time.time()

# ==========================================================
# MAIN LOOP
# ==========================================================

for dataset_name, filename in DATASETS.items():

    dataset_start_time = time.time()

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 80)

    path = os.path.join(PROCESSED_DIR, filename)

    # ======================================================
    # FILE CHECK
    # ======================================================

    if not os.path.exists(path):
        print(f"[ERROR] Missing file: {filename}")
        continue

    df = pd.read_csv(path)

    # ======================================================
    # LABEL CHECK
    # ======================================================

    if LABEL not in df.columns:
        print(f"[ERROR] Missing label column in {filename}")
        continue

    # ======================================================
    # BASIC CLEANING
    # ======================================================

    X = df.drop(columns=[LABEL], errors="ignore")
    y = df[LABEL].astype(int)

    # ======================================================
    # SAFE FEATURE FILTERING
    # ======================================================

    existing_safe_features = [
        f for f in SAFE_FEATURES
        if f in X.columns
    ]

    print("\nSafe Features Used:")
    print(existing_safe_features)

    # ======================================================
    # TRAIN / TEST SPLIT
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # ======================================================
    # TRAINING-ONLY CLASS IMBALANCE
    # (Leakage-Free)
    # ======================================================

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    scale_pos_weight = (
        neg_count / pos_count
        if pos_count > 0 else 1
    )

    # ======================================================
    # NUMERIC FEATURE DETECTION
    # ======================================================

    numeric_cols = X_train.select_dtypes(
        include="number"
    ).columns.tolist()

    print(f"\nTotal Features Used: {len(numeric_cols)}")

    # ======================================================
    # PREPROCESSOR
    # ======================================================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                numeric_cols
            )
        ],
        remainder="drop"
    )

    # ======================================================
    # BASE MODELS
    # ======================================================
    #
    # IMPORTANT:
    # v0.6 uses LIGHT regularization only.
    #
    # This avoids:
    # - severe overfitting on 458 rows
    # - premature hyperparameter optimization
    #
    # while still stabilizing the ensemble slightly.
    #
    # ======================================================

    xgb_model = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=140,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,

            reg_alpha=0.3,
            reg_lambda=1.2,

            min_child_weight=2,

            scale_pos_weight=scale_pos_weight,

            eval_metric="logloss",

            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    rf_model = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=220,
            max_depth=10,

            min_samples_split=5,
            min_samples_leaf=2,

            max_features="sqrt",

            class_weight="balanced_subsample",

            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    svm_base = SVC(
        kernel="rbf",
        C=2,
        gamma="scale",

        class_weight="balanced",

        random_state=RANDOM_STATE
    )

    svm_calibrated = CalibratedClassifierCV(
        estimator=svm_base,
        method="sigmoid",
        cv=3
    )

    svm_model = Pipeline([
        ("prep", preprocessor),
        ("clf", svm_calibrated)
    ])

    # ======================================================
    # PHASE A:
    # OOF PREDICTIONS
    # ======================================================

    print("\nPhase A: Generating OOF Predictions...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    xgb_oof = cross_val_predict(
        xgb_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    rf_oof = cross_val_predict(
        rf_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    svm_oof = cross_val_predict(
        svm_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    # ======================================================
    # PHASE B:
    # TRAIN BASE MODELS FULLY
    # ======================================================

    print("Phase B: Training Base Models...")

    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    # ======================================================
    # PHASE C:
    # TEST PROBABILITIES
    # ======================================================

    print("Phase C: Generating Test Probabilities...")

    xgb_test_prob = xgb_model.predict_proba(X_test)[:, 1]
    rf_test_prob = rf_model.predict_proba(X_test)[:, 1]
    svm_test_prob = svm_model.predict_proba(X_test)[:, 1]

    # ======================================================
    # META DATASET
    # ======================================================

    meta_train = pd.DataFrame({
        "xgb_prob": xgb_oof,
        "rf_prob": rf_oof,
        "svm_prob": svm_oof
    })

    meta_test = pd.DataFrame({
        "xgb_prob": xgb_test_prob,
        "rf_prob": rf_test_prob,
        "svm_prob": svm_test_prob
    })

    # ======================================================
    # SAFE FEATURE INJECTION
    # ======================================================

    for feature in existing_safe_features:

        meta_train[feature] = (
            X_train[feature]
            .reset_index(drop=True)
        )

        meta_test[feature] = (
            X_test[feature]
            .reset_index(drop=True)
        )

    # ======================================================
    # META COLUMN ALIGNMENT LOCK
    # ======================================================

    prob_cols = [
        "xgb_prob",
        "rf_prob",
        "svm_prob"
    ]

    FINAL_META_COLS = (
        prob_cols + existing_safe_features
    )

    meta_train = meta_train[FINAL_META_COLS]
    meta_test = meta_test[FINAL_META_COLS]

    # ======================================================
    # META PREPROCESSOR
    #
    # IMPORTANT:
    # Safe features are RAW here.
    #
    # They MUST be scaled again before
    # Logistic Regression.
    #
    # Probabilities are passed unchanged.
    # ======================================================

    meta_preprocessor = ColumnTransformer(
        transformers=[
            (
                "scale_features",
                StandardScaler(),
                existing_safe_features
            ),
            (
                "pass_probs",
                "passthrough",
                prob_cols
            )
        ]
    )

    # ======================================================
    # META LEARNER
    # ======================================================

    print("\nTraining Meta Learner...")

    meta_model = Pipeline([
        ("prep", meta_preprocessor),
        ("lr", LogisticRegression(
            C=0.5,
            penalty="l2",

            max_iter=5000,

            class_weight="balanced",

            random_state=RANDOM_STATE
        ))
    ])

    # ======================================================
    # THRESHOLD OPTIMIZATION
    # ======================================================

    print("\nOptimizing Meta-Learner Threshold...")

    meta_cv_prob = cross_val_predict(
        meta_model,
        meta_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1
    )[:, 1]

    fpr_cv, tpr_cv, thresholds = roc_curve(
        y_train,
        meta_cv_prob
    )

    best_idx = np.argmax(tpr_cv - fpr_cv)

    best_threshold = thresholds[best_idx]

    # ======================================================
    # THRESHOLD STABILITY PROTECTION
    # ======================================================

    best_threshold = np.clip(
        best_threshold,
        0.20,
        0.80
    )

    print(f"Optimal Threshold: {best_threshold:.4f}")

    # ======================================================
    # FINAL META TRAINING
    # ======================================================

    meta_model.fit(meta_train, y_train)

    # ======================================================
    # FINAL PREDICTIONS
    # ======================================================

    final_probs = meta_model.predict_proba(
        meta_test
    )[:, 1]

    final_preds = (
        final_probs >= best_threshold
    ).astype(int)

    # ======================================================
    # METRICS
    # ======================================================

    acc = accuracy_score(
        y_test,
        final_preds
    )

    prec = precision_score(
        y_test,
        final_preds,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        final_preds,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        final_preds,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        final_probs
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        final_preds
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\nRESULTS")
    print("-" * 50)

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"FPR      : {fpr:.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        final_preds,
        zero_division=0
    ))

    # ======================================================
    # SAVE METRICS CSV
    # ======================================================

    metrics_df = pd.DataFrame([{
        "dataset": dataset_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "false_positive_rate": fpr,
        "threshold": best_threshold,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }])

    metrics_path = os.path.join(
        REPORT_DIR,
        f"{dataset_name}_metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # ======================================================
    # ROC CURVE
    # ======================================================

    fpr_plot, tpr_plot, _ = roc_curve(
        y_test,
        final_probs
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr_plot,
        tpr_plot,
        label=f"Hybrid Stack (AUC={auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"Hybrid Stacking ROC Curve - {dataset_name.upper()}"
    )

    plt.legend()
    plt.tight_layout()

    roc_path = os.path.join(
        PLOT_DIR,
        f"ensemble_roc_{dataset_name}.png"
    )

    plt.savefig(
        roc_path,
        dpi=300
    )

    plt.close()

    print(f"Saved ROC Curve: {roc_path}")

    # ======================================================
    # SAVE MODEL
    # ======================================================

    model_path = os.path.join(
        MODEL_DIR,
        f"{dataset_name}_Hybrid_Stacking_v0.6.pkl"
    )

    joblib.dump({
        "meta_model": meta_model,
        "threshold": best_threshold,

        "base_models": {
            "xgb": xgb_model,
            "rf": rf_model,
            "svm": svm_model
        },

        "safe_features": existing_safe_features,

        "dataset": dataset_name,

        "meta_columns": FINAL_META_COLS

    }, model_path)

    print(f"\nSaved Model: {model_path}")

    # ======================================================
    # LEADERBOARD
    # ======================================================

    dataset_elapsed = (
        time.time() - dataset_start_time
    )

    leaderboard.append({
        "dataset": dataset_name,
        "model": "Hybrid_Stacking_v0.6",

        "optimal_threshold": round(best_threshold, 4),

        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),

        "false_positive_rate": round(fpr, 4),

        "training_time_sec": round(dataset_elapsed, 2)
    })

# ==========================================================
# SAVE LEADERBOARD
# ==========================================================

leaderboard_df = pd.DataFrame(
    leaderboard
).sort_values(
    by="accuracy",
    ascending=False
)

leaderboard_path = os.path.join(
    REPORT_DIR,
    "ensemble_leaderboard.csv"
)

leaderboard_df.to_csv(
    leaderboard_path,
    index=False
)

# ==========================================================
# FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 80)
print("FINAL ENSEMBLE LEADERBOARD")
print("=" * 80)

print(leaderboard_df.to_string(index=False))

print("\nSaved Leaderboard:")
print(leaderboard_path)

print("\n" + "=" * 80)
print(
    f"ALL COMPLETE. "
    f"Total Pipeline Time: "
    f"{time.time() - overall_start_time:.2f}s"
)
print("=" * 80)