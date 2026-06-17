# ======================================================================
# SENTINEL-BOT (Hybrid Stacked Ensemble)
# train_ensemble_opt_v1.8.py
# (Modular Orchestrator + Fully Refactored v1.8)
# ======================================================================

import os
import time
import random
import warnings
import joblib
import sklearn
import xgboost
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ==========================================================
# LOCAL MODULE IMPORTS
# ==========================================================

from config import (
    RANDOM_STATE,
    DATASETS,
    LABEL,
    SAFE_FEATURES,
    PROCESSED_DIR,
    MODEL_DIR,
    REPORT_DIR,
    TEST_SIZE,
    CALIB_SIZE,
    CV_FOLDS,
    XGB_PARAMS,
    RF_PARAMS,
    SVM_PARAMS,
    META_LR_PARAMS
)

from utils import build_meta
from evaluate import full_evaluation_pipeline

# ==========================================================
# GLOBAL RANDOM SEEDING
# ==========================================================

np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# ==========================================================
# MAIN
# ==========================================================

def main():

    overall_start_time = time.time()
    leaderboard = []

    # ======================================================
    # DATASET LOOP
    # ======================================================

    for dataset_name, filename in DATASETS.items():

        dataset_start_time = time.time()

        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 80)

        # ==================================================
        # LOAD DATASET
        # ==================================================

        dataset_path = os.path.join(PROCESSED_DIR, filename)

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Missing file: {filename}")
            continue

        df = pd.read_csv(dataset_path)

        # ==================================================
        # PRIORITY 1: DUPLICATE REMOVAL
        # ==================================================

        initial_rows = len(df)

        df = df.drop_duplicates().reset_index(drop=True)

        removed = initial_rows - len(df)

        if removed > 0:
            print(f"Audit: Removed {removed} duplicate rows.")

        # ==================================================
        # LABEL CHECK
        # ==================================================

        if LABEL not in df.columns:
            print(f"[ERROR] Missing label column in {filename}")
            continue

        # ==================================================
        # SPLIT FEATURES / LABELS
        # ==================================================

        X = df.drop(columns=[LABEL], errors="ignore")
        y = df[LABEL].astype(int)

        # ==================================================
        # PRIORITY 2: CONSTANT COLUMN PRUNING
        # ==================================================

        constant_cols = [
            c for c in X.columns
            if X[c].nunique(dropna=False) <= 1
        ]

        if constant_cols:
            print(f"Audit: Dropping constant features: {constant_cols}")
            X = X.drop(columns=constant_cols)

        # ==================================================
        # PRIORITY 5: FEATURE AUDIT REPORT
        # ==================================================

        feature_audit = pd.DataFrame({
            "feature": X.columns,
            "missing_pct": X.isna().mean() * 100,
            "unique_values": X.nunique()
        })

        audit_path = os.path.join(
            REPORT_DIR,
            f"{dataset_name}_feature_audit.csv"
        )

        feature_audit.to_csv(audit_path, index=False)

        # ==================================================
        # SAFETY CHECKS
        # ==================================================

        if y.nunique() < 2:
            print(f"[ERROR] Dataset {dataset_name} contains only one class.")
            continue

        X = X.replace([np.inf, -np.inf], np.nan)

        # ==================================================
        # SAFE FEATURE FILTERING
        # ==================================================

        existing_safe_features = sorted([
            f for f in SAFE_FEATURES
            if f in X.columns
        ])

        META_SAFE_FEATURES = existing_safe_features.copy()

        print("\nSafe Features Used:")
        print(META_SAFE_FEATURES)

        # ==================================================
        # TRAIN / CALIB / TEST SPLIT
        # ==================================================

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE
        )

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_train_full,
            y_train_full,
            test_size=CALIB_SIZE,
            stratify=y_train_full,
            random_state=RANDOM_STATE
        )

        # ==================================================
        # SAVE ORIGINAL INDICES FOR AUDITING
        # ==================================================

        train_indices = X_train.index.tolist()
        calib_indices = X_calib.index.tolist()
        test_indices = X_test.index.tolist()

        # ==================================================
        # RESET INDEXES
        # ==================================================

        X_train, X_calib, X_test = [
            d.reset_index(drop=True)
            for d in [X_train, X_calib, X_test]
        ]

        y_train, y_calib, y_test = [
            d.reset_index(drop=True)
            for d in [y_train, y_calib, y_test]
        ]

        print(
            f"\nTrain Size: {len(X_train)} | "
            f"Calib Size: {len(X_calib)} | "
            f"Test Size : {len(X_test)}"
        )

        # ==================================================
        # NUMERIC FEATURE EXTRACTION
        # ==================================================

        # Keep only genuine numeric columns
        numeric_cols = sorted(
            X_train.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        numeric_cols = sorted(numeric_cols)

        if len(numeric_cols) == 0:
            print(f"[ERROR] No valid numeric columns found for {dataset_name}")
            continue

        print(f"\nTotal Numeric Features: {len(numeric_cols)}")

        # ==================================================
        # CLASS BALANCE
        # ==================================================

        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()

        scale_pos_weight = (
            neg_count / pos_count
            if pos_count > 0 else 1
        )

        # ==================================================
        # PREPROCESSOR
        # ==================================================

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

        # ==================================================
        # BASE MODELS
        # ==================================================

        xgb_model = Pipeline([
            ("prep", preprocessor),
            ("clf", XGBClassifier(
                **XGB_PARAMS,
                scale_pos_weight=scale_pos_weight
            ))
        ])

        rf_model = Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(**RF_PARAMS))
        ])

        # probability=False makes SVC significantly faster
        svm_base = SVC(**{**SVM_PARAMS, "probability": False})

        svm_calibrated = CalibratedClassifierCV(
            estimator=svm_base,
            method="sigmoid",
            cv=3
        )

        svm_model = Pipeline([
            ("prep", preprocessor),
            ("clf", svm_calibrated)
        ])

        # ==================================================
        # CROSS VALIDATION SETUP
        # ==================================================

        cv = StratifiedKFold(
            n_splits=CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE
        )

        # ==================================================
        # PRIMARY OOF GENERATION
        # ==================================================

        print("\nGenerating Base OOF Predictions...")

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

        # ==================================================
        # TRAIN BASE MODELS
        # ==================================================

        print("Training Base Models...")

        xgb_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        svm_model.fit(X_train, y_train)

        # ==================================================
        # META TRAIN
        # ==================================================

        meta_train = pd.DataFrame({
            "xgb_prob": xgb_oof,
            "rf_prob": rf_oof,
            "svm_prob": svm_oof
        }).reset_index(drop=True)

        missing = [
            f for f in META_SAFE_FEATURES
            if f not in X_train.columns
        ]

        if missing:
            raise ValueError(f"Missing safe features: {missing}")

        for f in META_SAFE_FEATURES:
            meta_train[f] = X_train[f].values

        # ==================================================
        # META CALIB / TEST
        # ==================================================

        meta_calib = build_meta(
            X_calib,
            xgb_model,
            rf_model,
            svm_model,
            META_SAFE_FEATURES
        )

        meta_test = build_meta(
            X_test,
            xgb_model,
            rf_model,
            svm_model,
            META_SAFE_FEATURES
        )

        # ==================================================
        # FINAL META COLUMNS
        # ==================================================

        prob_cols = ["xgb_prob", "rf_prob", "svm_prob"]

        FINAL_META_COLS = prob_cols + META_SAFE_FEATURES

        meta_train = meta_train[FINAL_META_COLS]
        meta_calib = meta_calib[FINAL_META_COLS]
        meta_test = meta_test[FINAL_META_COLS]

        # ==================================================
        # NUMERIC VALIDATION FOR META FEATURES
        # ==================================================

        for col in META_SAFE_FEATURES:
            if not np.issubdtype(meta_train[col].dtype, np.number):
                raise TypeError(
                    f"Safe feature '{col}' is not numeric! "
                    f"Found {meta_train[col].dtype}"
                )

        # ==================================================
        # META PREPROCESSOR
        # ==================================================

        # Ensures the transformer list matches that exact order
        meta_preprocessor = ColumnTransformer(
            transformers=[
                (
                    "pass_probs",
                    "passthrough",
                    prob_cols
                ),
                (
                    "scale_features",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ]),
                    META_SAFE_FEATURES
                )
            ]
        )

        # ==================================================
        # META LEARNER
        # ==================================================

        print("\nTraining Meta Learner...")

        meta_model = Pipeline([
            ("prep", meta_preprocessor),
            ("lr", LogisticRegression(**META_LR_PARAMS))
        ])

        # ==================================================
        # FIT META MODEL ON OOF TRAIN
        # ==================================================

        meta_model.fit(meta_train, y_train)

        # ==================================================
        # BASE MODEL DICTIONARY
        # ==================================================

        base_models_dict = {
            "XGB": xgb_model,
            "RF": rf_model,
            "SVM": svm_model
        }

        # ==================================================
        # FULL EVALUATION PIPELINE
        # ==================================================

        print("\nRunning Full Evaluation Pipeline...")

        results = full_evaluation_pipeline(
            dataset_name=dataset_name,
            meta_model=meta_model,
            meta_oof_cv=meta_train,
            meta_test=meta_test,
            meta_calib=meta_calib,
            X_test=X_test,
            X_calib=X_calib,
            y_train=y_train,
            y_test=y_test,
            y_calib=y_calib,
            cv=cv,
            base_models_dict=base_models_dict,
            final_meta_cols=FINAL_META_COLS,
            runtime_seconds=(time.time() - dataset_start_time)
        )

        # ==================================================
        # SAVE MODEL
        # ==================================================

        model_path = os.path.join(
            MODEL_DIR,
            f"{dataset_name}_Hybrid_Stacking_v1.8.pkl"
        )

        joblib.dump({

            "meta_model": meta_model,
            "threshold": results["best_threshold"],

            "base_models": {
                "xgb": xgb_model,
                "rf": rf_model,
                "svm": svm_model
            },

            "safe_features": META_SAFE_FEATURES,
            "dataset": dataset_name,
            "meta_columns": FINAL_META_COLS,
            "numeric_cols": numeric_cols,

            # VERSION CONTROL & AUDITABILITY
            "versions": {
                "sklearn": sklearn.__version__,
                "xgboost": xgboost.__version__
            },

            "split_indices": {
                "train_index": train_indices,
                "calib_index": calib_indices,
                "test_index": test_indices
            }

        }, model_path)

        print(f"\nSaved Model: {model_path}")

        # ==================================================
        # LEADERBOARD
        # ==================================================

        leaderboard.append(results["ensemble_entry"])
        leaderboard.extend(results["baseline_entries"])

    # ======================================================
    # FINAL LEADERBOARD
    # ======================================================

    leaderboard_df = pd.DataFrame(leaderboard)

    leaderboard_df = leaderboard_df.sort_values(
        by=["roc_auc", "f1_score"],
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

    # ======================================================
    # FINAL OUTPUT
    # ======================================================

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

# ==========================================================
# ENTRYPOINT
# ==========================================================

if __name__ == "__main__":
    main()