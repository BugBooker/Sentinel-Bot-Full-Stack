# ======================================================================
# SENTINEL-BOT
# utils.py
# Core Utility Functions
# ======================================================================

import time
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score
)

from sklearn.base import clone

from config import (
    RANDOM_STATE,
    LATENCY_WARMUP_RUNS,
    LATENCY_MAX_SAMPLES,
    BOOTSTRAP_ITERATIONS
)

# ==========================================================
# CONSISTENT QUANTILE-BASED ECE / MCE
# ==========================================================

def calculate_ece_mce(
    y_true,
    y_prob,
    n_bins=10
):
    """
    Calculates:
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)

    Uses quantile binning for stable calibration estimation.
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    quantiles = np.linspace(
        0,
        1,
        n_bins + 1
    )

    bins = np.quantile(
        y_prob,
        quantiles
    )

    bins = np.unique(bins)

    if len(bins) < 2:
        return 0.0, 0.0

    ece = 0.0
    mce = 0.0

    for i in range(len(bins) - 1):

        if i == 0:

            mask = (
                (y_prob >= bins[i]) &
                (y_prob <= bins[i + 1])
            )

        else:

            mask = (
                (y_prob > bins[i]) &
                (y_prob <= bins[i + 1])
            )

        if np.any(mask):

            acc = np.mean(
                y_true[mask]
            )

            conf = np.mean(
                y_prob[mask]
            )

            gap = abs(
                conf - acc
            )

            ece += (
                np.mean(mask) * gap
            )

            mce = max(
                mce,
                gap
            )

    return ece, mce

# ==========================================================
# SAFE CONFUSION MATRIX UNPACK
# ==========================================================

def safe_confusion_unpack(
    y_true,
    y_pred
):
    """
    Safely extracts:
        TN, FP, FN, TP

    Prevents failures in edge-case class predictions.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    return tn, fp, fn, tp

# ==========================================================
# LATENCY PROFILING
# ==========================================================

def latency_profile(
    model,
    X_df
):
    """
    Measures inference latency statistics:
        - mean
        - median
        - p95

    Uses predict_proba() timing.
    """

    latencies = []

    # ======================================================
    # WARMUP
    # ======================================================

    for _ in range(LATENCY_WARMUP_RUNS):

        _ = model.predict_proba(
            X_df.head(1)
        )

    time.sleep(0.01)

    # ======================================================
    # TIMING LOOP
    # ======================================================

    for i in range(

        min(
            LATENCY_MAX_SAMPLES,
            len(X_df)
        )
    ):

        sample = X_df.iloc[[i]]

        start = time.perf_counter()

        _ = model.predict_proba(
            sample
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(
            elapsed_ms
        )

    # ======================================================
    # EMPTY SAFETY
    # ======================================================

    if len(latencies) == 0:

        return {
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0
        }

    # ======================================================
    # FINAL STATS
    # ======================================================

    return {
        "mean": float(
            np.mean(latencies)
        ),
        "median": float(
            np.median(latencies)
        ),
        "p95": float(
            np.percentile(latencies, 95)
        )
    }

# ==========================================================
# META FEATURE BUILDER
# ==========================================================

def build_meta(
    X_src,
    m1,
    m2,
    m3,
    safe_feats
):
    """
    Constructs meta-learner input dataframe.

    Includes:
        - Base model probabilities
        - Safe handcrafted biometric features
    """

    meta = pd.DataFrame({

        "xgb_prob": (
            m1.predict_proba(X_src)[:, 1]
        ),

        "rf_prob": (
            m2.predict_proba(X_src)[:, 1]
        ),

        "svm_prob": (
            m3.predict_proba(X_src)[:, 1]
        )

    }).reset_index(drop=True)

    X_src = X_src.reset_index(drop=True)

    # ======================================================
    # SAFE FEATURE APPEND (HARD VALIDATION)
    # ======================================================

    missing = [
        f for f in safe_feats
        if f not in X_src.columns
    ]

    if missing:
        raise ValueError(
            f"Missing safe features: {missing}"
        )

    for f in safe_feats:
        meta[f] = X_src[f].values

    return meta

# ==========================================================
# BOOTSTRAP AUC DIFFERENCE TEST
# ==========================================================

def bootstrap_auc_test(
    y_true,
    a,
    b,
    n=BOOTSTRAP_ITERATIONS
):
    """
    Bootstrap significance test comparing:
        Model A ROC-AUC vs Model B ROC-AUC

    Returns:
        p-value
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    y_true = np.asarray(y_true)
    a = np.asarray(a)
    b = np.asarray(b)

    diffs = []

    for _ in range(n):

        idx = rng.choice(
            len(y_true),
            len(y_true),
            replace=True
        )

        y_boot = y_true[idx]

        # ==================================================
        # SINGLE CLASS SAFETY
        # ==================================================

        if len(np.unique(y_boot)) < 2:
            continue

        auc_a = roc_auc_score(
            y_boot,
            a[idx]
        )

        auc_b = roc_auc_score(
            y_boot,
            b[idx]
        )

        diffs.append(
            auc_a - auc_b
        )

    diffs = np.asarray(diffs)

    # ======================================================
    # EMPTY SAFETY
    # ======================================================

    if len(diffs) == 0:
        return 1.0

    # ======================================================
    # TWO-SIDED P-VALUE
    # ======================================================

    p_value = 2 * min(
        np.mean(diffs <= 0),
        np.mean(diffs >= 0)
    )

    return min(
        p_value,
        1.0
    )