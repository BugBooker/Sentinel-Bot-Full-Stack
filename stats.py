# ======================================================================
# SENTINEL-BOT
# stats.py
# Statistical Evaluation + Metric Computation
# ======================================================================

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    brier_score_loss
)

from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests

from config import (
    RANDOM_STATE,
    BOOTSTRAP_ITERATIONS,
    ECE_BINS
)

from utils import (
    calculate_ece_mce,
    safe_confusion_unpack,
    bootstrap_auc_test
)

# ==========================================================
# BOOTSTRAP ROC-AUC CONFIDENCE INTERVAL
# ==========================================================

def bootstrap_auc_confidence_interval(
    y_true,
    y_prob,
    n_iterations=BOOTSTRAP_ITERATIONS
):
        
    """
    Computes bootstrap ROC-AUC confidence interval.

    Returns:
        auc_mean,
        auc_ci_lower,
        auc_ci_upper,
        bootstrap_distribution
    """

    rng = np.random.default_rng(RANDOM_STATE)

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bootstrap_aucs = []

    for _ in range(n_iterations):

        indices = rng.choice(
            len(y_true),
            len(y_true),
            replace=True
        )

        y_sample = y_true[indices]
        p_sample = y_prob[indices]

        # ==================================================
        # SINGLE CLASS SAFETY
        # ==================================================

        if len(np.unique(y_sample)) < 2:
            continue

        score = roc_auc_score(y_sample, p_sample)
        bootstrap_aucs.append(score)

    # ======================================================
    # MAIN AUC
    # ======================================================

    auc = roc_auc_score(y_true, y_prob)

    # ======================================================
    # EMPTY SAFETY
    # ======================================================

    if len(bootstrap_aucs) == 0:
        return (auc, auc, auc, [])

    # ======================================================
    # CONFIDENCE INTERVALS
    # ======================================================

    auc_ci_lower = np.percentile(bootstrap_aucs, 2.5)
    auc_ci_upper = np.percentile(bootstrap_aucs, 97.5)

    return (
        auc,
        auc_ci_lower,
        auc_ci_upper,
        bootstrap_aucs
    )

# ==========================================================
# META CV STABILITY EVALUATION
# ==========================================================

def compute_meta_cv_scores(
    meta_model,
    meta_oof_cv,
    y_train,
    cv
):
    """
    Performs leak-free meta-layer CV evaluation.

    Returns:
        cv_scores,
        cv_mean,
        cv_std
    """

    cv_scores = cross_val_score(
        clone(meta_model),
        meta_oof_cv,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1
    )

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    return (
        cv_scores,
        cv_mean,
        cv_std
    )

# ==========================================================
# STATISTICAL SIGNIFICANCE TESTING
# ==========================================================

def compute_significance_tests(
    y_true,
    ensemble_probs,
    base_probs_dict
):
    """
    Computes Holm-corrected bootstrap p-values
    comparing ensemble ROC-AUC against:

        - XGB
        - RF
        - SVM

    Returns:
        dict of corrected p-values
    """

    raw_ps = [
        bootstrap_auc_test(
            y_true,
            ensemble_probs,
            base_probs_dict[m]
        )
        for m in ["XGB", "RF", "SVM"]
    ]

    adjusted_ps = multipletests(
        raw_ps,
        method="holm"
    )[1]

    return dict(zip(
        ["p_vs_xgb", "p_vs_rf", "p_vs_svm"],
        adjusted_ps
    ))

# ==========================================================
# COMPLETE METRIC COMPUTATION
# ==========================================================

def compute_all_metrics(
    y_true,
    y_pred,
    y_prob
):
    """
    Computes all evaluation metrics used
    throughout Sentinel-Bot.

    Returns:
        dictionary of metrics
    """

    # ======================================================
    # MAIN METRICS
    # ======================================================

    acc = accuracy_score(y_true, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
    else:
        auc = 0.5  # Neutral fallback for single-class edge cases

    if len(np.unique(y_true)) > 1:
        ap_score = average_precision_score(y_true, y_prob)
    else:
        ap_score = 0.0

    brier = brier_score_loss(
        y_true,
        y_prob
    )

    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    tn, fp, fn, tp = safe_confusion_unpack(
        y_true,
        y_pred
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0 else 0
    )

    # ======================================================
    # CALIBRATION METRICS
    # ======================================================

    ece, mce = calculate_ece_mce(
        y_true,
        y_prob,
        n_bins=ECE_BINS
    )

    # ======================================================
    # RETURN
    # ======================================================

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "average_precision": ap_score,
        "brier_score": brier,
        "mcc": mcc,
        "false_positive_rate": fpr,
        "ece": ece,
        "mce": mce,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

# ==========================================================
# LEADERBOARD ENTRY CREATOR
# ==========================================================

def build_leaderboard_entry(
    dataset_name,
    model_name,
    metrics,
    threshold=0.50,
    extra_fields=None
):

    # Standardized leaderboard row builder.
    # Prevents duplicated leaderboard logic.
    entry = {
        "dataset": dataset_name,
        "model": model_name,
        "optimal_threshold": round(threshold, 4),
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1_score": round(metrics["f1_score"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "average_precision": round(metrics["average_precision"], 4),
        "brier_score": round(metrics["brier_score"], 4),
        "mcc": round(metrics["mcc"], 4),
        "false_positive_rate": round(metrics["false_positive_rate"], 4),
        "ece": round(metrics["ece"], 4),
        "mce": round(metrics["mce"], 4)
    }

    if extra_fields is not None:
        entry.update(extra_fields)

    return entry

# ==========================================================
# METRICS DATAFRAME EXPORT
# ==========================================================

def metrics_to_dataframe(metrics_dict):

    # Converts metrics dictionary into standardized export dataframe.
    return pd.DataFrame([metrics_dict])

# ==========================================================
# BOOTSTRAP DISTRIBUTION EXPORT
# ==========================================================

def bootstrap_distribution_to_dataframe(
    bootstrap_distribution
):
    # Converts bootstrap ROC-AUC distribution into export dataframe.
    return pd.DataFrame({
        "bootstrap_auc":
            bootstrap_distribution
    })

# ==========================================================
# CV SCORE EXPORT
# ==========================================================

def cv_scores_to_dataframe(cv_scores):

    # Converts CV fold scores into export dataframe.
    return pd.DataFrame({
        "cv_fold_auc":
            cv_scores
    })