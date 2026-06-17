# ======================================================================
# SENTINEL-BOT
# plots.py
# Evaluation Visualization Utilities
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)

from sklearn.calibration import (
    calibration_curve
)

from config import (
    PLOT_DIR,
    PLOT_DPI,
    ROC_FIGSIZE,
    PR_FIGSIZE,
    CM_FIGSIZE,
    CALIBRATION_FIGSIZE,
    DISTRIBUTION_FIGSIZE,
    FEATURE_IMPORTANCE_FIGSIZE,
    THRESHOLD_FIGSIZE,
    DIVERSITY_FIGSIZE,
    DISTRIBUTION_BINS,
    CALIBRATION_BINS
)

# ==========================================================
# ROC CURVE
# ==========================================================

def plot_roc_curve(
    y_true,
    y_prob,
    auc,
    dataset_name
):
    """
    Generates ROC Curve plot.
    """

    fpr_plot, tpr_plot, _ = roc_curve(
        y_true,
        y_prob
    )

    plt.figure(
        figsize=ROC_FIGSIZE
    )

    plt.plot(
        fpr_plot,
        tpr_plot,
        linewidth=2,
        label=f"Hybrid Stack (AUC={auc:.3f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        f"Hybrid Stacking ROC Curve - "
        f"{dataset_name.upper()}"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"ensemble_roc_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# PRECISION-RECALL CURVE
# ==========================================================

def plot_precision_recall_curve(
    y_true,
    y_prob,
    ap_score,
    dataset_name
):
    """
    Generates Precision-Recall curve.
    """

    precision_plot, recall_plot, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    plt.figure(
        figsize=PR_FIGSIZE
    )

    plt.plot(
        recall_plot,
        precision_plot,
        linewidth=2,
        color="purple",
        label=f"Hybrid Stack (AP={ap_score:.3f})"
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        f"Precision-Recall Curve - "
        f"{dataset_name.upper()}"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"precision_recall_curve_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    dataset_name
):
    """
    Generates confusion matrix heatmap.
    """

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(
        figsize=CM_FIGSIZE
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "Bot"],
        yticklabels=["Human", "Bot"]
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        f"Confusion Matrix - "
        f"{dataset_name.upper()}"
    )

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"confusion_matrix_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# CALIBRATION CURVE
# ==========================================================

def plot_calibration_curve(
    y_true,
    y_prob,
    dataset_name
):
    """
    Generates probability calibration curve.
    """

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=CALIBRATION_BINS,
        strategy="quantile"
    )

    plt.figure(
        figsize=CALIBRATION_FIGSIZE
    )

    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        label="Model Calibration"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        "--",
        color="gray",
        label="Perfect Calibration"
    )

    plt.xlabel(
        "Predicted Probability"
    )

    plt.ylabel(
        "True Probability"
    )

    plt.title(
        f"Calibration Curve - "
        f"{dataset_name.upper()}"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"calibration_curve_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# PROBABILITY DISTRIBUTION
# ==========================================================

def plot_probability_distribution(
    y_true,
    y_prob,
    best_threshold,
    dataset_name
):
    """
    Generates probability distribution histogram.
    """

    plt.figure(
        figsize=DISTRIBUTION_FIGSIZE
    )

    plt.hist(
        y_prob[y_true == 0],
        bins=DISTRIBUTION_BINS,
        alpha=0.6,
        label="Human",
        color="#1f77b4"
    )

    plt.hist(
        y_prob[y_true == 1],
        bins=DISTRIBUTION_BINS,
        alpha=0.6,
        label="Bot",
        color="#d62728"
    )

    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        label=f"Threshold={best_threshold:.2f}"
    )

    plt.xlabel(
        "Predicted Bot Probability"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.title(
        f"Probability Distribution - "
        f"{dataset_name.upper()}"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"probability_distribution_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# META FEATURE IMPORTANCE
# ==========================================================

def plot_meta_feature_importance(
    coef_df,
    dataset_name,
    top_n=15
):
    """
    Generates meta-learner coefficient plot.
    """

    # TO FORCE SORTING:
    top_coef = coef_df.sort_values(by="abs_weight", ascending=False).head(top_n)

    plt.figure(figsize=FEATURE_IMPORTANCE_FIGSIZE)

    bar_colors = [

        "#d62728"
        if d == "BOT"
        else "#1f77b4"

        for d in top_coef[
            "effect_direction"
        ]
    ]

    plt.barh(
        top_coef["feature"],
        top_coef["weight_coefficient"],
        color=bar_colors
    )

    plt.xlabel(
        "Coefficient Weight "
        "(Positive=BOT, Negative=HUMAN)"
    )

    plt.title(
        f"Meta-Learner Feature Importance - "
        f"{dataset_name.upper()}"
    )

    plt.gca().invert_yaxis()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"meta_feature_importance_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# THRESHOLD OPTIMIZATION CURVE
# ==========================================================

def plot_threshold_optimization(
    thresholds,
    f1_scores,
    best_threshold,
    dataset_name
):
    """
    Generates threshold-vs-F1 optimization curve.
    """

    plt.figure(
        figsize=THRESHOLD_FIGSIZE
    )

    if len(thresholds) > 0:

        valid_len = min(
            len(thresholds),
            len(f1_scores)
        )

        plt.plot(
            thresholds[:valid_len],
            f1_scores[:valid_len],
            linewidth=2,
            color="#1f77b4"
        )

    plt.axvline(
        best_threshold,
        linestyle="--",
        color="red",
        label=f"Best Threshold={best_threshold:.2f}"
    )

    plt.xlabel(
        "Threshold"
    )

    plt.ylabel(
        "F1 Score"
    )

    plt.title(
        f"Threshold Optimization - "
        f"{dataset_name.upper()}"
    )

    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"threshold_optimization_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# ENSEMBLE DIVERSITY MATRIX
# ==========================================================

def plot_ensemble_diversity(
    meta_test,
    dataset_name
):
    """
    Generates ensemble diversity correlation matrix.
    """

    base_prob_df = pd.DataFrame({

        "xgb": meta_test["xgb_prob"],

        "rf": meta_test["rf_prob"],

        "svm": meta_test["svm_prob"]

    })

    corr_matrix = base_prob_df.corr()

    # ======================================================
    # SAVE RAW CORRELATION MATRIX
    # ======================================================

    corr_matrix.to_csv(
        os.path.join(
            PLOT_DIR,
            f"{dataset_name}_ensemble_diversity.csv"
        )
    )

    # ======================================================
    # HEATMAP
    # ======================================================

    plt.figure(
        figsize=DIVERSITY_FIGSIZE
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )

    plt.title(
        f"Ensemble Diversity Matrix - "
        f"{dataset_name.upper()}"
    )

    plt.tight_layout()

    save_path = os.path.join(
        PLOT_DIR,
        f"ensemble_diversity_{dataset_name}.png"
    )

    plt.savefig(
        save_path,
        dpi=PLOT_DPI
    )

    plt.close()

# ==========================================================
# GENERATE ALL PLOTS
# ==========================================================

def generate_all_plots(
    y_true,
    y_pred,
    y_prob,
    auc,
    ap_score,
    best_threshold,
    coef_df,
    thresholds,
    f1_scores,
    meta_test,
    dataset_name
):
    """
    Master wrapper for generating all evaluation plots.
    """

    plot_roc_curve(
        y_true,
        y_prob,
        auc,
        dataset_name
    )

    plot_precision_recall_curve(
        y_true,
        y_prob,
        ap_score,
        dataset_name
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        dataset_name
    )

    plot_calibration_curve(
        y_true,
        y_prob,
        dataset_name
    )

    plot_probability_distribution(
        y_true,
        y_prob,
        best_threshold,
        dataset_name
    )

    plot_meta_feature_importance(
        coef_df,
        dataset_name
    )

    plot_threshold_optimization(
        thresholds,
        f1_scores,
        best_threshold,
        dataset_name
    )

    plot_ensemble_diversity(
        meta_test,
        dataset_name
    )