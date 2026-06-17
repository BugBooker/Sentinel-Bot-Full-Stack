# ======================================================================
# SENTINEL-BOT
# evaluate.py
# Evaluation + Reporting Pipeline (Refactored v1.7 Final)
# ======================================================================

import os
import time
import numpy as np
import pandas as pd

from sklearn.metrics import precision_recall_curve

from config import (
    REPORT_DIR,
    THRESHOLD_MIN,
    THRESHOLD_MAX
)

from utils import (
    latency_profile
)

from stats import (
    compute_all_metrics,
    compute_meta_cv_scores,
    bootstrap_auc_confidence_interval,
    build_leaderboard_entry,
    compute_significance_tests
)

from plots import generate_all_plots


# ==========================================================
# THRESHOLD OPTIMIZATION
# ==========================================================

def optimize_threshold(y_true, probs):
    """
    Optimizes classification threshold
    using maximum F1 score from PR curve.
    """

    # ======================================================
    # SAFETY CHECK
    # ======================================================

    if len(np.unique(y_true)) < 2:
        return 0.50, np.array([]), np.array([])

    precision_vals, recall_vals, thresholds = precision_recall_curve(
        y_true,
        probs
    )

    # ======================================================
    # F1 COMPUTATION
    # ======================================================

    f1_scores = (
        2 *
        precision_vals[:-1] *
        recall_vals[:-1]
    ) / (
        precision_vals[:-1] +
        recall_vals[:-1] +
        1e-10
    )

    # ======================================================
    # EMPTY / NAN SAFETY
    # ======================================================

    if len(thresholds) == 0:
        return 0.50, np.array([]), np.array([])

    if np.isnan(f1_scores).all():
        return 0.50, thresholds, f1_scores

    # ======================================================
    # BEST THRESHOLD
    # ======================================================

    best_idx = np.argmax(f1_scores)

    best_threshold = thresholds[best_idx]

    # ======================================================
    # CLAMPING
    # ======================================================

    best_threshold = np.clip(
        best_threshold,
        THRESHOLD_MIN,
        THRESHOLD_MAX
    )

    return (
        float(best_threshold),
        thresholds,
        f1_scores
    )


# ==========================================================
# ENSEMBLE EVALUATION
# ==========================================================

def evaluate_ensemble_model(
    dataset_name,
    meta_model,
    meta_test,
    y_test,
    best_threshold,
    meta_cv_auc_mean,
    meta_cv_auc_std,
    base_probs_dict,
    runtime_seconds
):
    """
    Full ensemble evaluation.
    """

    # ======================================================
    # LATENCY PROFILING
    # ======================================================

    latency_stats = latency_profile(
        meta_model,
        meta_test
    )

    # ======================================================
    # INFERENCE TIMING
    # ======================================================

    start_inf = time.time()

    final_probs = meta_model.predict_proba(
        meta_test
    )[:, 1]

    inference_time_ms = (
        (time.time() - start_inf)
        / len(meta_test)
    ) * 1000

    # ======================================================
    # FINAL PREDICTIONS
    # ======================================================

    final_preds = (
        final_probs >= best_threshold
    ).astype(int)

    # ======================================================
    # METRICS
    # ======================================================

    metrics = compute_all_metrics(
        y_test,
        final_preds,
        final_probs
    )

    # ======================================================
    # AUC CONFIDENCE INTERVAL
    # ======================================================

    (
        _,
        auc_ci_lower,
        auc_ci_upper,
        bootstrap_aucs
    ) = bootstrap_auc_confidence_interval(
        y_test.values,
        final_probs
    )

    # ======================================================
    # SAVE BOOTSTRAP DISTRIBUTION
    # ======================================================

    bootstrap_df = pd.DataFrame({
        "bootstrap_auc": bootstrap_aucs
    })

    bootstrap_df.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{dataset_name}_bootstrap_auc_distribution.csv"
        ),
        index=False
    )

    # ======================================================
    # SIGNIFICANCE TESTS (HOLM-CORRECTED)
    # ======================================================

    p_values = compute_significance_tests(
        y_test.values,
        final_probs,
        base_probs_dict
    )

    # ======================================================
    # LEADERBOARD ENTRY
    # ======================================================

    ensemble_entry = build_leaderboard_entry(
        dataset_name=dataset_name,
        model_name="Hybrid_Stacking_v1.7",
        metrics=metrics,
        threshold=best_threshold,

        extra_fields={

            "meta_cv_auc_mean": round(meta_cv_auc_mean, 4),
            "meta_cv_auc_std": round(meta_cv_auc_std, 4),
            "meta_stage_inference_ms":round(inference_time_ms, 4),
            "meta_stage_lat_p95_ms":round(latency_stats["p95"], 4),
            "auc_ci_lower":round(auc_ci_lower, 4),
            "auc_ci_upper":round(auc_ci_upper, 4),
            "pipeline_runtime_sec":round(runtime_seconds, 2),
            "p_vs_xgb":round(p_values["p_vs_xgb"], 6),
            "p_vs_rf":round(p_values["p_vs_rf"], 6),
            "p_vs_svm":round(p_values["p_vs_svm"], 6)
        }
    )

    # ======================================================
    # SAVE PREDICTIONS
    # ======================================================

    predictions_df = pd.DataFrame({
        "true_label": y_test.values,
        "predicted_probability": final_probs,
        "predicted_label": final_preds
    })

    predictions_df.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{dataset_name}_predictions.csv"
        ),
        index=False
    )

    return (
        ensemble_entry,
        final_probs,
        final_preds,
        bootstrap_aucs
    )


# ==========================================================
# BASELINE MODEL EVALUATION
# ==========================================================

def evaluate_baseline_models(
    dataset_name,
    base_models_dict,
    X_test,
    y_test,
    X_calib,
    y_calib,
    runtime_seconds
):
    """
    Evaluates all baseline models.
    """

    leaderboard_entries = []

    base_probs_dict = {}

    for model_name, model_obj in base_models_dict.items():

        # ==================================================
        # CALIBRATION PROBABILITIES
        # ==================================================

        calib_probs = model_obj.predict_proba(
            X_calib
        )[:, 1]

        # ==================================================
        # THRESHOLD OPTIMIZATION
        # ==================================================

        best_threshold, _, _ = optimize_threshold(
            y_calib,
            calib_probs
        )

        # ==================================================
        # TEST PROBABILITIES
        # ==================================================

        test_probs = model_obj.predict_proba(
            X_test
        )[:, 1]

        base_probs_dict[model_name] = test_probs

        # ==================================================
        # TEST PREDICTIONS
        # ==================================================

        test_preds = (
            test_probs >= best_threshold
        ).astype(int)

        # ==================================================
        # METRICS
        # ==================================================

        metrics = compute_all_metrics(
            y_test,
            test_preds,
            test_probs
        )

        # ==================================================
        # CONFIDENCE INTERVALS
        # ==================================================

        (
            _,
            auc_ci_lower,
            auc_ci_upper,
            _
        ) = bootstrap_auc_confidence_interval(
            y_test.values,
            test_probs
        )

        # ==================================================
        # LEADERBOARD ENTRY
        # ==================================================

        entry = build_leaderboard_entry(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics=metrics,
            threshold=best_threshold,

            extra_fields={

                "meta_cv_auc_mean": None,
                "meta_cv_auc_std": None,

                "meta_stage_inference_ms": None,
                "meta_stage_lat_p95_ms": None,

                "auc_ci_lower": round(auc_ci_lower, 4),
                "auc_ci_upper": round(auc_ci_upper, 4),

                "pipeline_runtime_sec": round(runtime_seconds, 2),

                "p_vs_xgb": None,
                "p_vs_rf": None,
                "p_vs_svm": None
            }
        )

        leaderboard_entries.append(entry)

    return (
        leaderboard_entries,
        base_probs_dict
    )


# ==========================================================
# FULL EVALUATION PIPELINE
# ==========================================================

def full_evaluation_pipeline(
    dataset_name,
    meta_model,
    meta_oof_cv,
    meta_test,
    meta_calib,
    X_test,
    X_calib,
    y_train,
    y_test,
    y_calib,
    cv,
    base_models_dict,
    final_meta_cols,
    runtime_seconds
):
    """
    Master evaluation pipeline for v1.7.
    """

    # ======================================================
    # THRESHOLD OPTIMIZATION
    # ======================================================

    print("Optimizing Threshold...")

    calib_probs = meta_model.predict_proba(
        meta_calib
    )[:, 1]

    (
        best_threshold,
        thresholds,
        f1_scores

    ) = optimize_threshold(
        y_calib,
        calib_probs
    )

    # ======================================================
    # SAVE THRESHOLD SEARCH
    # ======================================================

    if len(thresholds) > 0:
        # Guarantee alignment
        valid_len = min(len(thresholds), len(f1_scores))
        
        threshold_df = pd.DataFrame({
            "threshold": thresholds[:valid_len],
            "f1_score": f1_scores[:valid_len]
        })

        threshold_df.to_csv(
            os.path.join(
                REPORT_DIR,
                f"{dataset_name}_threshold_search.csv"
            ),
            index=False
        )

    # ======================================================
    # BASELINE EVALUATION
    # ======================================================

    print("Evaluating Baseline Models...")

    (
        baseline_entries,
        base_probs_dict

    ) = evaluate_baseline_models(
        dataset_name,
        base_models_dict,
        X_test,
        y_test,
        X_calib,
        y_calib,
        runtime_seconds
    )

    # ======================================================
    # META CV STABILITY
    # ======================================================

    print("\nComputing Meta CV Scores...")

    (
        meta_cv_scores,
        meta_cv_auc_mean,
        meta_cv_auc_std

    ) = compute_meta_cv_scores(
        meta_model,
        meta_oof_cv[final_meta_cols],
        y_train,
        cv
    )

    pd.DataFrame({"cv_fold_auc": meta_cv_scores}).to_csv(
        os.path.join(REPORT_DIR, f"{dataset_name}_meta_cv_scores.csv"), index=False
    )

    # ======================================================
    # ENSEMBLE EVALUATION
    # ======================================================

    print("Evaluating Ensemble...")

    (
        ensemble_entry,
        final_probs,
        final_preds,
        bootstrap_aucs

    ) = evaluate_ensemble_model(
        dataset_name=dataset_name,
        meta_model=meta_model,
        meta_test=meta_test,
        y_test=y_test,
        best_threshold=best_threshold,
        meta_cv_auc_mean=meta_cv_auc_mean,
        meta_cv_auc_std=meta_cv_auc_std,
        base_probs_dict=base_probs_dict,
        runtime_seconds=runtime_seconds
    )

    # ======================================================
    # META FEATURE IMPORTANCE
    # ======================================================

    lr_model = meta_model.named_steps["lr"]

    coef_df = pd.DataFrame({
        "feature": final_meta_cols,
        "weight_coefficient": lr_model.coef_[0]
    })

    coef_df["abs_weight"] = (
        coef_df["weight_coefficient"].abs()
    )

    coef_df["effect_direction"] = np.where(
        coef_df["weight_coefficient"] >= 0,
        "BOT",
        "HUMAN"
    )

    coef_df = coef_df.sort_values(
        by="abs_weight",
        ascending=False
    )

    coef_df.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{dataset_name}_meta_feature_importance.csv"
        ),
        index=False
    )

    # ======================================================
    # GENERATE PLOTS
    # ======================================================

    print("Generating Plots...")

    generate_all_plots(
        y_true=y_test,
        y_pred=final_preds,
        y_prob=final_probs,
        auc=ensemble_entry["roc_auc"],
        ap_score=ensemble_entry["average_precision"],
        best_threshold=best_threshold,
        coef_df=coef_df,
        thresholds=thresholds,
        f1_scores=f1_scores,
        meta_test=meta_test,
        dataset_name=dataset_name
    )

    # ======================================================
    # RETURN
    # ======================================================

    return {

        "ensemble_entry":ensemble_entry,
        "baseline_entries":baseline_entries,
        "best_threshold":best_threshold,
        "final_probs":final_probs,
        "final_preds":final_preds,
        "bootstrap_aucs":bootstrap_aucs,
    }