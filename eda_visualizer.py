# ==========================================================
# SENTINEL-BOT EDA_VISUALISER_V3
# Thesis-Grade Exploratory Data Analysis
# Supports smart_bot / evasive_bot / human datasets
# Publication-ready plots + reports
# ==========================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ==========================================================
# PATHS
# ==========================================================
CSV_PATH = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\processed\sentinel_full.csv"

OUTPUT_DIR = r"C:\Users\aamin\Professional\Projects\Capstone Project\eda_plots"
REPORT_DIR = os.path.join(OUTPUT_DIR, "eda reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# LOAD DATA
# ==========================================================
print("=" * 60)
print("SENTINEL-BOT EDA VISUALISER V3")
print("=" * 60)

df = pd.read_csv(CSV_PATH)

print("Rows   :", len(df))
print("Cols   :", len(df.columns))

# ==========================================================
# STYLE
# ==========================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

# ==========================================================
# HELPERS
# ==========================================================
def save_plot(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=300)
    plt.close()

def safe_numeric(df_in):
    return df_in.select_dtypes(include=["number"]).copy()

# ==========================================================
# BASIC REPORTS
# ==========================================================
df.describe(include="all").to_csv(os.path.join(REPORT_DIR, "summary_stats.csv"))

missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean() * 100).values
})
missing.to_csv(os.path.join(REPORT_DIR, "missing_report.csv"), index=False)

# ==========================================================
# 1. CLASS BALANCE
# ==========================================================
if "label" in df.columns:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="label", data=df, palette="viridis")

    for p in ax.patches:
        ax.annotate(
            str(int(p.get_height())),
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points"
        )

    plt.title("Dataset Class Balance")
    plt.xlabel("Class (0 = Human, 1 = Bot)")
    plt.ylabel("Sessions")
    save_plot("1_class_balance.png")

# ==========================================================
# 2. FEATURE DISTRIBUTIONS
# ==========================================================
features = [
    "straightness_ratio",
    "mean_velocity",
    "pause_ratio",
    "typing_speed_wpm",
    "backspace_ratio",
    "mouse_entropy",
    "ja4_digest_present",
    "trust_score"
]

for feat in features:
    if feat in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(
            data=df,
            x=feat,
            hue="label",
            bins=30,
            kde=True,
            alpha=0.5,
            stat="density",
            common_norm=False
        )
        plt.title(f"Distribution of {feat}")
        plt.xlabel(feat)
        plt.ylabel("Density")
        save_plot(f"2_dist_{feat}.png")

# ==========================================================
# 3. BOXPLOTS
# ==========================================================
box_feats = [
    "mean_velocity",
    "typing_speed_wpm",
    "pause_ratio",
    "backspace_ratio",
    "ua_ja4_mismatch",
    "trust_score"
]

for feat in box_feats:
    if feat in df.columns:
        plt.figure(figsize=(7, 5))
        sns.boxplot(x="label", y=feat, data=df)
        plt.title(f"{feat} by Class")
        plt.xlabel("Class")
        plt.ylabel(feat)
        save_plot(f"3_box_{feat}.png")

# ==========================================================
# 4. UA / JA4 MISMATCH
# ==========================================================
if "ua_ja4_mismatch" in df.columns:
    plt.figure(figsize=(6, 4))
    sns.barplot(
        x="label",
        y="ua_ja4_mismatch",
        data=df,
        errorbar=None,
        palette="magma"
    )
    plt.title("UA / JA4 Mismatch Rate")
    plt.xlabel("Class")
    plt.ylabel("Mismatch Rate")
    save_plot("4_ua_ja4_mismatch.png")

# ==========================================================
# 5. CORRELATION HEATMAP
# ==========================================================
num = safe_numeric(df)

if "label" in num.columns:
    corr = num.corr()

    top_feats = (
        corr["label"]
        .abs()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df[top_feats].corr(),
        cmap="coolwarm",
        linewidths=0.5,
        annot=False
    )
    plt.title("Top 20 Feature Correlation Heatmap")
    save_plot("5_heatmap_top20.png")

# ==========================================================
# 6. SESSION DURATION VS EVENTS
# ==========================================================
if "session_duration" in df.columns and "total_events" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="session_duration",
        y="total_events",
        hue="label",
        alpha=0.7
    )
    plt.title("Session Duration vs Total Events")
    plt.xlabel("Duration")
    plt.ylabel("Events")
    save_plot("6_duration_vs_events.png")

# ==========================================================
# 7. TOP CORRELATION BAR CHART
# ==========================================================
if "label" in num.columns:
    label_corr = (
        corr["label"]
        .drop("label")
        .abs()
        .sort_values(ascending=False)
        .head(15)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=label_corr.values,
        y=label_corr.index
    )
    plt.title("Top 15 Features Correlated with Label")
    plt.xlabel("Absolute Correlation")
    plt.ylabel("Feature")
    save_plot("7_top_label_correlations.png")

    pd.DataFrame({
        "feature": label_corr.index,
        "abs_corr": label_corr.values
    }).to_csv(os.path.join(REPORT_DIR, "top_correlations.csv"), index=False)

# ==========================================================
# 8. MUTUAL INFORMATION (NONLINEAR IMPORTANCE)
# ==========================================================
if "label" in df.columns:
    if len(df) < 10: 
        print("Dataset too small for meaningful Mutual Information analysis.")
    else:
        # If dataset is very large, sample it to keep EDA fast
        analysis_df = df.sample(500, random_state=42) if len(df) > 500 else df

    X = df.drop(columns=["label"]).copy()
    y = df["label"].astype(int)

    # encode categoricals
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = pd.DataFrame(
        SimpleImputer(strategy="constant", fill_value=0).fit_transform(X),
        columns=X.columns
    )

    mi = mutual_info_classif(X, y, random_state=42)

    mi_df = pd.DataFrame({
        "feature": X.columns,
        "mi_score": mi
    }).sort_values("mi_score", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="mi_score",
        y="feature",
        data=mi_df
    )
    plt.title("Top 15 Features by Mutual Information")
    plt.xlabel("MI Score")
    plt.ylabel("Feature")
    save_plot("8_mutual_information.png")

    mi_df.to_csv(os.path.join(REPORT_DIR, "mutual_information.csv"), index=False)

# ==========================================================
# 9. LEAKAGE DETECTOR
# suspicious near-perfect correlations
# ==========================================================
if "label" in num.columns:

    leak = (
        corr["label"]
        .drop("label")
        .abs()
        .sort_values(ascending=False)
    )

    suspicious = leak[leak > 0.95]

    if len(suspicious) > 0:
        plt.figure(figsize=(9, 5))
        sns.barplot(
            x=suspicious.values,
            y=suspicious.index
        )
        plt.title("Potential Leakage Features (>0.95 Correlation)")
        plt.xlabel("Abs Correlation")
        plt.ylabel("Feature")
        save_plot("9_leakage_flags.png")

        suspicious.to_csv(os.path.join(REPORT_DIR, "leakage_flags.csv"))

# ==========================================================
# 10. PCA VISUALIZATION
# ==========================================================
if "label" in df.columns:

    X = df.drop(columns=["label"]).copy()
    y = df["label"].astype(int)

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = pd.DataFrame(
        SimpleImputer(strategy="constant", fill_value=0).fit_transform(X),
        columns=X.columns
    )

    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X)

    pca_df = pd.DataFrame({
        "PC1": comps[:, 0],
        "PC2": comps[:, 1],
        "label": y
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="label",
        alpha=0.75
    )
    plt.title("PCA Projection of Sessions")
    save_plot("10_pca_projection.png")

# ==========================================================
# COMPLETE
# ==========================================================
print("=" * 60)
print("EDA COMPLETE")
print("Plots saved to :", OUTPUT_DIR)
print("Reports saved  :", REPORT_DIR)
print("=" * 60)