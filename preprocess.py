# preprocess.py v9.0

import os
import pandas as pd
import numpy as np

RANDOM_STATE = 42

# ==========================================================
# PATHS
# ==========================================================
INPUT_FILE = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\sentinel_master_dataset.csv"

BASE_DIR = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection"

# ==============================
# OPTION 2: PORTABLE (FOR GITHUB)
# ==============================
# BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# INPUT_FILE = os.path.join(BASE_PATH, "sentinel_master_dataset.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "processed")
REPORT_DIR = os.path.join(BASE_DIR, "preprocess reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ==========================================================
# LOAD
# ==========================================================
print("=" * 60)
print("LOADING MASTER DATASET...")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)
df.fillna(0, inplace=True)

LABEL = "label"

# ================================
# FEATURE DE-IDENTIFICATION LAYER
# ================================

df["raw_userAgent"] = "masked"
df["source_file"] = 0

# HELPERS
def existing(cols):
    return [c for c in cols if c in df.columns]

def save_csv(name, data):
    path = os.path.join(OUTPUT_DIR, name)
    data.to_csv(path, index=False)

# ==========================================================
# OUTLIER CLIPPING (NEW FIX)
# GLOBAL OUTLIER SANITIZATION
# We clip extreme 1st/99th percentile values globally to remove
# logging glitches and non-physical bot spikes.
# This is treated as data cleaning, not model fitting,
# and is therefore applied before any ML splitting.
# ==========================================================
def clip_outliers(df, cols):
    for col in cols:
        if col in df.columns:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower, upper)
    return df

# ==========================================================
# COLUMN GROUPS
# ==========================================================

RAW_COLUMNS = [
    "raw_userAgent",
    "extractor_version",
    "session_id",
    "timestamp",
    "raw_label",
    "bot_subclass",
    "bot_subclass_encoded"
]

CAT_COLS = [
    "browser",
    "ja4",
    "platform"
]

ENCODED = [
    "browser_encoded",
    "ja4_prefix_encoded",
    "platform_encoded"
]

NETWORK = [
    "accept_lang_present",
    "accept_lang_length",
    "accept_lang_comma_count",
    "accept_lang_q_count",
    "sec_ch_ua_present",
    "sec_ch_ua_length",
    "forwarded_for_count",
    "browser_header_score",
    "ua_ja4_mismatch",
    "ua_length",
    "trust_score"
]

DEVICE = [
    "cpu_cores",
    "device_memory",
    "cpu_cores_missing",
    "device_memory_missing",
    "mobile_vs_desktop",
    "device_platform_mismatch",
    "platform_is_windows",
    "platform_is_mac",
    "platform_is_linux",
    "platform_is_mobile"
]

MOUSE = [
    "move_count",
    "path_length",
    "straight_line_distance",
    "straightness_ratio",
    "mean_velocity",
    "std_velocity",
    "max_velocity",
    "min_velocity",
    "median_velocity",
    "mean_acceleration",
    "std_acceleration",
    "pause_count",
    "pause_ratio",
    "idle_time_mean",
    "direction_changes",
    "mouse_entropy",
    "acceleration_entropy",

    # NEW FEATURES SUPPORT
    "mouse_density"
]

KEYBOARD = [
    "key_event_count",
    "key_press_count",
    "avg_dwell_time",
    "std_dwell_time",
    "avg_flight_time",
    "std_flight_time",
    "key_overlap_count",
    "backspace_count",
    "backspace_ratio",
    "unique_dwell_ratio",
    "unique_key_ratio",
    "typing_speed_wpm",

    # NEW FEATURES
    "key_density",
    "has_backspace"
]

CROSS = [
    "scroll_count",
    "scroll_to_move_ratio",
    "mouse_to_key_ratio",
    "events_per_second",
    "session_duration",
    "total_events",
    "click_count",
    "click_interval_mean",
    "click_hold_mean"
]

LEAKY_COLUMNS = [
    "ja4",
    "raw_userAgent",
    "trust_score",
    "forwarded_for_count",
    "is_edge_observed",
    "network_consistency_score",
    "session_id",
    "source_file",
    "bot_subclass",
    "bot_subclass_encoded",

    # NETWORK SHORTCUT LEAKS (Trojan Horse)
    # These reintroduce forwarded_for_count indirectly
    "browser_header_score",
    "header_completeness_score",
    "sec_ch_ua_suspicion_score",

    # OPTIONAL (depends on extractor behavior — safe to drop)
    # "sec_ch_ua_present",
    # "sec_ch_ua_length",

    # MACRO TIME / SPEED LEAKS (THE BIG PROBLEM)
    # These allow model to detect bots via "speedrun behavior" instead of biometrics
    "session_duration",
    "events_per_second",

    # PAUSE / IDLE SHORTCUTS
    # These dominate your feature importance
    "idle_time_mean",
    "pause_ratio",
    "pause_count",

    # VOLUME / SCALE LEAKS
    # Models use "how much" instead of "how"
    "move_count",
    "key_event_count",
    "key_press_count",
    "click_count",
    "total_events",

    # ======================================================
    # (OPTIONAL BUT RECOMMENDED)
    # If still seeing near-perfect accuracy, drop these too:
    # ======================================================
    # "typing_speed_wpm",
    # "mouse_density",
    # "key_density"
]

df = df.drop(columns=LEAKY_COLUMNS, errors="ignore")
# ...but leaves 'ja4_encoded' and 'ja4_entropy' (calculated in extractor) 
# for the model to use as features.

# ==========================================================
# FILTER EXISTING
# ==========================================================
RAW_COLUMNS = existing(RAW_COLUMNS)
CAT_COLS = existing(CAT_COLS)
ENCODED = existing(ENCODED)
NETWORK = existing(NETWORK)
DEVICE = existing(DEVICE)
MOUSE = existing(MOUSE)
KEYBOARD = existing(KEYBOARD)
CROSS = existing(CROSS)

# ==========================================================
# BASE CLEAN DATASET
# ==========================================================
full = df.drop(columns=RAW_COLUMNS, errors="ignore").copy()

# Apply OUTLIER CLIPPING here (IMPORTANT)
numeric_cols = [
    c for c in full.select_dtypes(include=["number"]).columns
    if full[c].nunique() > 10
]
if LABEL in numeric_cols:
    numeric_cols.remove(LABEL)

print("\n--- STATISTICS BEFORE CLIPPING ---")
print(df[numeric_cols].describe())

full = clip_outliers(full, numeric_cols)

save_csv("sentinel_full.csv", full)

# Save column list for reproducibility/debugging
pd.Series(full.columns).to_csv(
    os.path.join(REPORT_DIR, "columns_used_full_dataset.csv"),
    index=False
)

# ==========================================================
# BIOMETRICS ONLY
# ==========================================================
bio_cols = existing(MOUSE + KEYBOARD + CROSS + [LABEL])
bio = full[bio_cols].copy()
save_csv("sentinel_biometrics_only.csv", bio)

# ==========================================================
# BIOMETRICS + DEVICE
# ==========================================================
bio_dev_cols = existing(MOUSE + KEYBOARD + CROSS + DEVICE + [LABEL])
bio_dev = full[bio_dev_cols].copy()
save_csv("sentinel_biometrics_device.csv", bio_dev)

# ==========================================================
# NETWORK ONLY
# ==========================================================
net_cols = existing(NETWORK + CAT_COLS + ENCODED + [LABEL])
net = full[net_cols].copy()
save_csv("sentinel_network_only.csv", net)

# ==========================================================
# LINEAR MODELS
# Scaling will be done inside train_tournament.py (NO LEAKAGE)
# ==========================================================
linear = full.drop(columns=CAT_COLS, errors="ignore").copy()
save_csv("sentinel_linear.csv", linear)

# ==========================================================
# TREE MODELS
# ==========================================================
tree = full.drop(columns=CAT_COLS, errors="ignore").copy()
save_csv("sentinel_tree_models.csv", tree)

# ==========================================================
# CATBOOST
# ==========================================================
cat = full.copy()
save_csv("sentinel_catboost.csv", cat)

# ==========================================================
# ABLATION (FIXED - NO LEAKAGE)
# Remove:
# - NETWORK
# - CAT_COLS
# - ENCODED (IMPORTANT FIX)
# ==========================================================
blind = full.drop(columns=NETWORK + CAT_COLS + ENCODED, errors="ignore").copy()
save_csv("sentinel_ablation_blind.csv", blind)

# ==========================================================
# REPORTS
# ==========================================================

from collections import Counter

# Console print (quick visibility)
print("Class distribution:", Counter(df[LABEL]))

# Class Balance
counts = df[LABEL].value_counts().sort_index()
with open(os.path.join(REPORT_DIR, "class_balance.txt"), "w") as f:
    for k, v in counts.items():
        cls = "Human (0)" if k == 0 else "Bot (1)"
        f.write(f"{cls}: {v}\n")

# Feature Counts
with open(os.path.join(REPORT_DIR, "feature_counts.txt"), "w") as f:
    f.write(f"Full Dataset          : {full.shape[1]-1}\n")
    f.write(f"Biometrics Only       : {bio.shape[1]-1}\n")
    f.write(f"Biometrics + Device   : {bio_dev.shape[1]-1}\n")
    f.write(f"Network Only          : {net.shape[1]-1}\n")
    f.write(f"Linear (Unscaled)     : {linear.shape[1]-1}\n")
    f.write(f"Tree Models           : {tree.shape[1]-1}\n")
    f.write(f"CatBoost              : {cat.shape[1]-1}\n")
    f.write(f"Ablation Blind        : {blind.shape[1]-1}\n")

# Missing Report
missing = pd.DataFrame({
    "column": df.columns,
    "missing_count": df.isna().sum().values,
    "missing_percent": (df.isna().mean() * 100).values
})

missing.to_csv(os.path.join(REPORT_DIR, "missing_report.csv"), index=False)

# ==========================================================
# FINAL
# ==========================================================
print("\n" + "=" * 60)
print("PREPROCESS_V7 COMPLETE (THESIS READY)")
print("=" * 60)

print("Full Dataset          :", full.shape)
print("Linear (Unscaled)       :", linear.shape)
print("Ablation (Clean)      :", blind.shape)

print("Saved to:", OUTPUT_DIR)
print("=" * 60)
