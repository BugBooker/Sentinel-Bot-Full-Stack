# ==========================================================
# feature_extractor.py v9.0
# SENTINEL-BOT MASTER FEATURE EXTRACTOR (THESIS GRADE)
# Supports:
#   human
#   bot
#   evasive_bot
#   smart_bot
#   aversarial_bot
#
# Multimodal Layers:
#   1. Behavioral Biometrics
#   2. HTTP Header Intelligence
#   3. TLS JA4 Fingerprinting
#   4. Device Signals
#   5. Cross-Modal Ratios
#
# Output:
#   sentinel_master_dataset.csv
# ==========================================================

import os
import json
import math
import traceback
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42

# CONFIG
INPUT_DIR = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\final multimodal"
OUTPUT_CSV = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\sentinel_master_dataset.csv"
ERROR_LOG = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\errors.log"

EXTRACTOR_VERSION = "9.0"

BOT_LABELS = ["bot", "evasive_bot", "smart_bot", "adversarial_bot"]

# ==========================================================
# FEATURE LEAKAGE SHIELD (ANTI-LAZY-LEARNING LAYER)
# ==========================================================

WEAK_SIGNAL_FEATURES = set([
    "sec_ch_ua_present",
    "sec_ch_ua_length",
    "accept_lang_present",
    "accept_lang_length",
    "accept_lang_comma_count",
    "accept_lang_q_count",
    "forwarded_for_count",
    "cpu_cores_missing",
    "device_memory_missing"
])

def sanitize_weak_signals(f):
    """
    Prevents model from over-relying on brittle infrastructure signals.
    Converts them into low-resolution bounded signals.
    """

    for k in WEAK_SIGNAL_FEATURES:
        if k in f:
            # compress signal into coarse bins
            try:
                val = float(f[k])

                # BINNING (prevents memorization)
                if val == 0:
                    f[k] = 0
                elif val == 1:
                    f[k] = 1
                else:
                    f[k] = 2
            except:
                f[k] = 0

    return f

# ==========================================================
# HELPERS
# ==========================================================
def safe_float(x):
    try:
        if x is None:
            return 0.0
        if str(x).strip().lower() in ["unknown", "undefined", "null", "nan", ""]:
            return 0.0
        return float(x)
    except:
        return 0.0


def safe_mean(arr):
    return float(np.mean(arr)) if len(arr) else 0.0


def safe_std(arr):
    return float(np.std(arr)) if len(arr) else 0.0


def safe_max(arr):
    return float(np.max(arr)) if len(arr) else 0.0


def safe_min(arr):
    return float(np.min(arr)) if len(arr) else 0.0


def safe_median(arr):
    return float(np.median(arr)) if len(arr) else 0.0


def euclid(p1, p2):
    return math.sqrt(
        (safe_float(p2.get("x")) - safe_float(p1.get("x"))) ** 2 +
        (safe_float(p2.get("y")) - safe_float(p1.get("y"))) ** 2
    )


def entropy(values):
    if len(values) < 2:
        return 0.0

    arr = np.array(values, dtype=float)
    bins = max(5, int(math.sqrt(len(arr))))

    hist, _ = np.histogram(arr, bins=bins)
    total = np.sum(hist)

    if total == 0:
        return 0.0

    p = hist / total
    p = p[p > 0]

    return float(-np.sum(p * np.log2(p)))


def get_timestamps(mouse, keys, scrolls):
    ts = []

    for arr in [mouse, keys, scrolls]:
        for item in arr:
            if isinstance(item, dict) and "t" in item:
                try:
                    ts.append(float(item["t"]))
                except:
                    pass

    return ts

# Added this helper for the string and float characters in a JA4 signature
def string_entropy(text):
    if not text:
        return 0.0
    
    counts = {}
    for char in text:
        counts[char] = counts.get(char, 0) + 1
        
    length = float(len(text))
    ent = 0.0
    for count in counts.values():
        p = count / length
        ent -= p * math.log2(p)
        
    return float(ent)


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================
def extract_features(data, filename):

    metadata = data.get("metadata", {})
    network = data.get("network_layer", {})
    biometric = data.get("biometric_layer", {})
    tls = data.get("tls_layer", {})

    mouse = biometric.get("mouse") or []
    keys = biometric.get("keys") or []
    scrolls = biometric.get("scroll") or []
    device = biometric.get("device") or {}

    f = {}

    # ------------------------------------------------------
    # META
    # ------------------------------------------------------
    label = str(metadata.get("label", "")).lower()

    f["extractor_version"] = EXTRACTOR_VERSION
    f["source_file"] = filename
    f["session_id"] = metadata.get("sessionId", "unknown")
    f["raw_label"] = label
    f["bot_subclass"] = label if label in BOT_LABELS else "human"
    f["label"] = int(label in BOT_LABELS)

    # ------------------------------------------------------
    # TLS / NETWORK
    # ------------------------------------------------------
    browser = str(tls.get("browser", "unknown"))
    ja4_raw = str(tls.get("ja4_raw", "unknown"))
    ja4_digest = str(tls.get("ja4_digest", "unknown"))

    f["trust_score"] = safe_float(tls.get("trust_score", 0))

    f["ja4_digest_present"] = int(ja4_digest not in ["not_captured", "not_observed", "unknown", ""])
    f["ja4_consistency"] = int(ja4_raw != "unknown" and ja4_raw == ja4_digest)

    ua = str(network.get("userAgent", ""))
    ua_low = ua.lower()

    f["browser"] = browser
    f["ja4"] = ja4_raw
    f["raw_userAgent"] = ua

    f["ua_length"] = len(ua)

    f["mobile_vs_desktop"] = int(
        any(x in ua_low for x in ["mobile", "iphone", "android", "ipad"])
    )

    f["ua_is_chrome"] = int("chrome" in ua_low and "edg" not in ua_low and "opr" not in ua_low)
    f["ua_is_firefox"] = int("firefox" in ua_low)
    f["ua_is_edge"] = int("edg" in ua_low)
    f["ua_is_safari"] = int("safari" in ua_low and "chrome" not in ua_low)
    f["ua_is_opera"] = int("opr" in ua_low or "opera" in ua_low)

    accept_lang = str(network.get("acceptLanguage", "missing"))

    f["accept_lang_present"] = int(
        accept_lang.lower() != "missing" and accept_lang != ""
    )

    f["accept_lang_length"] = len(accept_lang) if f["accept_lang_present"] else 0
    f["accept_lang_comma_count"] = accept_lang.count(",") if f["accept_lang_present"] else 0
    f["accept_lang_q_count"] = accept_lang.lower().count("q=") if f["accept_lang_present"] else 0

    sec_ch = str(network.get("secChUa", "missing"))

    f["sec_ch_ua_present"] = int(sec_ch.lower() != "missing" and sec_ch != "")
    f["sec_ch_ua_length"] = len(sec_ch) if f["sec_ch_ua_present"] else 0

    forwarded = str(network.get("forwardedFor", ""))

    forwarded_parts = []
    for p in forwarded.replace(";", ",").split(","):
        val = p.strip()
        if val:
            forwarded_parts.append(val)

    f["forwarded_for_count"] = len(forwarded_parts)

    f["browser_header_score"] = (
        f["accept_lang_present"] +
        f["sec_ch_ua_present"] +
        min(f["forwarded_for_count"], 3)
    )

    # JA4 intelligence
    ja4_low = ja4_raw.lower()

    f["ja4_unknown"] = int(ja4_low == "unknown")
    f["ja4_bot_tls"] = int("t13d3112h1" in ja4_low)
    f["ja4_modern_tls"] = int("h2" in ja4_low)
    f["ja4_length"] = len(ja4_raw)
    f["ja4_prefix"] = ja4_raw.split("_")[0] if "_" in ja4_raw else "unknown"
    f["ja4_entropy_proxy"] = len(set(ja4_raw)) / max(len(ja4_raw), 1)
    f["ja4_is_modern"] = int("h2" in ja4_raw)
    f["ja4_is_legacy"] = int("h1" in ja4_raw)
    f["ja4_entropy"] = string_entropy(ja4_raw)  #fix

    # 1. Define browser_claim first
    browser_claim = any(
        x in ua_low for x in ["chrome", "firefox", "edge", "safari", "brave", "samsung_internet"]
    )

    # Define ja4_match. Ensures we only flag a mismatch if we actually HAVE an edge-observed signature to compare against.
    f["ja4_match"] = int(
        tls.get("trust_score", 0) > 0 and 
        ja4_raw != "unknown" and 
        ja4_raw == ja4_digest
    )

    # 2. Then calculate mismatch
    f["ua_ja4_mismatch"] = int(
        f["ja4_unknown"] == 0 and
        browser_claim and
        f["ja4_match"] == 0
    )

    f["is_edge_observed"] = int(tls.get("method") == "vercel_edge_observed")
    
    # ================================
    # ANTI-LAZY-LEARNING FEATURES
    # ================================

    f["sec_ch_ua_suspicion_score"] = (
        int(sec_ch.lower() == "missing") *
        int(f["ua_is_chrome"] == 1)
    )

    f["header_completeness_score"] = (
        f["accept_lang_present"] +
        f["sec_ch_ua_present"]
    )

    f["network_consistency_score"] = (
        3 - min(f["forwarded_for_count"], 3)
    )

    # ------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------
    cores_raw = device.get("cores", 0)
    memory_raw = device.get("memory", 0)
    platform = str(device.get("platform", "unknown"))

    f["cpu_cores"] = safe_float(cores_raw)
    f["device_memory"] = safe_float(memory_raw)

    f["cpu_cores_missing"] = int(
        str(cores_raw).strip().lower() in
        ["unknown", "undefined", "null", "nan", ""]
        or cores_raw is None
    )

    f["device_memory_missing"] = int(
        str(memory_raw).strip().lower() in
        ["unknown", "undefined", "null", "nan", ""]
        or memory_raw is None
    )

    f["platform"] = platform

    platform_low = platform.lower()

    f["platform_is_windows"] = int("win" in platform_low)
    f["platform_is_mac"] = int("mac" in platform_low)
    f["platform_is_linux"] = int("linux" in platform_low)
    f["platform_is_mobile"] = int(
        any(x in platform_low for x in
            ["android", "iphone", "ipad", "ios", "mobile"])
    )

    f["device_platform_mismatch"] = int(
        (f["mobile_vs_desktop"] == 1 and f["platform_is_mobile"] == 0) or
        (f["mobile_vs_desktop"] == 0 and f["platform_is_mobile"] == 1)
    )

    # ------------------------------------------------------
    # SESSION
    # ------------------------------------------------------
    f["move_count"] = len(mouse)
    f["key_event_count"] = len(keys)
    f["scroll_count"] = len(scrolls)

    f["total_events"] = len(mouse) + len(keys) + len(scrolls)

    timestamps = get_timestamps(mouse, keys, scrolls)

    if len(timestamps) > 1:
        f["session_duration"] = max(timestamps) - min(timestamps)
    else:
        f["session_duration"] = 0

    dur = f["session_duration"]
    dur_safe = max(dur, 1)

    f["mouse_density"] = len(mouse) / dur_safe
    f["key_density"] = len(keys) / dur_safe

    # ------------------------------------------------------
    # MOUSE BIOMETRICS
    # ------------------------------------------------------
    velocities = []
    accelerations = []
    pauses = []
    segments = []
    angles = []

    if len(mouse) >= 2:

        for i in range(1, len(mouse)):
            p1 = mouse[i - 1]
            p2 = mouse[i]

            dt = safe_float(p2.get("t")) - safe_float(p1.get("t"))
            d = euclid(p1, p2)

            if dt > 0:
                velocities.append(d / dt)

                if dt > 300:
                    pauses.append(dt)

            segments.append(d)

        for i in range(1, len(velocities)):
            accelerations.append(abs(velocities[i] - velocities[i - 1]))

        for i in range(2, len(mouse)):
            a1 = math.atan2(
                safe_float(mouse[i-1].get("y")) - safe_float(mouse[i-2].get("y")),
                safe_float(mouse[i-1].get("x")) - safe_float(mouse[i-2].get("x"))
            )

            a2 = math.atan2(
                safe_float(mouse[i].get("y")) - safe_float(mouse[i-1].get("y")),
                safe_float(mouse[i].get("x")) - safe_float(mouse[i-1].get("x"))
            )

            diff = abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi)
            angles.append(diff)

        path = sum(segments)
        straight = euclid(mouse[0], mouse[-1])

        f["path_length"] = path
        f["straight_line_distance"] = straight
        f["straightness_ratio"] = straight / path if path > 0 else 1

    else:
        f["path_length"] = 0
        f["straight_line_distance"] = 0
        f["straightness_ratio"] = 1

    f["mean_velocity"] = safe_mean(velocities)
    f["std_velocity"] = safe_std(velocities)
    f["max_velocity"] = safe_max(velocities)
    f["min_velocity"] = safe_min(velocities)
    f["median_velocity"] = safe_median(velocities)

    f["mean_acceleration"] = safe_mean(accelerations)
    f["std_acceleration"] = safe_std(accelerations)

    f["pause_count"] = len(pauses)
    f["idle_time_mean"] = safe_mean(pauses)

    f["direction_changes"] = sum(1 for x in angles if x > 1.0)

    f["mouse_entropy"] = entropy(velocities)
    f["acceleration_entropy"] = entropy(accelerations)

    # ------------------------------------------------------
    # KEYBOARD
    # ------------------------------------------------------
    downs = [k for k in keys if str(k.get("type", "")).lower() == "down"]

    dwell = []
    flight = []
    active = {}
    overlap = 0

    for e in keys:

        typ = str(e.get("type", "")).lower()
        key = str(e.get("key", "")).lower()
        t = safe_float(e.get("t"))

        if typ == "down":

            if len(active) > 0 and key not in active:
                overlap += 1

            active[key] = t

        elif typ == "up":

            if key in active:
                dwell.append(t - active[key])
                del active[key]

    for i in range(1, len(downs)):
        gap = safe_float(downs[i].get("t")) - safe_float(downs[i - 1].get("t"))
        flight.append(max(0, gap))

    f["key_press_count"] = len(downs)

    f["avg_dwell_time"] = safe_mean(dwell)
    f["std_dwell_time"] = safe_std(dwell)

    f["avg_flight_time"] = safe_mean(flight)
    f["std_flight_time"] = safe_std(flight)

    f["key_overlap_count"] = overlap

    f["backspace_count"] = sum(
        1 for k in keys
        if str(k.get("key", "")).lower() in ["backspace", "delete"]
    )

    f["has_backspace"] = int(f["backspace_count"] > 0)
    f["multi_key_overlap"] = int(overlap > 0)

    f["backspace_ratio"] = f["backspace_count"] / max(len(downs), 1)

    f["unique_dwell_ratio"] = (
        len(set(round(x, 2) for x in dwell)) / len(dwell)
        if len(dwell) else 0
    )

    f["typing_speed_wpm"] = ((len(downs) / 5) / (dur / 60000)) if dur > 1000 else 0

    keys_pressed = [
        str(k.get("key", "")).lower()
        for k in keys
        if str(k.get("type", "")).lower() == "down"
    ]

    f["unique_key_ratio"] = len(set(keys_pressed)) / max(len(downs), 1)

    # ------------------------------------------------------
    # CROSS MODAL
    # ------------------------------------------------------
    f["mouse_to_key_ratio"] = len(mouse) / (len(downs) + 1)
    f["events_per_second"] = f["total_events"] / max(dur / 1000, 1)

    f["click_count"] = len(downs)
    f["click_interval_mean"] = f["avg_flight_time"]
    f["click_hold_mean"] = f["avg_dwell_time"]

    f["pause_ratio"] = f["pause_count"] / max(f["move_count"], 1)
    f["scroll_to_move_ratio"] = f["scroll_count"] / max(f["move_count"], 1)

    # ================================
    # LEAKAGE SHIELD APPLY STEP
    # ================================
    f["_weak_signal_flag"] = 1

    # ================================
    # LEAKAGE SHIELD APPLY STEP
    # ================================
    f = sanitize_weak_signals(f) # ADD THIS CALL

    return f

# ==========================================================
# BUILD DATASET
# ==========================================================
rows = []

print("=" * 60)
print(" SENTINEL-BOT FEATURE EXTRACTOR v9.0 ")
print("=" * 60)

for file in os.listdir(INPUT_DIR):

    if file.endswith(".json"):

        try:
            with open(
                os.path.join(INPUT_DIR, file),
                "r",
                encoding="utf-8"
            ) as fp:

                data = json.load(fp)
                rows.append(extract_features(data, file))

        except Exception:

            with open(ERROR_LOG, "a", encoding="utf-8") as log:
                log.write(f"\nFILE: {file}\n")
                log.write(traceback.format_exc())
                log.write("\n---------------------------\n")

# DATAFRAME
df = pd.DataFrame(rows)

# ==========================================
# FINAL LEAKAGE NORMALIZATION PASS
# ==========================================
def global_feature_normalization(df):

    # prevent identity leakage from raw strings
    if "raw_userAgent" in df.columns:
        df["raw_userAgent"] = df["raw_userAgent"].apply(lambda x: "masked")

    if "source_file" in df.columns:
        df["source_file"] = 0  # remove memorization path

    return df

df = global_feature_normalization(df)

df.drop_duplicates(inplace=True)

# LABEL ENCODING
for col in ["browser", "ja4_prefix", "platform", "bot_subclass"]:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna("unknown")
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col])

# SAVE
df.to_csv(OUTPUT_CSV, index=False)

print("MASTER DATASET CREATED SUCCESSFULLY")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Saved:", OUTPUT_CSV)
print("DONE.")