import os
import json
import random

random.seed()

# PATHS
INPUT_DIR = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\raw"
OUTPUT_DIR = r"C:\Users\aamin\Professional\Datasets\Sentinel-Bot human detection\enriched"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# JA4 FINGERPRINT POOLS
# =====================================================
JA4_POOL = {
    "Chrome_Desktop": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t12d1208h2_9e6316305715_2dae41c691ec",
        "t13d1517h2_8daaf6152771_b6f405a00624"
    ],

    "Chrome_Mobile": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t13d1517h2_8daaf6152771_b6f405a00624",
        "t12d1208h2_9e6316305715_2dae41c691ec"
    ],

    "Firefox": [
        "t13d1717h2_5b57614c22b0_3cbfd9057e0d",
        "t13d1717h2_5b57614c22b0_e6dcd7ae0a9e",
        "t13d2013h2_2b729b4bf6f3_e24568c0d440",
        "t12d1411h2_c866b44c5a26_810e2f290f6f",
        "t13d1718h2_5b57614c22b0_e7cacf613b58"
    ],

    "Edge": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t13d1517h2_8daaf6152771_b6f405a00624",
        "t12d1208h2_9e6316305715_2dae41c691ec"
    ],

    "Brave": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t12d180800_4b22cbed5bed_7af1ed941c26",
        "t12d1809h2_4b22cbed5bed_7af1ed941c26",
        "t13d3108h2_028448f1082c_ef9b7367c32e"
    ],

    "Brave_Mobile": [
        "t12d1208h2_9e6316305715_2dae41c691ec",
        "t13d1516h2_8daaf6152771_d8a2da3f94cd"
    ],

    "Opera": [
        "t12d1208h2_9e6316305715_2dae41c691ec",
        "t13d3108h2_028448f1082c_7d976077755b"
    ],

    "Safari_Desktop": [
        "t13d2013h2_2b729b4bf6f3_e24568c0d440"
    ],

    "Safari_Mobile": [
        "t13d1516h2_8daaf6152771_390299696238",
        "t13d2013h2_2b729b4bf6f3_e24568c0d440"
    ],

    "Samsung_Internet": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd"
    ],

    "Bot_Scripted": [
        "t13d3112h1_e8f1e7e78f70_b26ce05bbdd6"
    ],

    "Bot_Automation": [
        "t13d3112h1_e8f1e7e78f70_b26ce05bbdd6"
    ],

    "Bot_Spoofed": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t12d1208h2_9e6316305715_2dae41c691ec",
        "t13d1517h2_8daaf6152771_b6f405a00624"
    ]
}

# =====================================================
# USER AGENT DETECTION
# =====================================================
def detect_browser(user_agent):
    ua = user_agent.lower()

    if "python-requests" in ua or "curl" in ua or "wget" in ua:
        return "Bot_Scripted"

    if "selenium" in ua or "headless" in ua or "puppeteer" in ua or "playwright" in ua:
        return "Bot_Automation"

    if "samsungbrowser" in ua:
        return "Samsung_Internet"

    if "firefox" in ua:
        return "Firefox"

    if "opr/" in ua or "opera" in ua:
        return "Opera"

    if "brave" in ua:
        return "Brave_Mobile" if "mobile" in ua else "Brave"

    if "edg/" in ua or "edga/" in ua or "edgios" in ua:
        return "Edge"

    if "chrome" in ua or "crios" in ua:
        return "Chrome_Mobile" if ("mobile" in ua or "android" in ua) else "Chrome_Desktop"

    if "safari" in ua and "chrome" not in ua:
        return "Safari_Mobile" if ("iphone" in ua or "ipad" in ua or "mobile" in ua) else "Safari_Desktop"

    return "Unknown"

# =====================================================
# FIXED: SAFE JA4 ASSIGNMENT
# =====================================================
def assign_ja4(browser_name):
    pool = JA4_POOL.get(browser_name, None)
    if pool:
        return random.choice(pool)
    return "unknown"

# =====================================================
# HUMAN TLS (your original logic kept intact if you had it elsewhere)
# =====================================================
def assign_human_tls(browser_name):
    roll = random.random()

    # --- FIX: Simulate real Vercel Edge humans ---
    if roll < 0.55: # 55% of humans now look like real Vercel traffic
        ja4 = assign_ja4(browser_name)
        return {
            "browser": browser_name,
            "ja4_raw": ja4,
            "ja4_digest": ja4,
            "method": "vercel_edge_observed", # MUST match upload.js
            "trust_score": 1.0                # MUST match upload.js
        }

    elif roll < 0.85:
        return {
            "browser": browser_name,
            "ja4_raw": "unknown",
            "ja4_digest": "unknown",
            "method": "synthetic_enrichment",
            "trust_score": random.uniform(0.2, 0.5)
        }

    elif roll < 0.93:
        ja4 = assign_ja4(browser_name)
        return {
            "browser": browser_name,
            "ja4_raw": ja4,
            "ja4_digest": ja4,
            "method": "synthetic_enrichment",
            "trust_score": random.uniform(0.6, 0.85)
        }

    elif roll < 0.98:
        other = random.choice(list(JA4_POOL.keys()))
        ja4 = assign_ja4(other)
        return {
            "browser": browser_name,
            "ja4_raw": ja4,
            "ja4_digest": ja4,
            "method": "synthetic_enrichment",
            "trust_score": random.uniform(0.5, 0.8)
        }

    else:
        return {
            "browser": browser_name,
            "ja4_raw": "unknown",
            "ja4_digest": "unknown",
            "method": "synthetic_enrichment",
            "trust_score": random.uniform(0.0, 0.3)
        }

# =====================================================
# BOT STRATEGY (UNCHANGED)
# =====================================================
def classify_bot(label):
    if label == "smart_bot":
        return "Bot_Spoofed" if random.random() < 0.75 else "Bot_Automation"
    elif label == "evasive_bot":
        return "Bot_Spoofed" if random.random() < 0.60 else "Bot_Automation"
    else:
        roll = random.random()
        if roll < 0.35:
            return "Bot_Automation"
        elif roll < 0.55:
            return "Bot_Scripted"
        return "Bot_Spoofed"

# =====================================================
# MAIN PROCESSING (FIXED BUGS ONLY)
# =====================================================
total = 0
unknown_count = 0

for filename in os.listdir(INPUT_DIR):

    if not filename.endswith(".json"):
        continue

    total += 1

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ua = data.get("network_layer", {}).get("userAgent", "")
    label = data.get("metadata", {}).get("label", "").lower()

    browser_type = detect_browser(ua)

    synthetic_ja4 = "unknown"   # FIX 1: ALWAYS INITIALIZED

    if label in ["bot", "evasive_bot", "smart_bot"]:
        browser_type = classify_bot(label)
        synthetic_ja4 = assign_ja4(browser_type)

        tls_layer = {
            "browser": browser_type,
            "ja4_raw": synthetic_ja4,
            "ja4_digest": synthetic_ja4,
            "method": "synthetic_enrichment",
            "trust_score": 0
        }

    else:
        tls_layer = assign_human_tls(browser_type)

    # FIX 2: safe check
    if data.get("tls_layer", {}).get("ja4_raw", "unknown") == "unknown":
        unknown_count += 1

    data["tls_layer"] = tls_layer

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[{total}] {filename} --> {label} --> {browser_type}")

print("\n========== SUMMARY ==========")
print("Total Processed :", total)
print("Unknown Mappings:", unknown_count)
print("Output Folder   :", OUTPUT_DIR)