import requests
import time
import uuid
import random
import numpy as np

# ==========================================================
# CONFIG
# ==========================================================
TARGET_URL = "https://sentinel-bot-data-collection.vercel.app/api/upload"
SESSIONS_TO_GENERATE = 40

random.seed()
np.random.seed()

# ==========================================================
# SESSION CLOCK
# ==========================================================
def init_session_clock():
    roll = random.random()
    
    if roll < 0.70:
        # 70% of users are focused and act immediately (0.5s to 3s)
        return random.uniform(500, 3000)
    elif roll < 0.90:
        # 20% of users orient themselves first (3s to 8s)
        return random.uniform(3000, 8000)
    else:
        # 10% of users are distracted, opened in a new tab, or reading (8s to 60s)
        return random.uniform(8000, 60000)

# ==========================================================
# PERSONA ENGINE
# ==========================================================
def generate_persona():
    speed_profile = random.choice([
        ("very_slow", 2.0, 6.0),
        ("slow", 1.5, 2.5),
        ("normal", 1.0, 2.0),
        ("fast", 0.7, 1.3),
        ("very_fast", 0.5, 1.0)
    ])

    speed, low, high = speed_profile

    return {
        "speed": speed,
        "time_scale": random.uniform(low, high),
        "precision": random.uniform(0.7, 1.3),
        "error_rate": random.uniform(0.02, 0.15)
    }

# ==========================================================
# LANGUAGE
# ==========================================================
def generate_human_language():
    base_langs = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9,en-US;q=0.8",
        "ar-SA,ar;q=0.9,en-US;q=0.8,en;q=0.7",
        "ar-AE,ar;q=0.9,en-GB;q=0.8,en;q=0.7",
        "hi-IN,hi;q=0.9,en-US;q=0.8,en;q=0.7",
        "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
        "es-ES,es;q=0.9,en;q=0.8"
    ]

    lang = random.choice(base_langs)

    if random.random() < 0.4:
        lang = lang.replace("0.9", str(round(random.uniform(0.85, 0.95), 2)))
    if random.random() < 0.4:
        lang = lang.replace("0.8", str(round(random.uniform(0.75, 0.85), 2)))

    return lang

# ==========================================================
# BROWSER PROFILE
# ==========================================================
def generate_browser_profile():

    chrome_build = f"{random.randint(120,146)}.0.{random.randint(1000,7000)}.{random.randint(10,200)}"
    edge_build = f"{random.randint(120,146)}.0.{random.randint(1000,3000)}.{random.randint(10,100)}"

    profiles = [
        {
            "browser": "Chrome",
            "platform": "Win32",
            "ua": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  f"(KHTML, like Gecko) Chrome/{chrome_build} Safari/537.36",
            "cores": [4, 8, 12, 16],
            "memory": [8, 16, 32]
        },
        {
            "browser": "Edge",
            "platform": "Win32",
            "ua": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  f"(KHTML, like Gecko) Chrome/{chrome_build} Safari/537.36 Edg/{edge_build}",
            "cores": [8, 12, 16],
            "memory": [8, 16]
        },
        {
            "browser": "Chrome",
            "platform": "MacIntel",
            "ua": f"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  f"(KHTML, like Gecko) Chrome/{chrome_build} Safari/537.36",
            "cores": [8, 10],
            "memory": [8, 16]
        },
        {
            "browser": "Chrome",
            "platform": "Linux x86_64",
            "ua": f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  f"(KHTML, like Gecko) Chrome/{chrome_build} Safari/537.36",
            "cores": [4, 8],
            "memory": [4, 8, 16]
        }
    ]

    profile = random.choice(profiles)
    profile["lang"] = generate_human_language()
    return profile

# ==========================================================
# JA4
# ==========================================================
JA4_POOL = {
    "Chrome": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t12d1208h2_9e6316305715_2dae41c691ec",
        "t13d1517h2_8daaf6152771_b6f405a00624"
    ],
    "Edge": [
        "t13d1516h2_8daaf6152771_d8a2da3f94cd",
        "t13d1517h2_8daaf6152771_b6f405a00624",
        "t12d1208h2_9e6316305715_2dae41c691ec"
    ]
}

def generate_fake_ja4():
    return f"t13d{random.randint(1000,9999)}h2_{uuid.uuid4().hex[:12]}_{uuid.uuid4().hex[:12]}"

def build_tls_layer(profile):

    browser = profile["browser"]
    roll = np.random.beta(2, 2)

    if roll < 0.15:
        ja4_raw = "unknown"
    elif roll < 0.55:
        ja4_raw = random.choice(JA4_POOL[browser])
    elif roll < 0.8:
        ja4_raw = random.choice(JA4_POOL[random.choice(list(JA4_POOL.keys()))])
    else:
        ja4_raw = generate_fake_ja4()

    # FIX 4: better entropy (UUID suffix instead of small integer)
    if random.random() < 0.3:
        ja4_raw += "_" + uuid.uuid4().hex[:2]

    return {
        "browser": browser,
        "ja4": ja4_raw,
        "ja4_raw": ja4_raw,
        "ja4_digest": ja4_raw,
        "method": "synthetic_enrichment",
        "trust_score": np.clip(np.random.normal(0.75, 0.2), 0.3, 1.0)
    }

# ==========================================================
# MOUSE
# ==========================================================
def generate_mouse(base_start, persona):

    moves = []
    current = base_start + random.randint(100, 500)

    time_scale = persona["time_scale"]

    x, y = random.uniform(0.1,0.3), random.uniform(0.1,0.3)

    for _ in range(random.randint(2,4)):

        tx, ty = random.uniform(0.3,0.95), random.uniform(0.3,0.95)
        ox, oy = tx + random.uniform(0.02,0.06), ty + random.uniform(0.02,0.06)

        for step in np.linspace(0,1,random.randint(25,60)):
            curve = step**random.uniform(0.8, 1.5)

            nx = x + (ox-x)*curve + random.uniform(-0.002,0.002)
            ny = y + (oy-y)*curve + random.uniform(-0.002,0.002)

            moves.append({"x":round(nx,4),"y":round(ny,4),"t":round(current,2)})
            current += int(random.randint(5,25) * time_scale * random.uniform(0.7,1.3))

        # FIX 1: restored post-move jitter
        if random.random() < 0.6:
            for _ in range(random.randint(8,20)):
                moves.append({
                    "x": round(tx + random.uniform(-0.008,0.008),4),
                    "y": round(ty + random.uniform(-0.008,0.008),4),
                    "t": round(current,2)
                })
                current += int(random.randint(30,120) * time_scale)

        # FIX 6: idle gap (mouse)
        if random.random() < 0.08:
            current += int(random.randint(300,1200) * time_scale)

        x, y = tx, ty

    return moves, current

# ==========================================================
# SCROLL
# ==========================================================
def generate_scroll(base_start, persona):

    scrolls = []
    t = base_start + random.randint(800, 2000)
    time_scale = persona["time_scale"]

    for _ in range(random.randint(3,10)):

        burst = random.random() < 0.3

        # FIX 2: clean burst logic
        repeat = random.randint(1,3) if burst else 1

        for _ in range(repeat):
            if random.random() < 0.85:
                delta = int(np.random.normal(200, 80))
            else:
                delta = random.randint(-150, -20)

            scrolls.append({"deltaY": delta, "t": round(t,2)})
            t += int(random.randint(80, 300) * time_scale)

        t += int(random.randint(200, 800) * time_scale)

        # FIX 6: idle gap (scroll)
        if random.random() < 0.08:
            t += int(random.randint(300,1200) * time_scale)

    return scrolls

# ==========================================================
# KEYSTROKES
# ==========================================================
TEXT_POOL = [
    "download annual report from portal",
    "Security dashboard login failed retry",
    "upload final document to server",
    "meeting notes updated and shared",
    "Customer account verification required",
    "Hello how are you doing today",
    "Woke up to bomb alerts again",
    "Went on a roadtrip with friends",
    "js8#K2l@pQ9!",               # Password
    "A7x$9Lm2",                   # Short ID
    "1234567890",                 # Number row spam
    "password123!",               # Common password
    "asdfjkl;asdfjkl;",           # Restless keyboard mashing
    "admin@company-portal.local", # Email
    "example@email.com"
]

COMMON_BIGRAMS = ["th","he","in","er","an","re"]

def generate_keys(start_time, persona):

    keys = []
    text = random.choice(TEXT_POOL)

    time_scale = persona["time_scale"]
    error_rate = persona["error_rate"]

    current = start_time + int(random.randint(400, 1200) * time_scale)

    for i, ch in enumerate(text):

        if random.random() < 0.1:
            current += int(random.randint(300,1000) * time_scale)

        # FIX 6: idle gap (keys)
        if random.random() < 0.08:
            current += int(random.randint(300,1200) * time_scale)

        if random.random() < error_rate:
            wrong = random.choice(list("abcdefghijklmnopqrstuvwxyz0123456789@#$%&"))
            keys.append({"key":wrong,"type":"down","t":round(current,2)})
            keys.append({"key":wrong,"type":"up","t":round(current+90,2)})
            current += int(random.randint(120,250) * time_scale)

            keys.append({"key":"Backspace","type":"down","t":round(current,2)})
            keys.append({"key":"Backspace","type":"up","t":round(current+80,2)})
            current += int(random.randint(120,250) * time_scale)

        mode = random.choice(["fast", "normal", "slow"])
        dwell = np.random.normal(60,30) if mode=="fast" else np.random.normal(110,60) if mode=="normal" else np.random.normal(180,90)
        dwell = int(max(20, min(dwell * time_scale, 400)))

        flight = random.randint(20,60) if i>0 and text[i-1:i+1] in COMMON_BIGRAMS else random.randint(80,250)
        flight = int(flight * time_scale)

        keys.append({"key":ch,"type":"down","t":round(current,2)})
        keys.append({"key":ch,"type":"up","t":round(current+dwell,2)})

        if random.random() < 0.35 and i < len(text)-1:
            overlap = int(np.random.exponential(scale=dwell/2))
            overlap = max(5, min(overlap, dwell-5))
            current += (dwell - overlap)
        else:
            current += (dwell + flight)

    return keys

# ==========================================================
# PAYLOAD
# ==========================================================
def generate_payload():

    profile = generate_browser_profile()
    persona = generate_persona()
    base_start = init_session_clock()

    mouse, _ = generate_mouse(base_start, persona)

    # FIX 5: timeline offsets
    keys = generate_keys(base_start + random.randint(0,1500), persona)
    scrolls = generate_scroll(base_start + random.randint(0,2000), persona)

    # RESTORED: Session modality drops (Reader vs Typist personas)
    mode = random.random()
    if mode < 0.15:
        keys = []
    elif mode < 0.30:
        mouse = []

    # RESTORED: Partial session simulation (User abandons page midway)
    if random.random() < 0.2:
        cutoff = base_start + random.randint(2000, 5000)
        mouse = [m for m in mouse if m["t"] <= cutoff]
        keys = [k for k in keys if k["t"] <= cutoff]

    # RESTORED: Prevent fully empty sessions
    if not mouse and not keys:
        mouse, _ = generate_mouse(base_start, persona)

    payload = {
        "sessionId": str(uuid.uuid4()),
        "label": random.choice(["adversarial_bot","evasive_bot","smart_bot"]),
        "mouseMovements": sorted(mouse, key=lambda x: x["t"]),
        "keystrokes": sorted(keys, key=lambda x: x["t"]),
        "scrollEvents": sorted(scrolls, key=lambda x: x["t"]),
        "device": {
            "cores": random.choice(profile["cores"]),
            "memory": random.choice(profile["memory"]),
            "platform": profile["platform"]
        },
        "network_layer": {
            "userAgent": profile["ua"]
        },
        "tls_layer": build_tls_layer(profile)
    }

    headers = {
        "User-Agent": profile["ua"],
        "Accept-Language": profile["lang"],
        "Content-Type": "application/json",
        "Origin": "https://sentinel-bot-data-collection.vercel.app",
        "Referer": "https://sentinel-bot-data-collection.vercel.app/"
    }

    # FIX 3: header-TLS correlation
    try:
        major_version = profile["ua"].split("Chrome/")[1].split(".")[0]
        headers["sec-ch-ua"] = f'"Chromium";v="{major_version}", "Google Chrome";v="{major_version}"'
    except:
        pass

    # RESTORED: Platform header entropy
    if random.random() < 0.7:
        headers["sec-ch-ua-platform"] = random.choice([
            '"Windows"', '"macOS"', '"Linux"'
        ])

    return payload, headers

# ==========================================================
# RUN
# ==========================================================
def run_session(i):
    payload, headers = generate_payload()

    try:
        print(f"[{i}] Uploading {payload['sessionId']}")
        r = requests.post(TARGET_URL, json=payload, headers=headers, timeout=20)
        print("Status:", r.status_code)
    except Exception as e:
        print("Error:", e)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    print("="*60)
    print("SENTINEL BOT ATTACKER v4 ")
    print("="*60)

    for i in range(1, SESSIONS_TO_GENERATE + 1):
        run_session(i)
        time.sleep(random.uniform(1.5, 5.5))

    print("DONE.")