# ======================================================================
# SENTINEL-BOT
# predict_live.py v5.2
# Final Hardened Runtime Inference Engine
# ======================================================================

import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"

import json,time,uuid,math,hashlib,asyncio,logging,secrets
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict,Any

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI,HTTPException,Request,Header,Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel,Field,field_validator
from prometheus_client import Counter, Gauge,Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import Response

import runtime_config as settings

# ======================================================================
# PATH INIT
# ======================================================================

os.makedirs(settings.LOG_DIR,exist_ok=True)
os.makedirs(settings.METRICS_DIR,exist_ok=True)

# ======================================================================
# OPTIONAL SETTINGS
# ======================================================================

MAX_CONTEXTUAL_FEATURES=getattr(settings,"MAX_CONTEXTUAL_FEATURES",32)
MAX_CONTEXTUAL_STRING_LENGTH=getattr(settings,"MAX_CONTEXTUAL_STRING_LENGTH",128)
MAX_RATE_BUCKETS=getattr(settings,"MAX_RATE_BUCKETS",10000)
TRUST_PROXY_HEADERS=getattr(settings,"TRUST_PROXY_HEADERS",False)
EXPOSE_MODEL_BREAKDOWN=getattr(settings,"EXPOSE_MODEL_BREAKDOWN",False)

# ======================================================================
# LOGGING
# ======================================================================

logger=logging.getLogger("sentinel-runtime")
logger.setLevel(getattr(logging,settings.LOG_LEVEL.upper()))

# Encoding Boundary
log_handler=RotatingFileHandler(
    os.path.join(settings.LOG_DIR,"predict_live.log"),
    maxBytes=10*1024*1024,
    backupCount=5,
    encoding=getattr(settings, "LOG_ENCODING", "utf-8")
)

log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.propagate = False
if not logger.handlers:
    logger.addHandler(log_handler)

# ======================================================================
# PROMETHEUS
# ======================================================================

registry=CollectorRegistry()

PROM_REQUESTS_TOTAL=Counter(
    "sentinel_requests_total",
    "Total inbound requests",
    registry=registry
)

PROM_ALLOWED_TOTAL=Counter(
    "sentinel_allowed_total",
    "Allowed requests",
    registry=registry
)

PROM_BLOCKED_TOTAL=Counter(
    "sentinel_blocked_total",
    "Blocked requests",
    registry=registry
)

PROM_FAILED_TOTAL=Counter(
    "sentinel_failed_total",
    "Internal failures",
    registry=registry
)

PROM_RATE_LIMITED_TOTAL=Counter(
    "sentinel_rate_limited_total",
    "Rate limited requests",
    registry=registry
)

PROM_OVERLOAD_TOTAL=Counter(
    "sentinel_overload_total",
    "Overload rejected requests",
    registry=registry
)

PROM_VALIDATION_FAILURES_TOTAL=Counter(
    "sentinel_validation_failures_total",
    "Validation failures",
    registry=registry
)

PROM_ACTIVE_INFERENCE=Gauge(
    "sentinel_active_inference",
    "Current active inference",
    registry=registry
)

PROM_LATENCY=Histogram(
    "sentinel_inference_latency_seconds",
    "Inference latency",
    registry=registry,
    buckets=(0.005,0.01,0.025,0.05,0.1,0.25,0.5,1,2,5)
)

# ---- MULTIMODAL NOVELTY RESEARCH SIGNALS ----
# 1. JA4 Fingerprint Tracking Layer
PROM_JA4_OBSERVED = Counter(
    "sentinel_ja4_observed_total",
    "Track occurrences of unique JA4 prefixes",
    ["ja4_prefix"],
    registry=registry
)

PROM_JA4_HUMAN = Counter("sentinel_human_ja4_total", "Total legitimate human JA4 requests", registry=registry)
PROM_JA4_BOT = Counter("sentinel_bot_ja4_total", "Total adversarial bot JA4 requests", registry=registry)

# 2. Ensemble Component Confidence Indicators
PROM_XGB_PROB = Gauge("sentinel_xgb_probability", "Current XGBoost inference confidence", registry=registry)
PROM_RF_PROB = Gauge("sentinel_rf_probability", "Current Random Forest inference confidence", registry=registry)
PROM_SVM_PROB = Gauge("sentinel_svm_probability", "Current SVM inference confidence", registry=registry)
PROM_ENSEMBLE_PROB = Gauge("sentinel_ensemble_probability", "Current Stacked Meta-Learner probability", registry=registry)

# TRACKER: Buckets live scores into decimal ranges from 0.0 to 1.0
PROM_ENSEMBLE_PROB_DIST = Histogram(
    "sentinel_ensemble_probability_distribution",
    "Frequency distribution of composite stacking meta-learner output probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

# 3. Behavioral Telemetry Summary Statistics
PROM_BIOMETRIC_MOUSE_ENTROPY = Gauge("sentinel_mouse_entropy", "Rolling average of human/bot mouse curve entropy", registry=registry)
PROM_BIOMETRIC_DWELL_TIME = Gauge("sentinel_avg_dwell_time", "Rolling average of keystroke hold duration", registry=registry)
PROM_BIOMETRIC_FLIGHT_TIME = Gauge("sentinel_avg_flight_time", "Rolling average of key-to-key travel times", registry=registry)
PROM_BIOMETRIC_DIRECTION_CHANGES = Gauge("sentinel_direction_changes", "Rolling average of mouse trace vector alterations", registry=registry)

# ======================================================================
# SHA256 VALIDATION
# Model binaries are verified pre-execution via SHA-256 validation hashes. 
# Enforcing cryptographic checksum constraints prevents malicious code payload execution via joblib/pickle deserialization vectors.
# ======================================================================
def sha256_file(path):
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

# Ensure the validation matches central configuration state directives smoothly
if getattr(settings, "REQUIRE_MODEL_INTEGRITY", False):
    current_hash = sha256_file(settings.MODEL_PATH)

    if current_hash.lower() != settings.EXPECTED_MODEL_SHA256.lower():
        logger.critical(json.dumps({
            "event": "integrity_failure",
            "model_path": settings.MODEL_PATH,
            "timestamp_utc": datetime.utcnow().isoformat()
        }))
        raise RuntimeError("[CRITICAL] Model integrity validation failed.")

# ======================================================================
# MODEL LOADING
# ======================================================================

logger.info(json.dumps({
    "event":"model_load_start",
    "model_path":settings.MODEL_PATH,
    "timestamp_utc":datetime.utcnow().isoformat()
}))

try:
    bundle=joblib.load(settings.MODEL_PATH)

except Exception as e:

    logger.critical(json.dumps({
        "event":"bundle_load_failure",
        "error":repr(e),
        "timestamp_utc":datetime.utcnow().isoformat()
    }))

    raise

required_keys=[
    "meta_model",
    "base_models",
    "safe_features",
    "meta_columns",
    "numeric_cols"
]

for key in required_keys:

    if key not in bundle:

        logger.critical(json.dumps({
            "event":"corrupted_bundle",
            "missing_key":key,
            "timestamp_utc":datetime.utcnow().isoformat()
        }))

        raise RuntimeError(f"[CRITICAL] Missing bundle key: {key}")

META_MODEL=bundle["meta_model"]
BASE_MODELS=bundle["base_models"]

required_base_models=["xgb","rf","svm"]

for model_key in required_base_models:
    if model_key not in BASE_MODELS:
        raise RuntimeError(f"[CRITICAL] Missing base model: {model_key}")

XGB_MODEL=BASE_MODELS["xgb"]
RF_MODEL=BASE_MODELS["rf"]
SVM_MODEL=BASE_MODELS["svm"]

SAFE_FEATURES=bundle["safe_features"]
META_COLUMNS=bundle["meta_columns"]
NUMERIC_COLS=bundle["numeric_cols"]

if not isinstance(NUMERIC_COLS,list):
    raise RuntimeError("[CRITICAL] numeric_cols malformed.")

if not all(isinstance(col,str) for col in NUMERIC_COLS):
    raise RuntimeError("[CRITICAL] numeric_cols invalid types.")

if len(NUMERIC_COLS) > 10000:
    raise RuntimeError("[CRITICAL] numeric_cols contains an excessive feature allocation size boundary.")

if not isinstance(SAFE_FEATURES,list):
    raise RuntimeError("[CRITICAL] safe_features malformed.")

if len(SAFE_FEATURES)>10000:
    raise RuntimeError("[CRITICAL] safe_features excessive size.")

if not all(isinstance(feature,str) for feature in SAFE_FEATURES):
    raise RuntimeError("[CRITICAL] safe_features invalid types.")

if not isinstance(META_COLUMNS, list):
    raise RuntimeError("[CRITICAL] meta_columns malformed.")

if not all(isinstance(col, str) for col in META_COLUMNS):
    raise RuntimeError("[CRITICAL] meta_columns invalid types.")

MODEL_THRESHOLD = bundle.get("threshold", settings.THRESHOLD)

if not (0.0 <= MODEL_THRESHOLD <= 1.0):
    raise RuntimeError("[CRITICAL] Invalid model threshold boundary out of scale limits.")

if not hasattr(XGB_MODEL,"predict_proba"):
    raise RuntimeError("[CRITICAL] XGB missing predict_proba.")

if not hasattr(RF_MODEL,"predict_proba"):
    raise RuntimeError("[CRITICAL] RF missing predict_proba.")

if not hasattr(SVM_MODEL,"predict_proba") and not hasattr(SVM_MODEL,"decision_function"):
    raise RuntimeError("[CRITICAL] SVM probability interface missing.")

logger.info(json.dumps({
    "event":"model_loaded",
    "system_version":settings.SYSTEM_VERSION,
    "model_version":settings.MODEL_VERSION,
    "features_count":len(NUMERIC_COLS),
    "threshold":MODEL_THRESHOLD,
    "timestamp_utc":datetime.utcnow().isoformat()
}))

# ======================================================================
# EXECUTION RESOURCES
# ======================================================================

executor=ThreadPoolExecutor(max_workers=settings.MAX_CONCURRENT_INFERENCE)

inference_semaphore=asyncio.Semaphore(settings.MAX_CONCURRENT_INFERENCE)

rate_buckets=defaultdict(deque)
rate_limit_lock=asyncio.Lock()
cleanup_task=None


# ======================================================================
# AUTH
# ======================================================================
async def authenticate_waf_request(x_api_key: str | None = Header(default=None)):
    expected_key=settings.VERIFY_API_KEY

    if expected_key:
        token = x_api_key if x_api_key else ""
        if not secrets.compare_digest(token, expected_key):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized request."
            )

# ======================================================================
# METRICS ENGINE
# ======================================================================

MAX_METRIC_FILES=50

class RuntimeMetrics:

    def __init__(self):
        self.total_requests=0
        self.allowed_requests=0
        self.blocked_requests=0
        self.failed_requests=0
        self.total_latency_ms=0.0
        self.lock=asyncio.Lock()

    async def update(self,allowed,latency_ms):

        async with self.lock:

            self.total_requests+=1
            self.total_latency_ms+=latency_ms

            if allowed:
                self.allowed_requests+=1
            else:
                self.blocked_requests+=1

    async def increment_failures(self):

        async with self.lock:
            self.failed_requests+=1

    async def snapshot(self):

        async with self.lock:

            avg_latency=(
                self.total_latency_ms/self.total_requests
                if self.total_requests>0
                else 0.0
            )

            return {
                "system_version":settings.SYSTEM_VERSION,
                "model_version":settings.MODEL_VERSION,
                "total_requests":self.total_requests,
                "allowed_requests":self.allowed_requests,
                "blocked_requests":self.blocked_requests,
                "failed_requests":self.failed_requests,
                "average_latency_ms":round(avg_latency,3),
                "timestamp_utc":datetime.utcnow().isoformat()
            }

    async def export_json(self):

        if not settings.EXPORT_JSON_METRICS:
            return

        path=os.path.join(
            settings.METRICS_DIR,
            f"runtime_metrics_{int(time.time())}.json"
        )

        snapshot=await self.snapshot()

        with open(path,"w") as f:
            json.dump(snapshot,f,indent=4)

metrics=RuntimeMetrics()

def cleanup_old_metrics():

    try:

        files=sorted(
            [
                os.path.join(settings.METRICS_DIR,f)
                for f in os.listdir(settings.METRICS_DIR)
                if f.endswith(".json")
            ],
            key=os.path.getmtime
        )

        while len(files)>MAX_METRIC_FILES:

            try:
                os.remove(files.pop(0))
            except OSError:
                pass

    except Exception as e:

        logger.warning(json.dumps({
            "event":"metrics_cleanup_warning",
            "error":str(e),
            "timestamp_utc":datetime.utcnow().isoformat()
        }))

# ======================================================================
# PAYLOAD SCHEMA
# ======================================================================

class InferencePayload(BaseModel):
    # Optional Pre-Computed Fields (Used by simulation_engine.py for stress benchmarks)
    straightness_ratio: float | None = Field(default=None, ge=0.0, le=10.0)
    mean_velocity: float | None = Field(default=None, ge=0.0, le=1e6)
    std_velocity: float | None = Field(default=None, ge=0.0, le=1e6)
    direction_changes: float | None = Field(default=None, ge=0.0, le=1e6)
    mouse_entropy: float | None = Field(default=None, ge=0.0, le=20.0)
    acceleration_entropy: float | None = Field(default=None, ge=0.0, le=20.0)
    avg_dwell_time: float | None = Field(default=None, ge=0.0, le=1e6)
    std_dwell_time: float | None = Field(default=None, ge=0.0, le=1e6)
    avg_flight_time: float | None = Field(default=None, ge=0.0, le=1e6)
    std_flight_time: float | None = Field(default=None, ge=0.0, le=1e6)
    key_overlap_count: float | None = Field(default=None, ge=0.0, le=1e6)
    backspace_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    unique_dwell_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    unique_key_ratio: float | None = Field(default=None, ge=0.0, le=1.0)

    # Raw Biometric Tracking Layers
    mouseMovements: list[Dict[str, Any]] = Field(default_factory=list)
    keystrokes: list[Dict[str, Any]] = Field(default_factory=list)
    scrollEvents: list[Dict[str, Any]] = Field(default_factory=list)
    
    # Add network_layer map to prevent Pydantic data truncation drops
    network_layer: Dict[str, Any] = Field(default_factory=dict)
    
    device: Dict[str, Any] = Field(default_factory=dict)
    contextual_features: Dict[str, Any] = Field(default_factory=dict)
    tls_layer: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("*")
    @classmethod
    def validate_finite_inputs(cls, value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite value detected.")
        return value

# ======================================================================
# MIDDLEWARE
# ======================================================================

class BodySizeLimitMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if "application/json" not in content_type.lower():
                return JSONResponse(status_code=415, content={"detail": "Unsupported media type."})

            content_length = request.headers.get("content-length")
            if content_length:
                # HARDENING AMENDMENT: Safe validation wrapper prevents unhandled ValueErrors
                try:
                    parsed_length = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Malformed content length definition header payload."}
                    )

                if parsed_length > settings.MAX_BODY_SIZE:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large."}
                    )

        # Proceed with request pipeline resolution
        response = await call_next(request)

        # HARDENING AMENDMENT: Application perimeter security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store, max-age=0"

        return response

# ======================================================================
# MASTER INFERENCE FEATURE ENGINEERING EXTRACTOR
# Derived directly from feature_extractor_v9.py and preprocessing v7 schemas
# ======================================================================

def safe_float(x):
    try:
        if x is None: return 0.0
        if str(x).strip().lower() in ["unknown", "undefined", "null", "nan", ""]: return 0.0
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

def calculate_shannon_entropy(values):
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

def sanitize_weak_signals(f):
    weak_signal_features = ["sec_ch_ua_present", "sec_ch_ua_length", "accept_lang_present", 
                            "accept_lang_length", "accept_lang_comma_count", "accept_lang_q_count", 
                            "forwarded_for_count", "cpu_cores_missing", "device_memory_missing"]
    for k in weak_signal_features:
        if k in f:
            try:
                val = float(f[k])
                if val == 0: f[k] = 0
                elif val == 1: f[k] = 1
                else: f[k] = 2
            except:
                f[k] = 0
    return f

# ======== DATA PREP ==========
def prepare_dataframe(payload: dict) -> pd.DataFrame:
    """
    Transforms raw JSON payloads into a precise 61-dimensional numerical vector space.
    Surgically aligned with feature_extractor_v9.py and preprocessing validation rules.
    """
    f = {}
    
    mouse = payload.get("mouseMovements", [])
    keys = payload.get("keystrokes", [])
    scrolls = payload.get("scrollEvents", [])
    device = payload.get("device", {}) if isinstance(payload.get("device"), dict) else payload
    contextual = payload.get("contextual_features", {}) if isinstance(payload.get("contextual_features"), dict) else payload
    network = payload.get("network_layer", {}) if isinstance(payload.get("network_layer"), dict) else payload

    # If the proxy flattened the payload structure, fetch the header variables directly from the root level
    ua = str(network.get("userAgent", payload.get("userAgent", "")))
    ua_low = ua.lower()

    # FIX 4: Real User-Agent parsing and clean header presence maps
    f["ua_length"] = len(ua)
    f["mobile_vs_desktop"] = int(any(x in ua_low for x in ["mobile", "iphone", "android", "ipad"]))
    f["ua_is_chrome"] = int("chrome" in ua_low and "edg" not in ua_low and "opr" not in ua_low)
    f["ua_is_firefox"] = int("firefox" in ua_low)
    f["ua_is_edge"] = int("edg" in ua_low)
    f["ua_is_safari"] = int("safari" in ua_low and "chrome" not in ua_low)
    f["ua_is_opera"] = int("opr" in ua_low or "opera" in ua_low)

    # Aligned Language presence evaluation layers to eliminate webdriver tracking noise
    accept_lang_str = str(network.get("acceptLanguage", "")).strip()
    f["accept_lang_present"] = int(accept_lang_str != "" and accept_lang_str.lower() != "none")
    f["accept_lang_length"] = len(accept_lang_str) if f["accept_lang_present"] else 0
    f["accept_lang_comma_count"] = accept_lang_str.count(",") if f["accept_lang_present"] else 0
    f["accept_lang_q_count"] = accept_lang_str.lower().count("q=") if f["accept_lang_present"] else 0
    
    sec_ch = str(network.get("secChUa", ""))
    f["sec_ch_ua_present"] = int(sec_ch != "")
    f["sec_ch_ua_length"] = len(sec_ch)

    # FIX 5: Dynamic Cross-Layer JA4 Cryptographic Consistency Rules
    ja4_raw = str(contextual.get("ja4_raw", "unknown"))
    ja4_digest = str(contextual.get("ja4_digest", "unknown"))
    ja4_low = ja4_raw.lower()
    
    f["ja4_unknown"] = int(ja4_low == "unknown")
    f["ja4_bot_tls"] = int("t13d3112h1" in ja4_low)
    f["ja4_modern_tls"] = int("h2" in ja4_low)
    f["ja4_length"] = len(ja4_raw)
    
    ja4_structural_prefix = ja4_raw.split("_")[0].strip() if "_" in ja4_raw else ja4_raw
    f["ja4_prefix_encoded"] = float(settings.JA4_PREFIX_MAPPING.get(ja4_structural_prefix, 33.0))
    f["ja4_entropy_proxy"] = len(set(ja4_raw)) / max(len(ja4_raw), 1)
    f["ja4_is_modern"] = int("h2" in ja4_raw)
    f["ja4_is_legacy"] = int("h1" in ja4_raw)
    f["ja4_entropy"] = string_entropy(ja4_raw)

    f["ja4_digest_present"] = int(ja4_digest not in ["not_captured", "not_observed", "unknown", ""])
    f["ja4_consistency"] = int(ja4_raw != "unknown" and ja4_raw == ja4_digest)
    f["ja4_match"] = int(ja4_raw != "unknown" and ja4_raw == ja4_digest)
    
    browser_claim = any(x in ua_low for x in ["chrome", "firefox", "edge", "safari"])
    f["ua_ja4_mismatch"] = int(f["ja4_unknown"] == 0 and browser_claim and f["ja4_match"] == 0)

    # Device allocation structures maps
    f["cpu_cores"] = safe_float(device.get("cores", 4.0))
    f["device_memory"] = safe_float(device.get("memory", 8.0))
    f["cpu_cores_missing"] = int(device.get("cores") is None)
    f["device_memory_missing"] = int(device.get("memory") is None)
    
    plat_str = str(device.get("platform", "unknown")).strip()
    f["platform_encoded"] = float(settings.PLATFORM_MAPPING.get(plat_str, settings.PLATFORM_MAPPING.get("unknown", 0.0)))
    
    f["platform_is_windows"] = int("win" in plat_str.lower())
    f["platform_is_mac"] = int("mac" in plat_str.lower())
    f["platform_is_linux"] = int("linux" in plat_str.lower() and "android" not in plat_str.lower())
    f["platform_is_mobile"] = int(any(x in plat_str.lower() for x in ["android", "iphone", "ipad", "ios", "mobile"]))
    f["device_platform_mismatch"] = int((f["mobile_vs_desktop"] == 1 and f["platform_is_mobile"] == 0) or (f["mobile_vs_desktop"] == 0 and f["platform_is_mobile"] == 1))

    browser_str = "Chrome" if f["ua_is_chrome"] else "Edge" if f["ua_is_edge"] else "Firefox" if f["ua_is_firefox"] else "Safari" if f["ua_is_safari"] else "unknown"
    
    # FIX: Ensures absolute key parity by fallback checking against config types
    f["browser_encoded"] = float(settings.BROWSER_MAPPING.get(browser_str, settings.BROWSER_MAPPING.get("unknown", 0.0)))

    # Mouse Biometrics Processing
    velocities, accelerations, pauses, segments, angles = [], [], [], [], []
    timestamps = []

    for arr in [mouse, keys, scrolls]:
        for item in arr:
            if isinstance(item, dict) and "t" in item:
                try: timestamps.append(float(item["t"]))
                except: pass

    # FIX: Robust defensive conditional boundary pass prevents empty array ValueError crashes
    if len(timestamps) > 1:
        session_duration = max(timestamps) - min(timestamps)
    else:
        session_duration = 0.0
        
    duration_seconds = max(1.0, session_duration / 1000.0)

    if isinstance(mouse, list) and len(mouse) >= 2:
        for i in range(1, len(mouse)):
            p1, p2 = mouse[i - 1], mouse[i]
            dt = safe_float(p2.get("t")) - safe_float(p1.get("t"))
            d = euclid(p1, p2)
            if dt > 0:
                velocities.append(d / dt)
                if dt > 300: pauses.append(dt)
            segments.append(d)

        for i in range(1, len(velocities)): accelerations.append(abs(velocities[i] - velocities[i - 1]))
        for i in range(2, len(mouse)):
            a1 = math.atan2(safe_float(mouse[i-1].get("y")) - safe_float(mouse[i-2].get("y")), safe_float(mouse[i-1].get("x")) - safe_float(mouse[i-2].get("x")))
            a2 = math.atan2(safe_float(mouse[i].get("y")) - safe_float(mouse[i-1].get("y")), safe_float(mouse[i].get("x")) - safe_float(mouse[i-1].get("x")))
            angles.append(abs((a2 - a1 + math.pi) % (2 * math.pi) - math.pi))

        path = sum(segments)
        straight = euclid(mouse[0], mouse[-1])
        f["path_length"] = path
        f["straight_line_distance"] = straight
        f["straightness_ratio"] = straight / path if path > 0 else 1.0
    else:
        f["path_length"], f["straight_line_distance"], f["straightness_ratio"] = 0.0, 0.0, 1.0

    f["mean_velocity"] = safe_mean(velocities)
    f["std_velocity"] = safe_std(velocities)
    f["max_velocity"] = safe_max(velocities)
    f["min_velocity"] = safe_min(velocities)
    f["median_velocity"] = safe_median(velocities)
    f["mean_acceleration"] = safe_mean(accelerations)
    f["std_acceleration"] = safe_std(accelerations)
    f["mouse_entropy"] = calculate_shannon_entropy(velocities)
    f["acceleration_entropy"] = calculate_shannon_entropy(accelerations)
    f["direction_changes"] = sum(1 for x in angles if x > 1.0)
    
    # FIX 2: True, variable mouse density metric tracking
    f["mouse_density"] = len(mouse) / duration_seconds

    # Keyboard Telemetry Processing
    downs = [k for k in keys if str(k.get("type", "")).lower() == "down"]
    dwell = []
    flight = []
    active = {}
    overlap = 0

    if isinstance(keys, list):
        # Sort total keystroke events by raw timestamp coordinates to prevent cadence inversion
        sorted_keys = sorted(keys, key=lambda x: safe_float(x.get("t")))
        
        for e in sorted_keys:
            typ = str(e.get("type", "")).lower()
            key_char = str(e.get("key", "")).lower() # Synchronised variable reference
            t = safe_float(e.get("t"))
            
            if typ == "down":
                if len(active) > 0 and key_char not in active: 
                    overlap += 1
                active[key_char] = t
            elif typ == "up" and key_char in active:
                dwell.append(t - active[key_char])
                del active[key_char]

        # Chronological flight extraction loop computes real biological gaps
        for i in range(1, len(downs)):
            gap = safe_float(downs[i].get("t")) - safe_float(downs[i - 1].get("t"))
            if gap >= 0:
                flight.append(gap)

    f["avg_dwell_time"] = safe_mean(dwell)
    f["std_dwell_time"] = safe_std(dwell)
    f["avg_flight_time"] = safe_mean(flight)
    f["std_flight_time"] = safe_std(flight)
    f["key_overlap_count"] = overlap
    f["backspace_count"] = sum(1 for k in keys if str(k.get("key", "")).lower() in ["backspace", "delete"])
    f["has_backspace"] = int(f["backspace_count"] > 0)
    f["multi_key_overlap"] = int(overlap > 0)
    
    # Clean division handling using the downs array length
    f["backspace_ratio"] = f["backspace_count"] / max(len(downs), 1)
    f["unique_dwell_ratio"] = len(set(round(x, 2) for x in dwell)) / len(dwell) if len(dwell) else 0.0
    
    # Refactored key character extraction block maintains tight loop readability
    keys_pressed = [str(k.get("key", "")).lower() for k in keys if str(k.get("type", "")).lower() == "down"]
    f["unique_key_ratio"] = len(set(keys_pressed)) / max(len(downs), 1)
    f["key_density"] = len(keys) / duration_seconds
    f["scroll_count"] = len(scrolls)

    # Core data conditioning pass
    f = sanitize_weak_signals(f)
    
    frame = pd.DataFrame([f])
    frame = frame.replace([np.inf, -np.inf], np.nan)
    
    # Save the attributes cache cleanly before the shape reindex drop pass
    frame.attrs["raw_mouse_entropy"] = f.get("mouse_entropy", 0.0)
    frame.attrs["raw_avg_dwell_time"] = f.get("avg_dwell_time", 0.0)
    frame.attrs["raw_avg_flight_time"] = f.get("avg_flight_time", 0.0)
    frame.attrs["raw_direction_changes"] = f.get("direction_changes", 0.0)
    frame.attrs["raw_mean_velocity"] = f.get("mean_velocity", 0.0)
    
    frame = frame.reindex(columns=NUMERIC_COLS, fill_value=0.0)
    frame = frame.fillna(0.0)
    return frame.clip(-1e6, 1e6)

# ======================================================================
# STACKED INFERENCE
# ======================================================================

def svm_probability(frame):

    if hasattr(SVM_MODEL,"predict_proba"):
        return SVM_MODEL.predict_proba(frame)[:,1][0]

    decision=SVM_MODEL.decision_function(frame)[0]
    decision=np.clip(decision,-10,10)

    return float(1/(1+np.exp(-decision)))

def run_stacked_inference(frame):

    xgb_prob=XGB_MODEL.predict_proba(frame)[:,1][0]
    rf_prob=RF_MODEL.predict_proba(frame)[:,1][0]
    svm_prob=svm_probability(frame)

    meta_frame=pd.DataFrame([{
        "xgb_prob":xgb_prob,
        "rf_prob":rf_prob,
        "svm_prob":svm_prob
    }])

    for feature in SAFE_FEATURES:

        if feature in frame.columns:
            meta_frame[feature]=float(frame.iloc[0][feature])
        else:
            meta_frame[feature]=0.0

    meta_frame=meta_frame.reindex(
        columns=META_COLUMNS,
        fill_value=0.0
    )

    probability=META_MODEL.predict_proba(meta_frame)[:,1][0]

    return {
        "probability":float(probability),
        "xgb_prob":float(xgb_prob),
        "rf_prob":float(rf_prob),
        "svm_prob":float(svm_prob)
    }

# ======================================================================
# CLEANUP TASK
# ======================================================================

async def cleanup_rate_buckets():
    try:
        while True:
            now=time.time()
            stale_ips=[]

            async with rate_limit_lock:
                for ip,bucket in list(rate_buckets.items()):
                    while (
                        bucket
                        and now-bucket[0]
                        >settings.RATE_WINDOW_SECONDS
                    ):
                        bucket.popleft()

                    if not bucket:
                        stale_ips.append(ip)

                for ip in stale_ips:
                    rate_buckets.pop(ip,None)

            await asyncio.sleep(
                settings.RATE_BUCKET_CLEANUP_INTERVAL
            )

    except asyncio.CancelledError:

        logger.info(json.dumps({
            "event":"cleanup_task_cancelled",
            "timestamp_utc":datetime.utcnow().isoformat()
        }))

# ======================================================================
# FASTAPI LIFECYCLE
# ======================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cleanup_task
    logger.info(json.dumps({"event": "startup", "system_version": settings.SYSTEM_VERSION}))
    cleanup_old_metrics()

    # FIX: Warmup object matches real-time structure arrays to stop arithmetic log warnings
    dummy_payload = {
        "mouseMovements": [{"x": 0.5, "y": 0.5, "t": 100.0, "v": 0.0}, {"x": 0.6, "y": 0.6, "t": 200.0, "v": 0.01}],
        "keystrokes": [{"key": "a", "type": "down", "t": 150.0}, {"key": "a", "type": "up", "t": 250.0}],
        "scrollEvents": [],
        "device": {"cores": 4, "memory": 8, "platform": "Win32"},
        "contextual_features": {"ja4_raw": "t13d1516h2_8daaf6152771", "webdriver": False},
        "network_layer": {"userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0", "acceptLanguage": "en-US,en"}
    }

    dummy_frame = prepare_dataframe(dummy_payload)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, run_stacked_inference, dummy_frame)
    cleanup_task = asyncio.create_task(cleanup_rate_buckets())
    yield

    logger.info(json.dumps({
        "event":"shutdown",
        "timestamp_utc":datetime.utcnow().isoformat()
    }))

    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    executor.shutdown(wait=True)

    await metrics.export_json()

# ======================================================================
# FASTAPI APP
# ======================================================================

app=FastAPI(
    title="SENTINEL-BOT Runtime",
    version=settings.SYSTEM_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    BodySizeLimitMiddleware
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Metrics-Token"
    ]
)

# ======================================================================
# HEALTH
# ======================================================================

@app.get("/health")
async def health():
    async with rate_limit_lock:
        current_buckets = len(rate_buckets)

    # Extract exact state directly from the atomic concurrency semaphore primitive
    active_inference = settings.MAX_CONCURRENT_INFERENCE - inference_semaphore._value
    is_saturated = active_inference >= settings.MAX_CONCURRENT_INFERENCE

    # HARDENING AMENDMENT: Gate telemetry information leakage in production modes
    if getattr(settings, "EXPOSE_HEALTH_TELEMETRY", False) is False:
        return {"status": "degraded" if is_saturated else "healthy"}

    return {
        "status": "degraded" if is_saturated else "healthy",
        "system_version": settings.SYSTEM_VERSION,
        "model_version": settings.MODEL_VERSION,
        "active_rate_buckets": current_buckets,
        "active_inference": max(0, active_inference)
    }

# ======================================================================
# PROMETHEUS METRICS
# ======================================================================

@app.get("/metrics")
async def prometheus_metrics(
    request: Request,
    x_metrics_token: str | None = Header(default=None),
    authorization: str | None = Header(default=None)
):
    client_host = request.client.host if request.client else "unknown"

    if client_host in ["127.0.0.1", "::1", "localhost"]:
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    token = ""
    if x_metrics_token:
        token = x_metrics_token
    elif authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()

    if not token or not secrets.compare_digest(token, settings.METRICS_TOKEN):
        raise HTTPException(status_code=403, detail="Unauthorized metrics access.")

    return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# ======================================================================
# JSON METRICS
# ======================================================================
@app.get("/metrics-json")
async def metrics_json(x_metrics_token: str | None = Header(default=None)):

    token = x_metrics_token if x_metrics_token else ""

    if not secrets.compare_digest(
        token,
        settings.METRICS_TOKEN
    ):
        raise HTTPException(
            status_code=403,
            detail="Unauthorized metrics access."
        )

    return await metrics.snapshot()

# ======================================================================
# MAIN VERIFY ENDPOINT
# ======================================================================

@app.post(
    "/api/v1/verify",
    dependencies=[Depends(authenticate_waf_request)]
)
async def verify(payload: InferencePayload, request: Request):

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()

    PROM_REQUESTS_TOTAL.inc()

    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"
    else:
        client_ip = request.client.host if request.client else "unknown"

    # ATOMIC RATE SUPPRESSION & CEILING MITIGATION LAYER
    async with rate_limit_lock:
        if client_ip not in rate_buckets and len(rate_buckets) >= MAX_RATE_BUCKETS:
            PROM_OVERLOAD_TOTAL.inc()
            raise HTTPException(
                status_code=503,
                detail="Rate limiter saturated.",
                headers={"X-Request-ID": request_id}
            )

        bucket = rate_buckets[client_ip]
        now = time.time()

        while (
            bucket
            and now - bucket[0]
            > settings.RATE_WINDOW_SECONDS
        ):
            bucket.popleft()

        if len(bucket) >= settings.RATE_LIMIT_MAX_REQUESTS:
            PROM_RATE_LIMITED_TOTAL.inc()
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded.",
                headers={"X-Request-ID": request_id}
            )

        bucket.append(now)

    acquired = False
    gauge_incremented = False

    try:
        try:
            await asyncio.wait_for(
                inference_semaphore.acquire(),
                timeout=settings.SEMAPHORE_TIMEOUT_SECONDS
            )
            acquired = True
        except asyncio.TimeoutError:
            PROM_OVERLOAD_TOTAL.inc()
            raise HTTPException(
                status_code=503,
                detail="System overloaded.",
                headers={"X-Request-ID": request_id}
            )

        # Telemetry scales up immediately upon verified entry
        PROM_ACTIVE_INFERENCE.inc()
        gauge_incremented = True

        # Convert payload schema safely to dictionary
        raw_payload = payload.model_dump()

        # SAFE CONSOLIDATED SIMULATION OVERRIDE GATEWAY
        if raw_payload.get("straightness_ratio") is not None:
            # Reconstruct features cleanly using your safe_float helper to avoid null TypeErrors
            sim_features = {col: safe_float(raw_payload.get(col, 0.0)) for col in NUMERIC_COLS}
            frame = pd.DataFrame([sim_features])
            
            # Populate attributes cache fields to drive Prometheus time-series indicators
            frame.attrs["raw_mouse_entropy"] = safe_float(raw_payload.get("mouse_entropy", 4.5))
            frame.attrs["raw_avg_dwell_time"] = safe_float(raw_payload.get("avg_dwell_time", 120.0))
            frame.attrs["raw_avg_flight_time"] = safe_float(raw_payload.get("avg_flight_time", 90.0))
            frame.attrs["raw_direction_changes"] = safe_float(raw_payload.get("direction_changes", 90.0))
            frame.attrs["raw_mean_velocity"] = safe_float(raw_payload.get("mean_velocity", 240.0))
        else:
            # Fall back cleanly onto standard client-side browser tracking extraction pipelines
            frame = prepare_dataframe(raw_payload)

        # Execute stacked engine inference runs
        loop = asyncio.get_running_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, run_stacked_inference, frame),
            timeout=settings.INFERENCE_TIMEOUT_SECONDS
        )

        # FIXED: Pulled directly from your native aligned scikit-learn model column arrays
        probability = float(result["probability"])
        xgb_prob = float(result["xgb_prob"])
        rf_prob = float(result["rf_prob"])
        svm_prob = float(result["svm_prob"])

        # Enforce threshold security rules using your model's true target mapping boundaries
        is_bot = probability >= MODEL_THRESHOLD

        PROM_XGB_PROB.set(xgb_prob)
        PROM_RF_PROB.set(rf_prob)
        PROM_SVM_PROB.set(svm_prob)
        PROM_ENSEMBLE_PROB.set(probability)
        PROM_ENSEMBLE_PROB_DIST.observe(probability)

        if is_bot:
            PROM_JA4_BOT.inc()
            PROM_BLOCKED_TOTAL.inc()
        else:
            PROM_JA4_HUMAN.inc()
            PROM_ALLOWED_TOTAL.inc()

        # Extract JA4 structural signatures directly from your open contextual dict maps
        contextual_data = raw_payload.get("contextual_features", {})
        ja4_raw_string = str(contextual_data.get("ja4_raw", contextual_data.get("ja4_prefix", "unknown"))).strip().lower()
        extracted_prefix = ja4_raw_string.split("_")[0] if "_" in ja4_raw_string else ja4_raw_string
        
        # Increments Prometheus time-series registers safely without unbound errors
        PROM_JA4_OBSERVED.labels(ja4_prefix=extracted_prefix).inc()

        PROM_BIOMETRIC_MOUSE_ENTROPY.set(float(frame.attrs.get("raw_mouse_entropy", 0.0)))
        PROM_BIOMETRIC_DWELL_TIME.set(float(frame.attrs.get("raw_avg_dwell_time", 0.0)))
        PROM_BIOMETRIC_FLIGHT_TIME.set(float(frame.attrs.get("raw_avg_flight_time", 0.0)))
        PROM_BIOMETRIC_DIRECTION_CHANGES.set(float(frame.attrs.get("raw_direction_changes", 0.0)))

        latency_seconds = time.perf_counter() - start_time
        latency_ms = latency_seconds * 1000
        PROM_LATENCY.observe(latency_seconds)

        await metrics.update(allowed=not is_bot, latency_ms=latency_ms)

        # FIX 8: Enforce encapsulation rules to protect model parameters in production mode
        response = {
            "request_id": request_id,
            "bot_probability": round(probability, 4),
            # REMOVED: threshold key is stripped from the public response layout
            "decision": "BLOCK" if is_bot else "ALLOW",
            "latency_ms": round(latency_ms, 3),
            "system_version": settings.SYSTEM_VERSION,
            "jury_telemetry": {
                "extracted_mouse_entropy": round(frame.attrs.get("raw_mouse_entropy", 0.0), 3),
                "extracted_mean_velocity": round(frame.attrs.get("raw_mean_velocity", 0.0), 4),
                "extracted_avg_dwell_time_ms": round(frame.attrs.get("raw_avg_dwell_time", 0.0), 1),
                "extracted_avg_flight_time_ms": round(frame.attrs.get("raw_avg_flight_time", 0.0), 1)
            }
        }

        # Dynamically append model breakdowns and thresholds only if enabled in config
        if EXPOSE_MODEL_BREAKDOWN:
            response["threshold"] = MODEL_THRESHOLD # FIX: Gated safely for academic view only
            response["model_breakdown"] = {
                "xgb_prob": round(xgb_prob, 4),
                "rf_prob": round(rf_prob, 4),
                "svm_prob": round(svm_prob, 4)
            }

        return response

    except HTTPException:
        raise
    except Exception as e:
        PROM_FAILED_TOTAL.inc()
        await metrics.increment_failures()
        logger.exception(json.dumps({
            "event": "runtime_exception",
            "request_id": request_id,
            "error": repr(e),
            "timestamp_utc": datetime.utcnow().isoformat()
        }))
        raise HTTPException(
            status_code=500,
            detail="Internal inference failure.",
            headers={"X-Request-ID": request_id}
        )
    finally:
        if acquired:
            inference_semaphore.release()
        if gauge_incremented:
            PROM_ACTIVE_INFERENCE.dec()

# ======================================================================
# UVICORN ENTRYPOINT
# ======================================================================

if __name__ == "__main__":
    import uvicorn
    import os
    # Read dynamic port injected by the host OS environment context
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "predict_live:app", 
        host=settings.HOST, 
        port=port, 
        reload=False
    )