# ======================================================================
# SENTINEL-BOT
# simulation_engine_opt.py v7.6
# Unified Production Hybrid Stress + Adversarial Simulation Framework
# ======================================================================

import asyncio
import time
import uuid
import json
import math
import logging
from dataclasses import dataclass, field
from collections import deque
from typing import Dict, Any

import httpx
import numpy as np
import runtime_config as settings

from colorama import Fore, init
init(autoreset=True)

# ----------------------------------------------------------------------
# SYSTEM OBSERVABILITY CONFIGURATION (STRUCTURED TELEMETRY)
# ----------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("sentinel-sim")

def log_structured(level: int, event: str, context: dict, sim_time: float):
    """Broadcasts publication-grade JSON events matching telemetry specs."""
    payload = {
        "event": event.upper(),
        "sim_timestamp": round(sim_time, 4),
        "wall_timestamp": time.time(),
        **context
    }
    log.log(level, json.dumps(payload))

# ----------------------------------------------------------------------
# VIRTUAL SIMULATION MONOTONIC CLOCK
# ----------------------------------------------------------------------
class SimClock:
    """Provides a thread-safe relative time reference across simulation epochs."""
    def __init__(self):
        self._t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._t0

# ----------------------------------------------------------------------
# MULTI-LAYER ROUTING & ADVERSARIAL FINGERPRINT POOLS
# ----------------------------------------------------------------------
PAGE_FLOW = ["LANDING", "LOGIN", "SEARCH", "CATEGORY", "FILTER", "PRODUCT_DETAIL", "CART_ADD", "CHECKOUT_PREVIEW", "PAYMENT_GATEWAY"]
KINEMATIC_PROFILES = ["CASUAL", "GAMER", "OFFICE", "MOBILE"]

HUMAN_ASN_POOL = [
    {"asn": 5384, "country": "AE", "weight": 0.25}, 
    {"asn": 15895, "country": "AE", "weight": 0.20},
    {"asn": 7922, "country": "US", "weight": 0.15}, 
    {"asn": 7018, "country": "US", "weight": 0.10},
    {"asn": 2856, "country": "GB", "weight": 0.10}, 
    {"asn": 3320, "country": "DE", "weight": 0.20}
]

BOT_ASN_POOL = [
    {"asn": 14061, "country": "US"}, 
    {"asn": 16509, "country": "US"},
    {"asn": 24940, "country": "DE"}, 
    {"asn": 16276, "country": "FR"}
]

ENVIRONMENT_POOLS = {
    "DESKTOP_CHROME_WIN": {
        "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36", 
        "platform": "Win32", 
        "cores": [8, 12], 
        "memory": [16, 32],
        "ja4_pool": ["t13d1516h2_8daaf6152771_b186095e22b6", "t13d1516h2_9fffb6152771_c186095e23a1"]
    },
    "DESKTOP_SAFARI_MAC": {
        "ua": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15", 
        "platform": "MacIntel", 
        "cores": [8, 10], 
        "memory": [8, 16],
        "ja4_pool": ["t13d1517h2_4ba2b6c13d15_a1b2c3d4e5f6"]
    },
    "MOBILE_SAFARI_IOS": {
        "ua": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Mobile/15E148 Safari/605.1.15", 
        "platform": "iPhone", 
        "cores": [6, 8], 
        "memory": [4, 6],
        "ja4_pool": ["t13d1412h2_9a8b7c6d5e4f_3a2b1c0d9e8f"]
    },
    "AUTOMATION_BOT_LINUX": {
        "ua": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) AutomationEngine/5.3",
        "platform": "Linux x86_64",
        "cores": [2, 4],
        "memory": [4, 8],
        "ja4_pool": ["t13d9804h2_a8f3f93d22b6_c13d1516h221", "unknown"]
    },
    # ENHANCEMENT: Sophisticated residential crawler spoofing profile
    "RESIDENTIAL_PROXY_BOT": {
        "ua": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "platform": "Win32",
        "cores": [8, 12],
        "memory": [16, 32],
        "ja4_pool": ["t13d1516h2_8daaf6152771_b186095e22b6"] # Clean Chrome JA4 match
    }
}

# ----------------------------------------------------------------------
# STATEFUL SESSION PERSISTENCE OBJECT (LAYER 2 - COHESIVE SESSION MODEL)
# ----------------------------------------------------------------------
@dataclass
class SessionState:
    """Tracks persistent session context to prevent stateless request isolation anomalies."""
    session_id: str
    identity_class: str        # HUMAN or BOT
    adversarial_posture: str   # BENIGN, NAIVE_AUTOMATION, ADVANCED_MIMICRY
    kinematic_profile: str     # CASUAL, GAMER, OFFICE, MOBILE
    clock: SimClock

    step: int = 0
    request_index: int = 0
    start_time: float = 0.0
    last_action: float = 0.0

    ua: str = ""
    ja4: str = ""
    platform: str = ""
    cores: int = 8
    memory: int = 16

    asn: int = 0
    country: str = ""
    proxy_score: float = 0.0
    datacenter_flag: float = 0.0
    cookie_age: float = 0.0
    dataset_split: str = ""

    def __post_init__(self):
        self.start_time = self.clock.now()
        self.last_action = self.start_time
        
        # Pure mathematical index partitioning pass
        hash_seed = int(self.session_id[:8], 16)
        self.dataset_split = "TRAIN_CALIBRATION" if (hash_seed % 10) < 7 else "TEST_EVALUATION"

    def page(self) -> str:
        return PAGE_FLOW[self.step]

    def advance(self, rng) -> bool:
        """Increments sequence states, closing paths dynamically upon checkout bounds."""
        self.request_index += 1
        if self.request_index >= 15:
            return False

        r = rng.random()
        # FIXED: Removed dead block nested statements to let standard bots advance safely
        if self.identity_class == "HUMAN" or self.adversarial_posture == "ADVANCED_MIMICRY":
            if r < 0.6 and self.step < len(PAGE_FLOW) - 1:
                self.step += 1
            elif r < 0.8 and self.step > 0:
                self.step -= 1
            elif r > 0.95:
                self.step = 0
            return True
        else:
            if self.step < len(PAGE_FLOW) - 1:
                self.step += 1
                return True
        return False

    def evolve_time(self):
        """Calculates exact cookie aging metrics, preventing static interval footprints."""
        now = self.clock.now()
        self.cookie_age += (now - self.last_action)
        self.last_action = now

# ----------------------------------------------------------------------
# BEHAVIORAL TRAJECTORY GENERATOR (LAYER 1 - CORE BIOMETRIC ENGINE)
# ----------------------------------------------------------------------
class BiometricPipeline:
    @staticmethod
    def generate_events(rng: np.random.Generator, session: SessionState) -> dict:
        """
        Generates raw, high-fidelity kinematic and temporal tracking arrays.
        Outputs realistic trajectories that let prepare_dataframe() extract 
        the exact features your machine learning model expects.
        """
        mouse_events = []
        keystroke_events = []
        
        # Pull profile mechanics
        profile = session.kinematic_profile
        is_bot = session.identity_class == "BOT"
        
        # Base timing markers
        base_time = 100.0

        # ==========================================================
        # 1. GENERATE MOUSE EVENTS
        # ==========================================================
        if is_bot:
            # Bots move in highly linear paths, very fast, with zero jitter
            points = 25
            for i in range(points):
                base_time += 15.0  # Perfectly uniform, rapid polling
                mouse_events.append({
                    "x": float(np.clip(0.1 + (i * 0.03), 0.0, 1.0)),
                    "y": float(np.clip(0.1 + (i * 0.02), 0.0, 1.0)),
                    "t": float(base_time)
                })
        else:
            # Humans display high spatial entropy, curvature, and variable speeds
            points = int(rng.integers(45, 90))
            for i in range(points):
                # Variable pacing mimicking muscle fatigue / human velocity curves
                base_time += float(rng.uniform(20.0, 45.0))
                
                # Add natural trembling/jitter vectors based on persona profile
                jitter = 0.005 if profile == "GAMER" else 0.015
                noise_x = rng.normal(0, jitter)
                noise_y = rng.normal(0, jitter)
                
                # Use a non-linear curvature mapping pass
                t_ratio = i / points
                curve_x = 0.2 + (t_ratio * 0.6) + noise_x
                curve_y = 0.15 + (math.sin(t_ratio * math.pi) * 0.2) + (t_ratio * 0.5) + noise_y
                
                mouse_events.append({
                    "x": float(np.clip(curve_x, 0.0, 1.0)),
                    "y": float(np.clip(curve_y, 0.0, 1.0)),
                    "t": float(base_time)
                })

        # ==========================================================
        # 2. GENERATE KEYBOARD EVENTS
        # ==========================================================
        keystroke_count = int(rng.integers(5, 12)) if not is_bot else 8
        
        for i in range(keystroke_count):
            if is_bot:
                # Bots have ultra-short, completely uniform typing intervals
                base_time += 80.0  # Fixed typing gap
                down_t = base_time
                up_t = down_t + 20.0  # Fixed, short hold duration
                key_char = "x"
            else:
                # Humans have long, irregular typing cadences and variable flight times
                base_time += float(rng.uniform(110.0, 310.0))
                down_t = base_time
                
                dwell_base = 60.0 if profile == "GAMER" else 110.0
                up_t = down_t + float(rng.normal(dwell_base, 15.0))
                
                key_char = "backspace" if (i == 3 and rng.random() < 0.1) else str(rng.choice(["s","e","n","t","i","l"]))

            keystroke_events.append({"key": key_char, "type": "down", "t": float(down_t)})
            keystroke_events.append({"key": key_char, "type": "up", "t": float(up_t)})

        return {
            "mouseMovements": mouse_events,
            "keystrokes": keystroke_events,
            "scrollEvents": []
        }
    
# ----------------------------------------------------------------------
# TELEMETRY MONITORING ENGINE (LAYER 5 - ASYMMETRIC EVALUATION MATRIX)
# ----------------------------------------------------------------------

class MetricsEngine:
    """Tracks performance metrics across decoupled lock instances to prevent contention."""
    def __init__(self):
        self.lock = asyncio.Lock()          # Protects system counts and latency arrays
        self.matrix_lock = asyncio.Lock()   # Protects confusion matrix cells
        self.window_lock = asyncio.Lock()   # Protects transient PID window counters
        self.inflight_lock = asyncio.Lock() # Protects worker concurrency targets (Fixed Issue 7)
        
        self.total = 0
        self.responses_received = 0         # Renamed to prevent semantic metric inflation
        self.saturated_events = 0

        self.fail_network = 0
        self.fail_api = 0
        self.fail_overload = 0
        self.fail_rate_limited = 0
        self.fail_timeout = 0
        self.fail_queue = 0

        self.partition_counts = {"TRAIN_CALIBRATION": 0, "TEST_EVALUATION": 0}

        self.matrix = {
            "HUMAN": {"ALLOW": 0, "BLOCK": 0},
            "BOT_STANDARD": {"ALLOW": 0, "BLOCK": 0},
            "BOT_BORDERLINE": {"ALLOW": 0, "BLOCK": 0}
        }

        self.rtt_latency_ms = deque(maxlen=20000)
        self.ensemble_probs = deque(maxlen=20000)
        self.xgb_probs = deque(maxlen=20000)
        self.rf_probs = deque(maxlen=20000)
        self.svm_probs = deque(maxlen=20000)
        
        self.processed_requests_in_window = 0

    @property
    def tp(self) -> int:
        return self.matrix["BOT_STANDARD"]["BLOCK"] + self.matrix["BOT_BORDERLINE"]["BLOCK"]

    @property
    def fp(self) -> int:
        return self.matrix["HUMAN"]["BLOCK"]

    @property
    def tn(self) -> int:
        return self.matrix["HUMAN"]["ALLOW"]

    @property
    def fn(self) -> int:
        return self.matrix["BOT_STANDARD"]["ALLOW"] + self.matrix["BOT_BORDERLINE"]["ALLOW"]

    async def record_dequeue_sync(self):
        async with self.window_lock:
            self.processed_requests_in_window += 1

    async def reset_dequeue_window_sync(self) -> int:
        async with self.window_lock:
            val = self.processed_requests_in_window
            self.processed_requests_in_window = 0
            return val

    async def record_metric_entry(self, rtt: float, prob: float, breakdown: dict):
        async with self.lock:
            self.responses_received += 1
            self.rtt_latency_ms.append(rtt)
            self.ensemble_probs.append(prob)
            if breakdown:
                self.xgb_probs.append(float(breakdown.get("xgb_prob", 0.0)))
                self.rf_probs.append(float(breakdown.get("rf_prob", 0.0)))
                self.svm_probs.append(float(breakdown.get("svm_prob", 0.0)))

    async def record_failure(self, category: str):
        async with self.lock:
            if category == "network": self.fail_network += 1
            elif category == "api": self.fail_api += 1
            elif category == "overload": self.fail_overload += 1
            elif category == "rate_limited": self.fail_rate_limited += 1
            elif category == "timeout": self.fail_timeout += 1
            elif category == "queue": self.fail_queue += 1

    async def increment_total_dispatched(self):
        async with self.lock:
            self.total += 1

    async def record_verdict(self, decision: str, true_label: str, dataset_split: str):
        """Secures classification arrays. Dataset split is computed dynamically per request."""
        async with self.matrix_lock:
            self.partition_counts[dataset_split] += 1
            if dataset_split == "TEST_EVALUATION":
                self.matrix[true_label][decision] += 1

    def global_accuracy(self) -> float:
        """Calculates global validation accuracy safely across all active classifications."""
        total = self.tp + self.fp + self.tn + self.fn
        if total == 0:
            return 0.0
        return ((self.tn + self.tp) / total) * 100.0

    def calculate_macro_metrics(self) -> dict:
        """Calculates decoupled One-Vs-Rest Precision, Recall, and F1 metrics for your thesis data."""
        results = {}
        
        # Explicit multi-class token mapping ensures proper ratio boundaries
        classes = {
            "HUMAN": {
                "tp": self.matrix["HUMAN"]["ALLOW"],
                "fp": self.matrix["BOT_STANDARD"]["ALLOW"] + self.matrix["BOT_BORDERLINE"]["ALLOW"],
                "fn": self.matrix["HUMAN"]["BLOCK"]
            },
            "BOT_STANDARD": {
                "tp": self.matrix["BOT_STANDARD"]["BLOCK"],
                "fp": self.matrix["HUMAN"]["BLOCK"] + self.matrix["BOT_BORDERLINE"]["BLOCK"],
                "fn": self.matrix["BOT_STANDARD"]["ALLOW"]
            },
            "BOT_BORDERLINE": {
                "tp": self.matrix["BOT_BORDERLINE"]["BLOCK"],
                "fp": self.matrix["HUMAN"]["BLOCK"] + self.matrix["BOT_STANDARD"]["BLOCK"],
                "fn": self.matrix["BOT_BORDERLINE"]["ALLOW"]
            }
        }

        for cls, vals in classes.items():
            tp = vals["tp"]
            fp = vals["fp"]
            fn = vals["fn"]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            results[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        return results

    def summary(self):
        macro = self.calculate_macro_metrics()
        rtt_list = list(self.rtt_latency_ms) if self.rtt_latency_ms else [0.0]

        print("\n" + "=" * 75)
        print("SENTINEL-BOT OPERATIONAL SECURITY VALIDATION REPORT (v8.0 HARDENED)")
        print("=" * 75)
        print(f"Data Partition Load Allocations : {self.partition_counts}")
        print(f"Total Transactions Dispatched   : {self.total}")
        print(f"Successful Server Roundtrips    : {self.responses_received} / {self.total}")
        print(f"Global Evaluation Accuracy      : {self.global_accuracy():.3f}%\n")
        
        print("ASYMMETRIC COHERENT CONFUSION MATRIX (TEST_EVALUATION SPLIT ONLY):")
        print("-" * 75)
        print(f"HUMAN PROFILE         -> Allowed [TN]: {self.matrix['HUMAN']['ALLOW']:<5} | Blocked [FP]: {self.matrix['HUMAN']['BLOCK']}")
        print(f"STANDARD BOT VECTOR   -> Caught  [TP]: {self.matrix['BOT_STANDARD']['BLOCK']:<5} | Leaked  [FN]: {self.matrix['BOT_STANDARD']['ALLOW']}")
        print(f"BORDERLINE ADVERSARY  -> Caught  [TP]: {self.matrix['BOT_BORDERLINE']['BLOCK']:<5} | Leaked  [FN]: {self.matrix['BOT_BORDERLINE']['ALLOW']}\n")
        
        print("DECOUPLED SCORING Breakdowns:")
        print("-" * 75)
        for cls, metrics in macro.items():
            print(f"{cls:<22} -> Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1-Score: {metrics['f1']:.4f}")
        print("=" * 75 + "\n")
        
        print(f"P50 Ingress Path Latency        : {float(np.percentile(rtt_list, 50)):.2f} ms")
        print(f"P95 Ingress Network Ceiling     : {float(np.percentile(rtt_list, 95)):.2f} ms")
        print(f"P99 Extreme Network tail Bounds : {float(np.percentile(rtt_list, 99)):.2f} ms")
        
        if self.ensemble_probs:
            print(f"Mean Meta-Learner Probability   : {float(np.mean(list(self.ensemble_probs))):.4f}")
            if self.xgb_probs:
                print(f"Mean Sub-Classifier Forecasts   : XGB: {float(np.mean(list(self.xgb_probs))):.3f} | RF: {float(np.mean(list(self.rf_probs))):.3f} | SVM: {float(np.mean(list(self.svm_probs))):.3f}")
        print("=" * 75 + "\n")

# ----------------------------------------------------------------------
# DECOUPLED SESSION LAYER FACTORY
# ----------------------------------------------------------------------
def create_session(rng: np.random.Generator, clock: SimClock, mode: str) -> SessionState:
    """Unified session builder factory handles variable multi-class threat allocations."""
    if mode == "PRODUCTION":
        is_human = rng.random() > 0.05  # 95/5 Real-world operational footprint
    else:
        is_human = rng.random() > 0.50  # 50/50 Balanced validation distribution
        
    sid = uuid.uuid4().hex
    kinematic_profile = str(rng.choice(KINEMATIC_PROFILES))
    
    if is_human:
        identity_class = "HUMAN"
        adversarial_posture = "BENIGN"
    else:
        identity_class = "BOT"
        adversarial_posture = str(rng.choice(["NAIVE_AUTOMATION", "ADVANCED_MIMICRY", "RESIDENTIAL_PROXY_BOT"]))

    session = SessionState(
        session_id=sid, identity_class=identity_class,
        adversarial_posture=adversarial_posture, kinematic_profile=kinematic_profile,
        clock=clock
    )

    target_pool_key = "AUTOMATION_BOT_LINUX" if adversarial_posture == "NAIVE_AUTOMATION" else (
        "RESIDENTIAL_PROXY_BOT" if adversarial_posture == "RESIDENTIAL_PROXY_BOT" else (
            "MOBILE_SAFARI_IOS" if kinematic_profile == "MOBILE" else rng.choice(["DESKTOP_CHROME_WIN", "DESKTOP_SAFARI_MAC"])
        )
    )
    
    env = ENVIRONMENT_POOLS[target_pool_key]
    session.ua = env["ua"]
    session.platform = env["platform"]
    session.cores = int(rng.choice(env["cores"]))
    session.memory = int(rng.choice(env["memory"]))

    # ADVANCEDRealism: Introduce slight client-side validation inconsistencies (Fixed Issue 5)
    if adversarial_posture == "ADVANCED_MIMICRY" and rng.random() < 0.10:
        session.ja4 = "unknown"  # Inject a fingerprint mismatch anomaly
    else:
        session.ja4 = str(rng.choice(env["ja4_pool"]))

    # Synchronize Network Profiles
    if identity_class == "HUMAN":
        weights = np.array([n["weight"] for n in HUMAN_ASN_POOL], dtype=np.float64)
        net = rng.choice(HUMAN_ASN_POOL, p=weights / np.sum(weights))
        session.asn, session.country, session.proxy_score, session.datacenter_flag = net["asn"], net["country"], float(rng.beta(1, 12) * 0.1), 0.0
        session.cookie_age = float(rng.uniform(3600, 86400 * 5))
    elif adversarial_posture == "RESIDENTIAL_PROXY_BOT":
        # Residential proxy crawlers mix clean consumer ASNs with short cookie lifetimes
        weights = np.array([n["weight"] for n in HUMAN_ASN_POOL], dtype=np.float64)
        net = rng.choice(HUMAN_ASN_POOL, p=weights / np.sum(weights))
        session.asn, session.country, session.proxy_score, session.datacenter_flag = net["asn"], net["country"], float(rng.uniform(0.15, 0.40)), 0.0
        session.cookie_age = float(rng.uniform(0.0, 45.0))
    else: # NAIVE & ADVANCED DATA-CENTER INFRASTRUCTURES
        net = rng.choice(BOT_ASN_POOL)
        session.asn, session.country, session.proxy_score, session.datacenter_flag = net["asn"], net["country"], float(rng.uniform(0.80, 1.00)), 1.0
        session.cookie_age = float(rng.uniform(0.0, 15.0))

    return session

# ----------------------------------------------------------------------
# SYSTEM WORKER POOL EXECUTION PLUGINS (LAYER 6 - OPERATIONAL CONTROLS)
# ----------------------------------------------------------------------
class WorkerPool:
    def __init__(self, client: httpx.AsyncClient, queue: asyncio.Queue, metrics: MetricsEngine, stop_event: asyncio.Event, clock: SimClock):
        self.client = client
        self.queue = queue
        self.metrics = metrics
        self.stop_event = stop_event
        self.clock = clock
        self.workers: Dict[str, asyncio.Task] = {}
        self.in_flight_requests = 0

    async def worker(self, wid: str):
        rng = np.random.default_rng(uuid.uuid4().int & (2**32 - 1))

        while True:
            session = await self.queue.get()
            if session is None:
                self.queue.task_done()
                break

            async with self.metrics.inflight_lock:
                self.in_flight_requests += 1

            try:
                if self.stop_event.is_set():
                    return  

                session.evolve_time()
                events_payload = BiometricPipeline.generate_events(rng, session)

                # FIXED: Force request_split to reference your deterministic session state matrix rules
                request_split = session.dataset_split

                payload = {
                    "sessionId": session.session_id,
                    "request_index": session.request_index,
                    "dataset_partition": request_split,

                    # --- CORE KINEMATIC COEFFICIENTS MAP ---
                    "straightness_ratio": float(np.clip(rng.normal(8.5, 1.2), 1.0, 10.0)) if session.identity_class == "BOT" else float(np.clip(rng.normal(1.8, 0.3), 1.0, 3.5)),
                    "mean_velocity": float(max(10.0, rng.normal(950.0, 120.0))) if session.identity_class == "BOT" else float(max(10.0, rng.normal(240.0, 50.0))),
                    "std_velocity": float(max(1.0, rng.normal(20.0, 8.0))) if session.identity_class == "BOT" else float(max(1.0, rng.normal(60.0, 18.0))),
                    "direction_changes": float(max(0.0, rng.normal(8.0, 4.0))) if session.identity_class == "BOT" else float(max(0.0, rng.normal(90.0, 20.0))),
                    "mouse_entropy": float(max(0.0, rng.normal(0.7, 0.2))) if session.identity_class == "BOT" else float(max(0.0, rng.normal(4.5, 0.4))),
                    "acceleration_entropy": float(max(0.0, rng.normal(0.5, 0.2))) if session.identity_class == "BOT" else float(max(0.0, rng.normal(4.0, 0.4))),
                    "avg_dwell_time": float(max(1.0, rng.normal(25.0, 8.0))) if session.identity_class == "BOT" else float(max(1.0, rng.normal(120.0, 22.0))),
                    "std_dwell_time": float(max(1.0, rng.normal(4.0, 2.0))) if session.identity_class == "BOT" else float(max(1.0, rng.normal(40.0, 8.0))),
                    "avg_flight_time": float(max(1.0, rng.normal(8.0, 3.0))) if session.identity_class == "BOT" else float(max(1.0, rng.normal(90.0, 18.0))),
                    "std_flight_time": float(max(1.0, rng.normal(2.0, 1.0))) if session.identity_class == "BOT" else float(max(1.0, rng.normal(25.0, 6.0))),
                    "key_overlap_count": float(max(0.0, rng.normal(0.3, 0.2))) if session.identity_class == "BOT" else float(max(0.0, rng.normal(6.0, 1.5))),
                    "backspace_ratio": float(np.clip(rng.normal(0.01, 0.01), 0.0, 1.0)) if session.identity_class == "BOT" else float(np.clip(rng.normal(0.08, 0.02), 0.0, 1.0)),
                    "unique_dwell_ratio": float(np.clip(rng.normal(0.15, 0.05), 0.0, 1.0)) if session.identity_class == "BOT" else float(np.clip(rng.normal(0.72, 0.07), 0.0, 1.0)),
                    "unique_key_ratio": float(np.clip(rng.normal(0.12, 0.05), 0.0, 1.0)) if session.identity_class == "BOT" else float(np.clip(rng.normal(0.74, 0.06), 0.0, 1.0)),

                    # --- FIXED: UNROLL CATEGORICAL VALUES TO ROOT LEVEL TO MATCH NUMERIC_COLS PARSING ---
                    "browser_encoded": 0.0 if session.identity_class == "BOT" else 1.0,
                    "platform_encoded": 3.0 if session.identity_class == "BOT" else 1.0,
                    "ja4_prefix_encoded": 32.0 if session.identity_class == "BOT" else 4.0,
                    "ua_length": len(session.ua),
                    "datacenter_flag": 1.0 if session.identity_class == "BOT" else 0.0,
                    "proxy_score": session.proxy_score,

                    **events_payload,
                    "device": {"cores": session.cores, "memory": session.memory, "platform": session.platform},
                    "contextual_features": {"webdriver": 1.0 if session.adversarial_posture == "NAIVE_AUTOMATION" else 0.0, "ja4_raw": session.ja4, "ja4_digest": session.ja4},
                    "network_layer": {"userAgent": session.ua, "acceptLanguage": "en-US,en;q=0.9" if session.identity_class == "HUMAN" else "none"},
                    "network_profile": {"asn": session.asn, "country": session.country, "proxy_score": session.proxy_score, "datacenter_flag": session.datacenter_flag},
                    "page_context": session.page(),
                    "cookie_age": session.cookie_age
                }

                await self.metrics.increment_total_dispatched()
                t0 = self.clock.now()

                response = await self.client.post(
                    settings.TARGET_URL,
                    json=payload,
                    headers={"X-API-Key": settings.VERIFY_API_KEY, "Content-Type": "application/json"},
                    timeout=httpx.Timeout(5.0)
                )

                rtt = (self.clock.now() - t0) * 1000

                if response.status_code == 503:
                    await self.metrics.record_failure("overload")
                    decision = "BLOCK"
                elif response.status_code == 429:
                    await self.metrics.record_failure("rate_limited")
                    decision = "BLOCK"
                elif response.status_code == 401:
                    # ENHANCEMENT: Catch token authentication exceptions cleanly
                    await self.metrics.record_failure("api")
                    decision = "BLOCK"
                else:
                    response.raise_for_status()
                    data = response.json()
                    decision = data.get("decision", "ALLOW")
                    prob = float(data.get("bot_probability", 0.0))
                    breakdown = data.get("model_breakdown", {})

                await self.metrics.record_metric_entry(rtt, prob, breakdown)
                
                # FIXED: Map both mimicry and proxy bots to BOT_BORDERLINE accurately (Fixed Issue 13)
                if session.identity_class == "HUMAN":
                    true_label = "HUMAN"
                else:
                    true_label = "BOT_BORDERLINE" if session.adversarial_posture in ["ADVANCED_MIMICRY", "RESIDENTIAL_PROXY_BOT"] else "BOT_STANDARD"
                
                await self.metrics.record_verdict(decision, true_label, request_split)
                await self.metrics.record_dequeue_sync()

                # FIXED: Balanced multi-class user churn thresholds prevent session depth bias (Fixed Issue 6)
                churn_probability = 0.10 if session.identity_class == "HUMAN" else (
                    0.05 if session.adversarial_posture == "RESIDENTIAL_PROXY_BOT" else 0.02
                )
                is_session_abandoned = rng.random() < churn_probability
                
                if not self.stop_event.is_set() and not is_session_abandoned and session.advance(rng):
                    try:
                        # FIXED: Changed from await put() to put_nowait() to prevent asynchronous thread locks
                        self.queue.put_nowait(session)
                    except asyncio.QueueFull:
                        # If the queue hits its capacity ceiling, gracefully log the drop and clear the task context safely
                        await self.metrics.record_failure("queue")

            except httpx.TimeoutException:
                await self.metrics.record_failure("timeout")
            except (httpx.HTTPStatusError, json.JSONDecodeError):
                await self.metrics.record_failure("api")
            except Exception:
                await self.metrics.record_failure("network")
            finally:
                async with self.metrics.inflight_lock:
                    self.in_flight_requests -= 1
                # Enforce reliable tracking counters cleanup
                self.queue.task_done()

    def scale(self, target: int):
        """Adjusts worker configuration limits dynamically. Permits explicit shutdown runs."""
        # FIXED: Allows target to equal absolute 0 during a teardown sweep pass
        if target != 0:
            target = max(settings.MIN_WORKERS, min(settings.MAX_WORKERS, target))
            
        current = len(self.workers)
        while current < target:
            wid = str(uuid.uuid4())[:6]
            self.workers[wid] = asyncio.create_task(self.worker(wid))
            current += 1
        while current > target and self.workers:
            wid, task = self.workers.popitem()
            try:
                asyncio.get_event_loop().create_task(self.queue.put(None))
            except Exception:
                task.cancel()
            current -= 1

# ----------------------------------------------------------------------
# FIXED: ADDED THE PI CONTROLLER MECHANICS CLASS LAYER (INFRASTRUCTURE LAYER)
# ----------------------------------------------------------------------
class PIDController:
    """Executes feedback worker scaling corrections under metric pressure."""
    def __init__(self, kp: float, ki: float, kd: float, target_depth: float, clock: SimClock):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_depth = target_depth
        self.clock = clock

        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = clock.now()

    def update(self, current: int) -> int:
        now = self.clock.now()
        dt = max(0.1, now - self.last_time)

        error = current - self.target_depth
        self.integral = max(-50.0, min(50.0, self.integral + (error * dt)))
        derivative = (error - self.last_error) / dt

        self.last_error = error
        self.last_time = now

        adjustment = int(self.kp * error + self.ki * self.integral + self.kd * derivative)
        if abs(adjustment) < 2:
            adjustment = 0
        return adjustment

# ======================================================================
# SYSTEM LIFECYCLE COORDINATOR (LAYER 6 & 7 - INFRASTRUCTURE RUNTIME)
# ======================================================================

async def health_monitor(client: httpx.AsyncClient):
    try:
        response = await client.get(settings.HEALTH_URL, timeout=2.0)
        if response.status_code == 200:
            data = response.json()
            print(Fore.CYAN + (
                f"[WAF RUNTIME Check] status={data.get('status')} | "
                f"active_semaphore={data.get('active_inference')} | "
                f"limiter_buckets={data.get('active_rate_buckets')}"
            ))
    except Exception:
        print(Fore.RED + "[WAF PROTECTION ALARM] Cloud inference endpoint unreachable.")

async def run_simulation(metrics_instance: MetricsEngine):
    queue = asyncio.Queue(maxsize=settings.QUEUE_CAPACITY)
    stop_event = asyncio.Event()
    prod_rng = np.random.default_rng(uuid.uuid4().int & (2**32 - 1))
    
    # Read active parameters straight from the unified runtime config variables (Fixed Issue 1)
    active_mode = getattr(settings, "SIMULATION_MODE", "PRODUCTION")

    limits = httpx.Limits(max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE, max_connections=settings.HTTP_MAX_CONNECTIONS)
    log_structured(logging.INFO, "simulation_start", {"mode": active_mode, "ceiling_sec": settings.SIMULATION_DURATION_SECONDS}, 0.0)

    async with httpx.AsyncClient(limits=limits, http2=False) as client:
        pool = WorkerPool(client, queue, metrics_instance, stop_event, SimClock())
        pool.scale(settings.MIN_WORKERS)
        pid = PIDController(kp=0.12, ki=0.01, kd=0.03, target_depth=settings.QUEUE_CAPACITY * 0.25, clock=pool.clock)

        # FIXED: State flags process transitions cleanly to stop overcounting faults (Fixed Issue 8)
        is_queue_saturated = False

        async def producer_loop():
            nonlocal is_queue_saturated
            while not stop_event.is_set():
                q_depth = queue.qsize()
                feedback_scalar = min(1.0 + (q_depth / settings.QUEUE_CAPACITY) ** 2, 2.5)
                batch_size = 5 if q_depth < (0.75 * settings.QUEUE_CAPACITY) else 15
                batch_size = max(1, int(batch_size / feedback_scalar))

                if q_depth >= settings.QUEUE_CAPACITY:
                    if not is_queue_saturated:
                        await metrics_instance.record_failure("queue")
                        log_structured(logging.WARNING, "queue_saturation", {"backlog": q_depth}, pool.clock.now())
                        is_queue_saturated = True
                else:
                    is_queue_saturated = False

                for _ in range(batch_size):
                    if not stop_event.is_set() and queue.qsize() < settings.QUEUE_CAPACITY:
                        # FIXED: Eliminated duplicated logic block. Spawns clean objects from factory natively
                        session = create_session(prod_rng, pool.clock, active_mode)
                        queue.put_nowait(session)

                await asyncio.sleep(settings.BASE_THROTTLE * feedback_scalar)

        async def monitor_loop():
            ema_backlog_depth = 0.0
            prev_backlog_depth = 0
            last_scale_time = pool.clock.now()
            
            while not stop_event.is_set():
                await asyncio.sleep(settings.CONTROLLER_INTERVAL)
                current_depth = queue.qsize()
                
                async with metrics_instance.lock:
                    rtt_list = list(metrics_instance.rtt_latency_ms) if metrics_instance.rtt_latency_ms else [0.0]
                    outage_pressure = metrics_instance.fail_overload + metrics_instance.fail_rate_limited + metrics_instance.fail_timeout
                
                async with metrics_instance.inflight_lock:
                    current_allocated_workers = len(pool.workers)
                    in_flight = pool.in_flight_requests

                p95_latency = float(np.percentile(rtt_list, 95))
                latency_penalty = max(0, int((p95_latency - 150.0) / 8.0)) if p95_latency > 150.0 else 0
                composite_system_load = current_depth + latency_penalty + min(60, outage_pressure)
                
                ema_backlog_depth = (0.35 * composite_system_load) + (0.65 * ema_backlog_depth)
                dequeued_items = await metrics_instance.reset_dequeue_window_sync()
                drain_velocity = dequeued_items / settings.CONTROLLER_INTERVAL
                
                velocity_scalar = 1.0 if drain_velocity > 10 else 0.5
                pid_adjustment = int(np.round(pid.update(int(ema_backlog_depth)) * velocity_scalar))

                now_clock = pool.clock.now()
                # FIXED: Enforce a 4-second minimum cool-down block before scaling down to prevent oscillations (Fixed Issue 12)
                if outage_pressure > 20:
                    optimized_worker_target = max(settings.MIN_WORKERS, current_allocated_workers - 5)
                    last_scale_time = now_clock
                elif pid_adjustment < 0 and (now_clock - last_scale_time) < 4.0:
                    optimized_worker_target = current_allocated_workers
                else:
                    optimized_worker_target = max(settings.MIN_WORKERS, min(settings.MAX_WORKERS, current_allocated_workers + pid_adjustment))
                    if optimized_worker_target != current_allocated_workers:
                        last_scale_time = now_clock

                pool.scale(optimized_worker_target)
                log_structured(logging.INFO, "worker_scale", {"allocated": len(pool.workers), "in_flight": in_flight, "p95_ms": round(p95_latency, 2)}, now_clock)

                print(Fore.YELLOW + f"[SOC REGISTER MONITOR] allocated_pool={len(pool.workers)} | queue_backlog={current_depth} | tp_caught={metrics_instance.tp} | fp_blocks={metrics_instance.fp} | fn_leaked={metrics_instance.fn}")
                
                # FIXED: Fired health checks inside an un-awaited background task to prevent execution lag (Fixed Issue 10)
                asyncio.create_task(health_monitor(client))
                prev_backlog_depth = current_depth

        execution_tasks = [asyncio.create_task(producer_loop()), asyncio.create_task(monitor_loop())]
        await asyncio.sleep(settings.SIMULATION_DURATION_SECONDS)

        print(Fore.RED + "\nLifespan coordinate threshold complete. Running final queue drain loop...")
        stop_event.set()
        execution_tasks[0].cancel()
        await asyncio.gather(execution_tasks[0], return_exceptions=True)

        for _ in range(max(1, len(pool.workers))):
            await queue.put(None)

        await queue.join()
        execution_tasks[1].cancel()
        await asyncio.gather(execution_tasks[1], return_exceptions=True)

        pool.scale(0)
        log_structured(logging.INFO, "simulation_complete", {"total_dispatched": metrics_instance.total}, pool.clock.now())
        metrics_instance.summary()

# ======================================================================
# SYSTEM ENTRYPOINT
# ======================================================================
if __name__ == "__main__":
    global_metrics = MetricsEngine()
    try:
        # Launch the asynchronous runtime core simulation environment
        asyncio.run(run_simulation(global_metrics))
    except KeyboardInterrupt:
        print(Fore.RED + "\n[SYSTEM ALARM] Simulation sequence interrupted dynamically by administrator thread.")
    finally:
        # Extract data straight from locked properties to match v4.7 terminal outputs cleanly
        macro_breakdowns = global_metrics.calculate_macro_metrics()
        rtt_vector = list(global_metrics.rtt_latency_ms) if global_metrics.rtt_latency_ms else [0.0]

        print("\n" + "=" * 75)
        print("SENTINEL-BOT v4.7 HARDENED RUNTIME TERMINAL SUMMARY")
        print("=" * 75)
        print(f"Total Transactions Dispatched   : {global_metrics.total}")
        print(f"Successful Server Roundtrips    : {global_metrics.responses_received}")
        print(f"Global Validation Accuracy      : {global_metrics.global_accuracy():.2f}%")
        
        print("\nFailure Categories Counter Logs")
        print("-" * 75)
        print(f"Network Outages / Socket Drops  : {global_metrics.fail_network}")
        print(f"Server 500 Fatal Anomalies      : {global_metrics.fail_api}")
        print(f"HTTP 503 Semaphore Load-Shed    : {global_metrics.fail_overload}")
        print(f"HTTP 429 Ingress Rate Clamps    : {global_metrics.fail_rate_limited}")
        print(f"Internal Buffer Queue Drops     : {global_metrics.fail_queue}")
        print(f"Client Fetch Timeout Exceeded   : {global_metrics.fail_timeout}")

        print("\nAsymmetric Coherent Matrix Profiles (TEST Partition Only)")
        print("-" * 75)
        print(f"True Positives  [Caught Bots]   : {global_metrics.tp}")
        print(f"False Positives [Human Blocks]  : {global_metrics.fp}")
        print(f"True Negatives  [Human Cleared] : {global_metrics.tn}")
        print(f"False Negatives [Leaked Attacks]: {global_metrics.fn}")

        print("\nDecoupled One-Vs-Rest Macro Scores Breakdown")
        print("-" * 75)
        for target_class, score_map in macro_breakdowns.items():
            print(f"Class: {target_class:<16} -> Precision: {score_map['precision']:.4f} | Recall: {score_map['recall']:.4f} | F1: {score_map['f1']:.4f}")

        print("-" * 75)
        print(f"P50 Average Ingress Latency     : {float(np.percentile(rtt_vector, 50)):.2f} ms")
        print(f"P95 Ingress Network Ceiling     : {float(np.percentile(rtt_vector, 95)):.2f} ms")
        if global_metrics.ensemble_probs:
            print(f"Mean Classifier Probability     : {float(np.mean(list(global_metrics.ensemble_probs))):.4f}")
        print("=" * 75 + "\n")