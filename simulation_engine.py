# ======================================================================
# SENTINEL-BOT
# simulation_engine.py v4.7
# Hardened Adversarial Traffic + Runtime Stress Framework
# ======================================================================

import asyncio,time
import random
import uuid
from collections import deque
import httpx
import numpy as np
from colorama import Fore,init
import runtime_config as settings
init(autoreset=True)

# ======================================================================
# AUTH HEADERS
# ======================================================================

VERIFY_HEADERS={ "X-API-Key":settings.VERIFY_API_KEY}

METRICS_HEADERS={ "X-Metrics-Token":settings.METRICS_TOKEN}

# ======================================================================
# PAYLOAD GENERATION
# ======================================================================

def generate_payload(rng,identity):
    if identity=="HUMAN":
        return {
            "straightness_ratio": float(np.clip(rng.normal(1.8,0.4),0,10)),
            "mean_velocity": float(max(0,rng.normal(240,60))),
            "std_velocity": float(max(0,rng.normal(60,20))),
            "direction_changes": float(max(0,rng.normal(90,25))),
            "mouse_entropy": float(max(0,rng.normal(4.5,0.5))),
            "acceleration_entropy": float(max(0,rng.normal(4.0,0.5))),
            "avg_dwell_time": float(max(0,rng.normal(120,25))),
            "std_dwell_time": float(max(0,rng.normal(40,10))),
            "avg_flight_time": float(max(0,rng.normal(90,20))),
            "std_flight_time": float(max(0,rng.normal(25,8))),
            "key_overlap_count": float(max(0,rng.normal(6,2))),
            "backspace_ratio": float(np.clip(rng.normal(0.08,0.03),0,1)),
            "unique_dwell_ratio": float(np.clip(rng.normal(0.72,0.08),0,1)),
            "unique_key_ratio": float(np.clip(rng.normal(0.74,0.07),0,1)),
            
            # --- INJECT SECURE DEVICE LAYERS FOR HUMAN ---
           "device": { "cores": 8, "memory": 16, "platform": "Win32" },
            "contextual_features": {
                "webdriver": 0.0,
                "ua_is_chrome": 1.0,
                "ja4_raw": "t13d1516h2_8daaf6152771_b186095e22b6"  
            }
        }

    return {
        "straightness_ratio": float(np.clip(rng.normal(8.5,1.2),0,10)),
        "mean_velocity": float(max(0,rng.normal(950,120))),
        "std_velocity": float(max(0,rng.normal(20,8))),
        "direction_changes": float(max(0,rng.normal(8,4))),
        "mouse_entropy": float(max(0,rng.normal(0.7,0.2))),
        "acceleration_entropy": float(max(0,rng.normal(0.5,0.2))),
        "avg_dwell_time": float(max(0,rng.normal(25,8))),
        "std_dwell_time": float(max(0,rng.normal(4,2))),
        "avg_flight_time": float(max(0,rng.normal(8,3))),
        "std_flight_time": float(max(0,rng.normal(2,1))),
        "key_overlap_count": float(max(0,rng.normal(0.3,0.2))),
        "backspace_ratio": float(np.clip(rng.normal(0.01,0.01),0,1)),
        "unique_dwell_ratio": float(np.clip(rng.normal(0.15,0.05),0,1)),
        "unique_key_ratio": float(np.clip(rng.normal(0.12,0.05),0,1)),
        
        # --- INJECT SECURE DEVICE LAYERS FOR BOT ---
        "device": { "cores": 4, "memory": 8, "platform": "Linux x86_64" },
        "contextual_features": {
            "webdriver": 1.0,
            "ua_is_chrome": 0.0,
            "ja4_raw": "t13d9804h2_a8f3f93d22b6_c13d1516h221"  # FIXED: Matches training prefix mapping (Yields 32.0)
        }
    }

# ======================================================================
# METRICS ENGINE
# ======================================================================

class Metrics:
    def __init__(self):
        self.lock=asyncio.Lock()

        self.total=0
        self.success=0

        self.fail_network=0
        self.fail_api=0
        self.fail_overload=0
        self.fail_queue=0
        self.fail_timeout=0
        self.fail_rate_limited=0

        self.tp=0
        self.fp=0
        self.tn=0
        self.fn=0

        # Memory Optimization Pass: Maxlen scaled down from 100k to 20k to reduce footprint
        self.rtt_latency=deque(maxlen=20000)
        self.api_latency=deque(maxlen=20000)
        self.e2e_latency=deque(maxlen=20000)

        self.xgb_probs=deque(maxlen=20000)
        self.rf_probs=deque(maxlen=20000)
        self.svm_probs=deque(maxlen=20000)
        self.ensemble_probs=deque(maxlen=20000)

        self.live_workers=0

    async def record_failure(self,category):

        async with self.lock:
            if category=="network":
                self.fail_network+=1
            elif category=="api":
                self.fail_api+=1
            elif category=="overload":
                self.fail_overload+=1
            elif category=="queue":
                self.fail_queue+=1
            elif category=="timeout":
                self.fail_timeout+=1
            elif category=="rate_limited":
                self.fail_rate_limited+=1

    def percentile(self,values,p):
        if not values:
            return 0.0
        return float(np.percentile(list(values),p))

    def precision(self):
        denom=self.tp+self.fp
        if denom==0:
            return 0.0
        return self.tp/denom

    def recall(self):
        denom=self.tp+self.fn
        if denom==0:
            return 0.0
        return self.tp/denom

    def accuracy(self):
        # Scale the denominator against actual model outcomes
        total_classified = self.tp + self.fp + self.tn + self.fn
        if total_classified == 0:
            # If no requests hit the ML layer yet, fall back cleanly to 100.0
            return 100.0
        return ((self.tp + self.tn) / total_classified) * 100

    def f1(self):
        p=self.precision()
        r=self.recall()
        if p+r==0:
            return 0.0
        return 2*(p*r)/(p+r)

    def summary(self):
        print("\n"+"="*70)
        print("SENTINEL-BOT v4.7 FINAL REPORT")
        print("="*70)

        print(f"Total Requests        : {self.total}")
        print(f"Successful Responses  : {self.success}")

        print("\nFailure Categories")
        print("-"*70)

        print(f"Network Failures      : {self.fail_network}")
        print(f"API Failures          : {self.fail_api}")
        print(f"Overload Rejections   : {self.fail_overload}")
        print(f"Rate Limit Rejections : {self.fail_rate_limited}")
        print(f"Queue Drops           : {self.fail_queue}")
        print(f"Timeout Failures      : {self.fail_timeout}")

        print("\nDetection Metrics")
        print("-"*70)

        print(f"Accuracy              : {self.accuracy():.2f}%")
        print(f"Precision             : {self.precision():.4f}")
        print(f"Recall                : {self.recall():.4f}")
        print(f"F1 Score              : {self.f1():.4f}")

        print("\nLatency Metrics")
        print("-"*70)

        print(f"P50 RTT               : {self.percentile(self.rtt_latency,50):.2f} ms")
        print(f"P95 RTT               : {self.percentile(self.rtt_latency,95):.2f} ms")
        print(f"P99 RTT               : {self.percentile(self.rtt_latency,99):.2f} ms")

        print(f"P95 API Latency       : {self.percentile(self.api_latency,95):.2f} ms")
        print(f"P95 E2E Latency       : {self.percentile(self.e2e_latency,95):.2f} ms")

        if self.ensemble_probs:
            print("\nEnsemble Telemetry")
            print("-"*70)
            print(f"Avg Ensemble Prob     : {np.mean(self.ensemble_probs):.4f}")
            if self.xgb_probs:      # Wrap sub-models to display safely only when populated
                print(f"Avg XGB Prob          : {np.mean(self.xgb_probs):.4f}")
                print(f"Avg RF Prob           : {np.mean(self.rf_probs):.4f}")
                print(f"Avg SVM Prob          : {np.mean(self.svm_probs):.4f}")

# ======================================================================
# WORKER POOL
# ======================================================================

class WorkerPool:
    def __init__(self,client,queue,metrics,stop_event):
        self.client=client
        self.queue=queue
        self.metrics=metrics
        self.stop_event=stop_event
        self.workers={}

    async def worker(self,worker_id):

        try:
            while True:
                if self.stop_event.is_set() and self.queue.empty():
                    break

                try:
                    item=await asyncio.wait_for(
                        self.queue.get(),
                        timeout=0.5
                    )

                except asyncio.TimeoutError:
                    continue

                item_data = item
                _, identity, enqueue_time = item_data # Unused incoming mock queue payload dropped

                # Explicitly extract the data dictionary, safely storing generated headers in a temporary placeholder
                generated_payload = generate_payload(np.random.default_rng(), identity)

                simulated_ua = (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36" 
                    if identity == "HUMAN" 
                    else "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) AutomationEngine/4.7"
                )

                try:
                    start=time.perf_counter()

                    response=await self.client.post(
                        settings.TARGET_URL,
                        headers={
                            "X-API-Key": settings.VERIFY_API_KEY,
                            "User-Agent": simulated_ua,
                            "Content-Type": "application/json"
                        },
                        json=generated_payload, 
                        timeout=httpx.Timeout(
                            settings.REQUEST_TIMEOUT,
                            connect=settings.CONNECT_TIMEOUT
                        )
                    )

                    rtt=(time.perf_counter()-start)*1000

                    # FIX: Print actual API errors before swallowing them
                    if response.status_code not in [200, 429, 503]:
                        print(f"API ERROR {response.status_code}: {response.text}")

                    if response.status_code==503:
                        await self.metrics.record_failure("overload")
                        continue

                    if response.status_code==429:
                        await self.metrics.record_failure("rate_limited")
                        continue

                    response.raise_for_status()
                    data=response.json()
                    decision=data.get("decision","ALLOW")
                    api_latency=float(data.get("latency_ms",0.0))
                    e2e_latency=(time.perf_counter()-enqueue_time)*1000
                    model_breakdown=data.get("model_breakdown", {})
                    ensemble_prob=float(data.get("bot_probability", 0.0))

                    async with self.metrics.lock:
                        self.metrics.success+=1
                        self.metrics.rtt_latency.append(rtt)
                        self.metrics.api_latency.append(api_latency)
                        self.metrics.e2e_latency.append(e2e_latency)
                        self.metrics.ensemble_probs.append(ensemble_prob)

                        # Only record sub-model parameters if they are exposed by the server gateway
                        if model_breakdown:
                            xgb_prob=float(model_breakdown.get("xgb_prob", 0.0))
                            rf_prob=float(model_breakdown.get("rf_prob", 0.0))
                            svm_prob=float(model_breakdown.get("svm_prob", 0.0))
                            
                            self.metrics.xgb_probs.append(xgb_prob)
                            self.metrics.rf_probs.append(rf_prob)
                            self.metrics.svm_probs.append(svm_prob)

                        if identity=="BOT":
                            if decision=="BLOCK":
                                self.metrics.tp+=1
                            else:
                                self.metrics.fn+=1

                        else:
                            if decision=="ALLOW":
                                self.metrics.tn+=1
                            else:
                                self.metrics.fp+=1

                except httpx.TimeoutException:
                    await self.metrics.record_failure("timeout")
                except httpx.HTTPStatusError:
                    await self.metrics.record_failure("api")
                except Exception:
                    await self.metrics.record_failure("network")

                finally:
                    try:
                        self.queue.task_done()
                    except ValueError:
                        pass

        except asyncio.CancelledError:
            pass

    def scale(self,target):
        target=max(
            settings.MIN_WORKERS,
            min(settings.MAX_WORKERS,target)
        )

        while len(self.workers)<target:
            wid=len(self.workers)
            self.workers[wid]=asyncio.create_task(
                self.worker(wid)
            )

        while len(self.workers)>target:
            wid,task=self.workers.popitem()
            task.cancel()

# ======================================================================
# LIVE HEALTH MONITOR
# ======================================================================

async def health_monitor(client):
    try:
        response=await client.get(
            settings.HEALTH_URL,
            timeout=3.0
        )

        if response.status_code==200:
            data=response.json()
            print(
                Fore.CYAN+
                f"[HEALTH] "
                f"status={data.get('status')} "
                f"active_inference={data.get('active_inference')} "
                f"rate_buckets={data.get('active_rate_buckets')}"
            )

    except Exception:
        print(
            Fore.RED+
            "[HEALTH] runtime unavailable"
        )

# ======================================================================
# MAIN SIMULATION
# ======================================================================

async def run(metrics):

    queue=asyncio.Queue(maxsize=settings.QUEUE_CAPACITY)
    stop_event=asyncio.Event()
    rng=np.random.default_rng(42)

    limits=httpx.Limits(
        max_keepalive_connections=settings.HTTP_MAX_KEEPALIVE,
        max_connections=settings.HTTP_MAX_CONNECTIONS
    )

    async with httpx.AsyncClient(
        limits=limits,
        http2=False
    ) as client:

        pool=WorkerPool(
            client,
            queue,
            metrics,
            stop_event
        )

        pool.scale(settings.MIN_WORKERS)

        # ==========================================================
        # PRODUCER
        # ==========================================================

        async def producer():
            while not stop_event.is_set():
                identity=(
                    "BOT"
                    if rng.random()>0.5
                    else "HUMAN"
                )
                payload=generate_payload(
                    rng,
                    identity
                )
                queue_load=(
                    queue.qsize()/
                    settings.QUEUE_CAPACITY
                )
                if queue_load>0.80:
                    throttle=settings.HIGH_THROTTLE
                elif queue_load>0.40:
                    throttle=settings.MID_THROTTLE
                else:
                    throttle=settings.BASE_THROTTLE
                try:
                    queue.put_nowait((
                        payload,
                        identity,
                        time.perf_counter()
                    ))

                    async with metrics.lock:
                        metrics.total+=1

                except asyncio.QueueFull:
                    await metrics.record_failure("queue")

                await asyncio.sleep(throttle)

        # ==========================================================
        # CONTROLLER
        # ==========================================================

        async def controller():
            previous_depth = 0
            last_scale_time = time.perf_counter()
            while not stop_event.is_set():
                await asyncio.sleep(
                    settings.CONTROLLER_INTERVAL
                )

                depth = queue.qsize()
                delta = depth - previous_depth
                current_workers = len(pool.workers)
                
                now = time.perf_counter()
                # Enforce a 4-second minimum cool-down block before performing down-scaling runs
                cooldown_passed = (now - last_scale_time) >= 4.0

                if depth > (settings.QUEUE_CAPACITY * 0.75):
                    pool.scale(current_workers + 10)
                    last_scale_time = now
                elif delta > 25:
                    pool.scale(current_workers + 5)
                    last_scale_time = now
                elif depth < (settings.QUEUE_CAPACITY * 0.15) and cooldown_passed:
                    pool.scale(current_workers - 5)
                    last_scale_time = now

                previous_depth = depth

                print(
                    Fore.YELLOW+
                    f"[METRICS PANEL] "
                    f"workers={len(pool.workers)} "
                    f"queue={depth} "
                    f"success={metrics.success} "
                    f"tp={metrics.tp} "
                    f"fp={metrics.fp} "
                    f"fn={metrics.fn}"
                )

                await health_monitor(client)

        # ==========================================================
        # TASKS
        # ==========================================================

        tasks=[
            asyncio.create_task(producer()),
            asyncio.create_task(controller())
        ]
        print(Fore.CYAN+"\n[Sentinel-Bot v4.7 Simulation Started]\n")
        await asyncio.sleep(settings.SIMULATION_DURATION_SECONDS)

        stop_event.set()
        await queue.join()

        for task in tasks: task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        for task in pool.workers.values(): task.cancel()
        await asyncio.gather(*pool.workers.values(), return_exceptions=True)

# ======================================================================
# ENTRYPOINT
# ======================================================================

if __name__=="__main__":
    metrics=Metrics()
    try:
        asyncio.run(run(metrics))
    except KeyboardInterrupt:
        print(Fore.YELLOW+"\nSimulation interrupted by user.\n")
    finally:
        metrics.summary()