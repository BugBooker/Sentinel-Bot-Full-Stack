# ======================================================================
# SENTINEL-BOT
# playwright_sim.py - Production Evaluation Harness
# Form-Targeted High-Fidelity Client Automation Script
# ======================================================================

import asyncio
import random
import uuid
from playwright.async_api import async_playwright
from colorama import Fore, init

# Initialize colorama auto-reset handles cleanly
init(autoreset=True)

# ======================================================================
# CONTROL CONFIGURATION
# ======================================================================
# Toggles your adversarial signature behavior:
# - "NAIVE": Hyper-fast mechanical vectors. GUARANTEES EXCEEDING THE 80% WAF THRESHOLD.
# - "EVASIVE": Calibrated curves, variable delays. CHECKS ENSEMBLE LIMITS.
BOT_MODE = "NAIVE" 

CURSOR_TRACKER = {"x": 0.0, "y": 0.0}

async def human_mouse_glide(page, target_element, steps=150):
    """Glides the cursor continuously to generate dense telemetry arrays for the listener."""
    global CURSOR_TRACKER
    await target_element.wait_for(state="attached")
    box = await target_element.bounding_box()
    if not box:
        return
    
    target_x = box["x"] + box["width"] / 2
    target_y = box["y"] + box["height"] / 2
    
    start_x = CURSOR_TRACKER["x"]
    start_y = CURSOR_TRACKER["y"]

    if BOT_MODE == "NAIVE":
        print(f"[NAIVE] Flooding tracking array with linear vectors to: ({int(target_x)}, {int(target_y)})")
        # Generates a highly dense stream of uniform coordinate ticks
        for i in range(1, steps + 1):
            t = i / steps
            next_x = start_x + (target_x - start_x) * t
            next_y = start_y + (target_y - start_y) * t
            
            await page.mouse.move(next_x, next_y)
            await asyncio.sleep(0.002) # Fast mechanical interval to maximize data velocity arrays
            
        # Execute an intentional micro-oscillation directly over the field center
        # This injects zero-entropy repetitive coordinates that humans cannot replicate
        for _ in range(20):
            await page.mouse.move(target_x + 1, target_y)
            await asyncio.sleep(0.002)
            await page.mouse.move(target_x, target_y)
            await asyncio.sleep(0.002)

        CURSOR_TRACKER["x"] = target_x
        CURSOR_TRACKER["y"] = target_y
        return

    print(f"[EVASIVE] Gliding pointer smoothly to coordinate: ({int(target_x)}, {int(target_y)})")
    for i in range(1, steps + 1):
        t = i / steps
        s_curve = t * t * (3 - 2 * t) 
        
        next_x = start_x + (target_x - start_x) * s_curve
        next_y = start_y + (target_y - start_y) * s_curve
        
        jitter_x = next_x + random.uniform(-1.2, 1.2)
        jitter_y = next_y + random.uniform(-1.2, 1.2)
        
        await page.mouse.move(jitter_x, jitter_y)
        await asyncio.sleep(random.uniform(0.01, 0.02))
        
    CURSOR_TRACKER["x"] = target_x
    CURSOR_TRACKER["y"] = target_y

async def human_type(page, element, text):
    """Types text using explicit down/up events to simulate rigid robotic dwell times."""
    await element.click()
    await asyncio.sleep(0.1)
    
    if BOT_MODE == "NAIVE":
        print("[NAIVE] Dispatching inhuman high-speed keystroke array...")
        for char in text:
            # Forces extreme biometric indicators well beyond human muscle capability
            await page.keyboard.down(char)
            await asyncio.sleep(0.005) # Inhuman 5ms Dwell Time
            await page.keyboard.up(char)
            await asyncio.sleep(0.005) # Inhuman 5ms Flight Time
        return

    for char in text:
        await page.keyboard.down(char)
        await asyncio.sleep(random.uniform(0.04, 0.08))
        await page.keyboard.up(char)
        await asyncio.sleep(random.uniform(0.06, 0.14))

async def run_simulation():
    global CURSOR_TRACKER
    session_uid = str(uuid.uuid4())[:8]
    print(f"=== Sentinel-Bot Evaluation | Mode: [{BOT_MODE}] | Session: [{session_uid}] ===")
    
    async with async_playwright() as p:
        # Launch maximized browser window to give your coordinate normalizer clean scaling data
        browser = await p.chromium.launch(headless=False, args=["--start-maximized"])
        context = await browser.new_context(no_viewport=True)
        page = await context.new_page()
        
        # Start at top left reference anchor point
        CURSOR_TRACKER["x"], CURSOR_TRACKER["y"] = 0.0, 0.0
        await page.mouse.move(0, 0)
        
        target_url = "https://sentinel-bot-data-collection.vercel.app/inference.html"
        print(f"Navigating to production testing gateway: {target_url}")
        await page.goto(target_url)
        await page.wait_for_load_state("networkidle")
        
        # Sweep across viewport window space to populate tracking canvas arrays
        print("Pre-populating tracking arrays with an initialization sweep...")
        await page.mouse.move(400, 400, steps=40)
        await asyncio.sleep(0.1)
        CURSOR_TRACKER["x"], CURSOR_TRACKER["y"] = 400.0, 400.0
        
        # --- INPUT FIELD PROCESSING INTERACTION LOOP ---
        form_elements = page.locator("input, textarea")
        element_count = await form_elements.count()
        
        for i in range(element_count):
            element = form_elements.nth(i)
            if not await element.is_visible() or not await element.is_editable():
                continue
                
            element_id = await element.get_attribute("id")
            element_type = await element.get_attribute("type")
            
            if element_id == "username":
                payload_text = f"Bot_Node_{session_uid}"
            elif element_type == "email" or element_id == "email":
                payload_text = f"malicious_agent_{session_uid}@botnet-node.com"
            elif element.evaluate("el => el.tagName.toLowerCase() === 'textarea'") or element_id == "message":
                payload_text = "Automated scraping threat signature validation process string pattern. Injecting zero-entropy behavioral metrics to trigger active WAF block threshold limits."
            else:
                payload_text = f"Field_Data_{i}"
                
            await human_mouse_glide(page, element)
            await human_type(page, element, payload_text)
            await asyncio.sleep(0.4)

        # --- SUBMISSION PASS ---
        verification_button = page.locator("#downloadBtn")
        print("Gliding pointer straight to verification button (#downloadBtn)...")
        await human_mouse_glide(page, verification_button)
        await asyncio.sleep(0.2)
        
        print("Executing verification action click.")
        await verification_button.click()

        # --- EVALUATION STATUS WAITER ---
        print("Telemetry package sent. Waiting for classification decision...")
        
        try:
            await page.wait_for_selector("text=ACCESS GRANTED, text=ACCESS REJECTED", state="visible", timeout=15000)
            
            outcome_text = await page.locator("#downloadBtn").inner_text()
            print(f"\n[WAF RESPONSE] -> Element status text changed to: {outcome_text}")
            
            if "REJECTED" in outcome_text:
                print("\n" + "="*60)
                print(f"{Fore.GREEN} MITIGATION SUCCESS: WAF successfully reached critical threshold limits and blocked the session!")
                print("="*60)
            else:
                print("\n" + "="*60)
                print(f"{Fore.YELLOW} EVASION ALERT: The script bypassed the 80% threshold boundary.")
                print("="*60)
                
        except Exception:
            print(" Telemetry Timeout: The inference gateway dropped the request packet or timed out.")
        
        await asyncio.sleep(4)
        print("Tearing down browser instances safely.")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run_simulation())