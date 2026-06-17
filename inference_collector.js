// =====================================================================
// SENTINEL-BOT INFERENCE ENGINE COUPLER
// inference_collector.js (v5.2 Production-Hardened Dynamic Edition)
// =====================================================================

let interactionData = {
    sessionId: crypto.randomUUID(),
    label: "human", 
    mouseMovements: [],
    keystrokes: [],
    device: {
        cores: navigator.hardwareConcurrency || "unknown",
        memory: navigator.deviceMemory || "unknown",
        platform: navigator.platform
    },
    visibilityShifts: [] 
};

// 1. High-Resolution Normalized Mouse Tracking
window.addEventListener('mousemove', (e) => {
    const now = performance.now();
    const last = interactionData.mouseMovements[interactionData.mouseMovements.length - 1];
   
    const normX = (e.clientX / (window.innerWidth || 1)).toFixed(4);
    const normY = (e.clientY / (window.innerHeight || 1)).toFixed(4);
   
    let velocity = 0;
    if (last) {
        const dist = Math.sqrt(Math.pow(normX - last.x, 2) + Math.pow(normY - last.y, 2));
        velocity = dist / (now - last.t);
    }

    interactionData.mouseMovements.push({
        x: parseFloat(normX), 
        y: parseFloat(normY),
        t: parseFloat(now.toFixed(2)),
        v: parseFloat(velocity.toFixed(5))
    });

    if (interactionData.mouseMovements.length > 2000) interactionData.mouseMovements.shift();
});

// 2. Keystroke Dynamics
window.addEventListener('keydown', (e) => {
    interactionData.keystrokes.push({ key: e.key, type: 'down', t: parseFloat(performance.now().toFixed(2)) });
});
window.addEventListener('keyup', (e) => {
    interactionData.keystrokes.push({ key: e.key, type: 'up', t: parseFloat(performance.now().toFixed(2)) });
});

window.addEventListener('blur', () => {
    interactionData.visibilityShifts.push({ event: 'blur', t: parseFloat(performance.now().toFixed(2)) });
});
window.addEventListener('focus', () => {
    interactionData.visibilityShifts.push({ event: 'focus', t: parseFloat(performance.now().toFixed(2)) });
});

// 3. Dynamic Upload & Real-Time Classification Ingress Logic
document.getElementById('downloadBtn').addEventListener('click', async () => {
    const statusBtn = document.getElementById('downloadBtn');
    const statusMessage = document.getElementById('statusMessage');

    statusBtn.innerText = "Verifying your interactions...";
    statusBtn.style.backgroundColor = "#f39c12";
    if (statusMessage) { statusMessage.style.display = "none"; }

    const UPLOAD_URL = '/api/inference';

    try {
        let payload = {
            sessionId: interactionData.sessionId,
            label: interactionData.label,
            mouseMovements: interactionData.mouseMovements,
            keystrokes: interactionData.keystrokes,
            visibilityShifts: interactionData.visibilityShifts,
            device: interactionData.device,
            // FIX: Explicitly populates the network layer fields for our ML model
            network_layer: {
                userAgent: navigator.userAgent,
                acceptLanguage: navigator.language || navigator.userLanguage || "en-US"
            },
            contextual_features: {
                webdriver: window.navigator.webdriver ? 1.0 : 0.0,
                ua_is_chrome: /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor) ? 1.0 : 0.0
            }
        };

        const response = await fetch(UPLOAD_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`WAF Edge Gateway Error: Status ${response.status}`);
        }

        const result = await response.json();
        if (statusMessage) { statusMessage.style.display = "block"; }

        // Clean-room presentation UI isolates final decisions cleanly from dashboard telemetry
        if (result.decision === "BLOCK") {
            statusBtn.innerText = "ACCESS REJECTED";
            statusBtn.style.backgroundColor = "#e74c3c";
            if (statusMessage) {
                statusMessage.style.color = "#e74c3c";
                statusMessage.innerHTML = `<strong>Mitigated Automated Session Signature.</strong><br>` +
                                          `Bot Probability: ${(result.bot_probability * 100).toFixed(2)}%`;
            }
        } else {
            statusBtn.innerText = "ACCESS GRANTED";
            statusBtn.style.backgroundColor = "#27ae60";
            if (statusMessage) {
                statusMessage.style.color = "#27ae60";
                statusMessage.innerHTML = `<strong>Authenticated Human Verified.</strong><br>` +
                                          `Bot Probability: ${(result.bot_probability * 100).toFixed(2)}%`;
            }
        }

    } catch (err) {
        console.error("[CRITICAL GATEWAY EXCEPTION]", err);
        statusBtn.innerText = "Perimeter Fault - Retry";
        statusBtn.style.backgroundColor = "#e74c3c";
        if (statusMessage) {
            statusMessage.style.display = "block";
            statusMessage.style.color = "#e74c3c";
            statusMessage.innerText = `Network communication fault: ${err.message}`;
        }
    }
});