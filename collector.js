// SENTINEL-BOT
// collector.js - REFINED SENTINEL-BOT Behavioral Biometrics
let interactionData = {
    sessionId: crypto.randomUUID(),
    label: "human", // Added for server-side smart labeling
    mouseMovements: [],
    keystrokes: [],
    device: {
        cores: navigator.hardwareConcurrency || "unknown",
        memory: navigator.deviceMemory || "unknown",
        platform: navigator.platform
    }
};

// 1. High-Resolution Normalized Mouse Tracking
window.addEventListener('mousemove', (e) => {
    const now = performance.now();
    const last = interactionData.mouseMovements[interactionData.mouseMovements.length - 1];
   
    // SPATIAL NORMALIZATION: Convert pixels to 0.0 - 1.0 range
    const normX = (e.clientX / (window.innerWidth || 1)).toFixed(4);
    const normY = (e.clientY / (window.innerHeight || 1)).toFixed(4);
   
    let velocity = 0;
    if (last) {
        const dist = Math.sqrt(Math.pow(normX - last.x, 2) + Math.pow(normY - last.y, 2));
        velocity = dist / (now - last.t);
    }

    interactionData.mouseMovements.push({
        x: parseFloat(normX), // Store as numbers for easier AI processing
        y: parseFloat(normY),
        t: parseFloat(now.toFixed(2)),
        v: parseFloat(velocity.toFixed(5))
    });

    if (interactionData.mouseMovements.length > 1000) interactionData.mouseMovements.shift();
});

// 2. Keystroke Dynamics
window.addEventListener('keydown', (e) => {
    interactionData.keystrokes.push({ key: e.key, type: 'down', t: parseFloat(performance.now().toFixed(2)) });
});
window.addEventListener('keyup', (e) => {
    interactionData.keystrokes.push({ key: e.key, type: 'up', t: parseFloat(performance.now().toFixed(2)) });
});

// 3. Dynamic Upload Logic
document.getElementById('downloadBtn').addEventListener('click', async () => {
    const statusBtn = document.getElementById('downloadBtn');

    statusBtn.innerText = "Syncing to Sentinel Cloud...";
    statusBtn.style.backgroundColor = "#f39c12";

    const UPLOAD_URL = '/api/upload';

    try {

        interactionData.tls_layer = {
            browser: navigator.userAgent,
            method: "client_placeholder"
        };

        const response = await fetch(UPLOAD_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(interactionData)
        });

        if (response.ok) {
            statusBtn.innerText = "Sync Complete";
            statusBtn.style.backgroundColor = "#27ae60";
        } else {
            throw new Error("Server Error");
        }

    } catch (err) {
        console.error(err);
        statusBtn.innerText = "Retry Sync";
        statusBtn.style.backgroundColor = "#e74c3c";
    }
});