# ======================================================================
# SENTINEL-BOT DECOUPLED INFRASTRUCTURE AUDIT ENGINE
# inspect_bundle.py
# Standalone execution script (Bypasses runtime_config production guards)
# ======================================================================
import joblib

# SURGICAL PATCH: Directly map model path string to bypass runtime_config guards
MODEL_BINARY_PATH = "full_Hybrid_Stacking_v1.8.pkl"

try:
    print("\n" + "="*70)
    print("      SENTINEL-BOT MULTIMODAL FEATURE REGISTRY VERIFICATION AUDIT")
    print("="*70)
    
    # Safely load pickle bundle directly via joblib
    bundle = joblib.load(MODEL_BINARY_PATH)
    
    # 1. Extract and audit structural feature vectors
    numeric_cols = bundle.get("numeric_cols", [])
    print(f"\n[✓] Model Binary Loaded Successfully.")
    print(f"[✓] Total Numerical Inference Features Expected: {len(numeric_cols)}")
    print("\nExact Structural Feature Column Names Inside Stacking Matrix:")
    print("-" * 70)
    
    # Print sorted names to spot discrepancies easily
    for index, column_name in enumerate(sorted(numeric_cols), 1):
        print(f"  {index:02d}. {column_name}")
        
    print("-" * 70)
    print("=  AUDIT COMPLETE: Cross-check the names above with predict_live.py  =")
    print("="*70 + "\n")
    
except FileNotFoundError:
    print(f"\n[!] FILE NOT FOUND: Ensure '{MODEL_BINARY_PATH}' is placed inside:")
    print("    C:\\Users\\aamin\\Professional\\VS Projects\\Sentinel-Bot-Project\\Sentinel-Bot-Project\\\n")
except Exception as e:
    print(f"\n[!] METRICS CRITICAL EXCEPTION: Failed to read model bundle: {str(e)}\n")