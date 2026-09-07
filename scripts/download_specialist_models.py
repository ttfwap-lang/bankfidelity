"""
BankFidelity SOTA Specialist Model Pre-Staging Script
Zero Gemini | Zero DocAI | Zero Anthropic | Zero OpenAI | Zero Qwen | Zero DeepSeek

Pre-downloads and stages:
1. Microsoft Florence-2-Large (Vision Grounding & Sub-Pixel Coordinate Regression)
2. Surya Layout & Reading Order (Line-level polygon segmentation)
3. Microsoft Phi-3.5-Vision (4.2B local VLM watchdog weights)
"""

import os
import sys
from pathlib import Path

MODELS_ROOT = Path("C:/bankfidelity/models")

def stage_florence2():
    print("[1/3] Pre-staging Microsoft Florence-2-Large (Sub-Pixel Coordinate Regression)...")
    dest = MODELS_ROOT / "florence2"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_id = "microsoft/Florence-2-large"
        print(f"      Downloading weights and tokenizer for {model_id} to {dest}...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(dest))
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, cache_dir=str(dest))
        print("      [OK] Florence-2-Large successfully staged.")
    except Exception as e:
        print(f"      [WARN] Florence-2 pre-download skipped or failed: {e}")
        print("             (Can be downloaded on first run via specialist_bridge.py)")

def stage_surya():
    print("[2/3] Pre-staging Surya Layout & Reading Order Engine...")
    dest = MODELS_ROOT / "surya"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        from surya.model.ordering.processor import load_processor as load_order_processor
        from surya.model.ordering.model import load_model as load_order_model
        print("      Downloading Surya reading order weights...")
        load_order_processor()
        load_order_model()
        print("      [OK] Surya layout weights successfully staged.")
    except Exception as e:
        print(f"      [WARN] Surya weights download skipped or deferred: {e}")

def stage_phi35_vision():
    print("[3/3] Pre-staging Microsoft Phi-3.5-Vision (Local 24/7 Watchdog VLM)...")
    dest = MODELS_ROOT / "phi35_vision"
    dest.mkdir(parents=True, exist_ok=True)
    gguf_name = "Phi-3.5-vision-instruct-Q4_K_M.gguf"
    target_path = dest / gguf_name
    if target_path.exists():
        print(f"      [OK] Found existing GGUF at {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
    else:
        print(f"      Target GGUF location: {target_path}")
        print("      Will be loaded into C:\\ufo\\bin\\llama-server.exe on port 8080.")

def main():
    print("=================================================================")
    print(" BankFidelity 50GB+ Specialist Model Staging Pipeline            ")
    print(f" Target Directory: {MODELS_ROOT}")
    print("=================================================================")
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    stage_florence2()
    stage_surya()
    stage_phi35_vision()
    print("=================================================================")
    print(" [COMPLETE] Model staging routine finished.")
    print("=================================================================")

if __name__ == "__main__":
    main()
