import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = "models"
BASE_MODEL = "Qwen/Qwen2-0.5B"
ADAPTER_DIR = os.path.join(MODEL_DIR, "adapter")
QUANTIZED_DIR = os.path.join(MODEL_DIR, "quantized")
MERGED_DIR = os.path.join(MODEL_DIR, "_merged_temp")


def get_dir_size_mb(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def quantize_model():
    # ── Step 1: Load base model and merge LoRA adapter ──
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    if os.path.exists(ADAPTER_DIR):
        print("Loading and merging LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model = model.merge_and_unload()
        print("Adapter merged successfully.")
    else:
        print("WARNING: No adapter found. Using base model.")
        model = base_model

    # Save merged model to temp dir for reloading
    os.makedirs(MERGED_DIR, exist_ok=True)
    model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clean old quantized output
    if os.path.exists(QUANTIZED_DIR):
        shutil.rmtree(QUANTIZED_DIR)
    os.makedirs(QUANTIZED_DIR, exist_ok=True)

    # ── Step 2: Try quantization methods ──
    success = False

    # ─── Method 1: quanto int4 (target: ~200-250MB) ───
    if not success:
        print("\n" + "=" * 50)
        print("Method 1: quanto int4 quantization")
        print("=" * 50)
        try:
            from quanto import quantize, freeze, qint4

            print("Loading merged model...")
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )

            print("Quantizing weights to int4...")
            quantize(model, weights=qint4)
            print("Freezing quantized weights...")
            freeze(model)

            # Save with safe_serialization=False to avoid safetensors
            # compatibility issues with quanto tensor types
            print(f"Saving to {QUANTIZED_DIR} (PyTorch format)...")
            model.save_pretrained(QUANTIZED_DIR, safe_serialization=False)
            tokenizer.save_pretrained(QUANTIZED_DIR)

            size = get_dir_size_mb(QUANTIZED_DIR)
            print(f"quanto model size: {size:.1f} MB")
            success = True
            del model

        except Exception as e:
            print(f"quanto method 1 failed: {e}")
            # Clean partial output
            if os.path.exists(QUANTIZED_DIR):
                shutil.rmtree(QUANTIZED_DIR)
                os.makedirs(QUANTIZED_DIR, exist_ok=True)

    # ─── Method 2: quanto via transformers QuantoConfig ───
    if not success:
        print("\n" + "=" * 50)
        print("Method 2: quanto via QuantoConfig")
        print("=" * 50)
        try:
            from transformers import QuantoConfig

            quanto_config = QuantoConfig(weights="int4")
            print("Loading model with QuantoConfig int4...")
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR,
                quantization_config=quanto_config,
                device_map="cpu",
                trust_remote_code=True,
            )

            print(f"Saving to {QUANTIZED_DIR}...")
            model.save_pretrained(QUANTIZED_DIR, safe_serialization=False)
            tokenizer.save_pretrained(QUANTIZED_DIR)

            size = get_dir_size_mb(QUANTIZED_DIR)
            print(f"QuantoConfig model size: {size:.1f} MB")
            success = True
            del model

        except Exception as e:
            print(f"QuantoConfig failed: {e}")
            if os.path.exists(QUANTIZED_DIR):
                shutil.rmtree(QUANTIZED_DIR)
                os.makedirs(QUANTIZED_DIR, exist_ok=True)

    # ─── Method 3: Manual int8 quantization ───
    if not success:
        print("\n" + "=" * 50)
        print("Method 3: Manual int8 dynamic quantization")
        print("=" * 50)
        try:
            print("Loading merged model...")
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )

            print("Applying PyTorch dynamic int8 quantization...")
            model_fp32 = model.float()
            quantized_model = torch.quantization.quantize_dynamic(
                model_fp32,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )

            print(f"Saving to {QUANTIZED_DIR}...")
            quantized_model.save_pretrained(QUANTIZED_DIR, safe_serialization=False)
            tokenizer.save_pretrained(QUANTIZED_DIR)

            size = get_dir_size_mb(QUANTIZED_DIR)
            print(f"Int8 model size: {size:.1f} MB")
            success = True
            del model, model_fp32, quantized_model

        except Exception as e:
            print(f"Manual int8 failed: {e}")
            if os.path.exists(QUANTIZED_DIR):
                shutil.rmtree(QUANTIZED_DIR)
                os.makedirs(QUANTIZED_DIR, exist_ok=True)

    # ─── Method 4: bitsandbytes 4-bit (fallback) ───
    if not success:
        print("\n" + "=" * 50)
        print("Method 4: bitsandbytes 4-bit (fallback)")
        print("=" * 50)
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            print("Loading with bitsandbytes 4-bit...")
            model = AutoModelForCausalLM.from_pretrained(
                MERGED_DIR,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )

            print(f"Saving to {QUANTIZED_DIR}...")
            model.save_pretrained(QUANTIZED_DIR)
            tokenizer.save_pretrained(QUANTIZED_DIR)

            size = get_dir_size_mb(QUANTIZED_DIR)
            print(f"bitsandbytes model size: {size:.1f} MB")
            success = True
            del model

        except Exception as e:
            print(f"bitsandbytes failed: {e}")

    # ── Cleanup temp ──
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR, ignore_errors=True)

    # ── Report ──
    size_mb = get_dir_size_mb(QUANTIZED_DIR)
    print("\n" + "=" * 50)
    print(f"FINAL QUANTIZED MODEL SIZE: {size_mb:.1f} MB")
    print("=" * 50)

    if size_mb > 500:
        print("❌ FAIL: Exceeds 500MB hard gate!")
    elif size_mb > 250:
        print("⚠️  Passes hard gate but misses ≤250MB bonus.")
    else:
        print("✅ SUCCESS: ≤250MB — eligible for +10 bonus!")

    return size_mb


if __name__ == "__main__":
    size = quantize_model()
    print(f"\nFinal size: {size:.1f} MB")