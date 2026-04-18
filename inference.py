import os
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = "models"
BASE_MODEL = "Qwen/Qwen2-0.5B"
ADAPTER_DIR = os.path.join(MODEL_DIR, "adapter")
QUANTIZED_DIR = os.path.join(MODEL_DIR, "quantized")

SYSTEM_PROMPT = """You are a tool-calling assistant. When the user asks for weather, calendar, convert, currency, or SQL queries, respond with a JSON tool call inside <tool_call> tags.
For other requests, respond with plain text (no tool call).

Examples:
- User: "What's the weather in London?" -> <tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>
- User: "Hello!" -> I can't help with that.

Tools: weather(location, unit), calendar(action, date, title), convert(value, from, to), currency(amount, from, to), sql(query)."""

_cached_model = None
_cached_tokenizer = None

def _is_gptq_model(path):
    """Check if the model at path is a GPTQ quantized model."""
    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        return "quantization_config" in config and config.get("quantization_config", {}).get("quant_method") == "gptq"
    return False


def load_model():
    global _cached_model, _cached_tokenizer
    
    if _cached_model is not None and _cached_tokenizer is not None:
        return _cached_model, _cached_tokenizer
    
    model = None

    # Priority 1: Try quantized model
    if os.path.exists(QUANTIZED_DIR):
        try:
            if _is_gptq_model(QUANTIZED_DIR):
                # GPTQ model — use auto-gptq loader
                from auto_gptq import AutoGPTQForCausalLM
                model = AutoGPTQForCausalLM.from_quantized(
                    QUANTIZED_DIR,
                    device="cpu",
                    trust_remote_code=True,
                )
            else:
                # Regular or quanto/bnb model
                model = AutoModelForCausalLM.from_pretrained(
                    QUANTIZED_DIR,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            tokenizer = AutoTokenizer.from_pretrained(QUANTIZED_DIR, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            model = None

    # Priority 2: Try adapter on base model
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        if os.path.exists(ADAPTER_DIR):
            model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        else:
            model = base_model

    model.eval()
    _cached_model = model
    _cached_tokenizer = tokenizer
    return model, tokenizer

def run(prompt: str, history: list[dict] = None) -> str:
    if history is None:
        history = []
    
    model, tokenizer = load_model()
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        if isinstance(turn, dict) and "role" in turn:
            messages.append({"role": turn["role"], "content": turn.get("content", "")})
    messages.append({"role": "user", "content": prompt})
    
    # Build text without chat template
    text = ""
    for msg in messages:
        text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=80,
            temperature=0.0,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    print("Testing inference...")
    test_cases = [
        ("What's the weather in London?", []),
        ("Convert 100 meters to feet", []),
    ]
    for prompt, history in test_cases:
        import time
        start = time.time()
        response = run(prompt, history)
        latency = (time.time() - start) * 1000
        print(f"User: {prompt}")
        print(f"Response: {response}")
        print(f"Latency: {latency:.0f}ms\n")