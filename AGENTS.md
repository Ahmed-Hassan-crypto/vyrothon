# Vyrothon — ML Fine-Tuning Hackathon

## Task Overview
Fine-tune an open-weight model (≤2B params) for tool-calling on-device assistant.

## Key Constraints
- **Size**: ≤800MB after quantization (bonus at ≤250MB)
- **Latency**: ≤200ms/turn on Colab CPU
- **Offline**: No network calls at inference (AST-scanned)

## Tool Schema (Fixed)
```
weather(location, unit=C|F)
calendar(action=list|create, date, title?)
convert(value, from_unit, to_unit)
currency(amount, from_iso3, to_iso3)
sql(query)
```
Output: JSON in `<tool_call>...</tool_call>` tags

## Required Behaviors
1. Single-turn: valid tool call for unambiguous requests
2. Multi-turn: resolve references ("convert that to euros") against history
3. Refusals: plain text when no tool fits

## Bonus Collection Strategy

**+10 (Quantized ≤250MB) — Easiest bonus:**
- Use Qwen2.5-0.5B base model (naturally ~350MB, compresses to ~150MB with 4-bit AWQ/GPTQ)
- Test file size early and often during quantization

**+10 (Beat GPT-4o-mini on Slice C):**
- Generate 500+ adversarial examples: typos, code-switched prompts (Hindi/Spanish/Arabic), unit ambiguity, hallucination bait
- Include refusals: 30% of training data should be refusal cases (chitchat, impossible requests)
- Iterate on slice C during training — don't wait till end

**+5 (README error analysis):**
- Document what failed in evaluation and why
- Add a "Lessons Learned" section with specific debugging insights

## Hard Gates (fail = 0)
- Adapter loads on declared base model (transformers v5)
- Quantized model ≤500MB
- Mean latency ≤200ms/turn
- No network imports in inference.py
- Training data ≠ public test set (SHA-256)

## Submission
- `inference.py`: `run(prompt, history) -> str`
- Chatbot demo (Gradio/Streamlit/CLI)
- Trained artifacts (LoRA + quantized model)
- Reproducible via `make all` or documented commands

## Approach
1. Start with small base model (Qwen2.5-0.5B or similar)
2. Generate diverse synthetic training data (≥1000 examples)
3. Use LoRA fine-tuning with QLoRA for memory efficiency
4. Quantize with GPTQ/AWQ
5. Test latency early - don't wait until end

## File Structure Expected
```
Vyrothon/
├── inference.py
├── README.md
├── train.py or fine_tune.py
├── generate_data.py
├── evaluate.py
├── models/           # trained artifacts
├── data/             # synthetic data
└── AGENTS.md
```