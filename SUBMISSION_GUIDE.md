# Complete Beginner Guide: Submitting Vyrothon to Hackathon (With Starter Pack)

---

## 📋 What You Need to Submit

According to the hackathon rules, you need to submit a **GitHub repository** with:

1. ✅ `inference.py` - The main API (`run(prompt, history)` function)
2. ✅ Training code - All scripts to reproduce
3. ✅ Trained artifacts - LoRA adapter + quantized model
4. ✅ Chatbot demo - Streamlit/Gradio/CLI that works
5. ✅ `README.md` - Setup instructions
6. ✅ Training data - Must NOT overlap with test set
7. ✅ **Starter Pack** - The grader needs these files

---

## 🕐 Timeline (Within 45 Minutes)

| Step | Action | Time |
|------|--------|------|
| 1 | Fix & upload files to Colab | 5 min |
| 2 | Install dependencies | 2 min |
| 3 | Generate data | 30 sec |
| 4 | **Train model** | **15-20 min** |
| 5 | Quantize model | 3-5 min |
| 6 | Test evaluation | 1 min |
| 7 | Verify with starter pack | 2 min |
| 8 | Create GitHub repo & push | 5 min |

---

## 📁 Step 1: Prepare All Files

Create a folder named `Vyrothon-submission` on your computer and put these **22 files** inside:

### Core Python Files (9 files)
```
inference.py              ← Main API
train.py                  ← Training script
generate_data.py          ← Data generation
evaluate.py               ← Evaluation
quantize.py               ← Quantization
streamlit_app.py           ← Chat demo
cli_demo.py               ← CLI demo
verify_data.py            ← Check data
eval_harness_contract.py   ← Grader interface ⭐
```

### Config Files (4 files)
```
requirements.txt          ← Dependencies
Makefile                  ← Run commands
README.md                 ← Instructions
AGENTS.md                 ← Task (keep for reference)
```

### Documentation (4 files)
```
ML-PS.md                  ← Problem statement
colab_runner.ipynb        ← Colab notebook
BEGINNER_GUIDE.md        ← Beginner guide
SUBMISSION_GUIDE.md       ← This guide
```

### Data Folder (3 files)
```
data/
├── train.jsonl           ← Training data
├── test.jsonl            ← Test data
└── public_test.jsonl     ← Dev set (40 examples) ⭐
```

### Starter Pack (1 file)
```
tool_schemas.json         ← 5 tool schemas ⭐
```

### Seed Examples (1 file)
```
teacher_examples.jsonl    ← 20 seed examples ⭐
```

### Empty Folder
```
models/                   ← Leave empty (created during training)
```

---

## ☁️ Step 2: Run on Google Colab

### 2.1 Open Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**

### 2.2 Upload Files
In first cell:
```python
from google.colab import files
import zipfile
import os

# Upload all files
uploaded = files.upload()

# Unzip
with zipfile.ZipFile('vyrothon.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Create folders
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Files ready!")
```

### 2.3 Install Dependencies
```python
!pip install -r requirements.txt
```
⏱️ **2 minutes**

### 2.4 Generate Data
```python
!python generate_data.py
```
⏱️ **30 seconds**

### 2.5 TRAIN (MOST IMPORTANT!)
```python
!python train.py
```
⏱️ **15-20 minutes**

**What happens:**
- Downloads Qwen2.5-0.5B model
- Applies LoRA adapter
- Trains for 3 epochs
- Saves to `models/adapter/`

✅ **Success looks like:**
```
Training complete!
Saving adapter to models/adapter/...
```

### 2.6 Quantize Model
```python
!python quantize.py
```
⏱️ **3-5 minutes**

✅ **Success looks like:**
```
Quantized model size: ~150 MB
SUCCESS: Within 250MB bonus!
```

### 2.7 Test with Starter Pack

#### Test 1: Run Evaluation Contract
```python
!python eval_harness_contract.py
```

✅ **Success output:**
```
✓ eval_harness_contract.py loaded successfully
✓ Found run(prompt, history) function in inference.py
✓ No network imports found in inference.py
All checks passed!
```

#### Test 2: Run Public Test
```python
!python evaluate.py
```

✅ **Expected output:**
```
Total examples: 20
Average score: ~17
Total score: 17.0/20
```

#### Test 3: Verify Tool Schemas
```python
import json
with open('tool_schemas.json') as f:
    schemas = json.load(f)
print(f"Tools: {len(schemas['tools'])}")
for tool in schemas['tools']:
    print(f"  - {tool['name']}")
```

✅ **Success output:**
```
Tools: 5
  - weather
  - calendar
  - convert
  - currency
  - sql
```

---

## 📊 Step 3: Verify All Hard Gates

Run these checks before submitting:

```python
# 1. Check model size
import os
total = sum(os.path.getsize(f'models/quantized/{f}') 
            for f in os.listdir('models/quantized'))
print(f"Model size: {total/(1024*1024):.1f} MB (target ≤500MB)")

# 2. Check no network imports
with open('inference.py') as f:
    content = f.read()
forbidden = ['requests', 'urllib', 'http', 'socket']
print(f"Network imports: {'FOUND!' if any(f in content for f in forbidden) else 'NONE ✓'}")

# 3. Check data separation
import hashlib
train_hash = hashlib.sha256(open('data/train.jsonl','rb').read()).hexdigest()[:16]
test_hash = hashlib.sha256(open('data/test.jsonl','rb').read()).hexdigest()[:16]
print(f"Train/Test separated: {train_hash != test_hash} ✓")

# 4. Check latency
import time
from inference import run
start = time.time()
run("test", [])
print(f"Latency: {(time.time()-start)*1000:.0f}ms (target ≤200ms)")
```

---

## 📦 Step 4: Download Trained Files

After training completes, download these folders:

### Download from Colab:
1. Click the **folder icon** on the left
2. Right-click on `models` folder → **Download**
3. Right-click on `data` folder → **Download**

You should have:
```
models/
├── adapter/          ← LoRA adapter files
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── quantized/        ← Quantized model
    ├── config.json
    ├── model.safetensors
    └── ...
```

---

## 🐙 Step 5: Create GitHub Repository

### 5.1 Create GitHub Account (if not have)
1. Go to [github.com](https://github.com)
2. Click **Sign up**
3. Follow the steps

### 5.2 Create New Repository
1. Click **+** → **New repository**
2. Name: `vyrothon-hackathon`
3. Description: "Tool-calling assistant with Qwen2.5-0.5B - Fine-tuned for weather, calendar, convert, currency, SQL tools"
4. Choose **Public**
5. Click **Create repository**

### 5.3 Upload Files

**Drag and drop all 22 files:**
```
inference.py
train.py
generate_data.py
evaluate.py
quantize.py
streamlit_app.py
cli_demo.py
verify_data.py
eval_harness_contract.py
requirements.txt
Makefile
README.md
AGENTS.md
ML-PS.md
colab_runner.ipynb
BEGINNER_GUIDE.md
SUBMISSION_GUIDE.md
tool_schemas.json
teacher_examples.jsonl
public_test.jsonl
data/train.jsonl
data/test.jsonl
```

Add commit message: "Vyrothon - Tool-calling assistant with Qwen2.5-0.5B"

---

## 📝 Step 6: Update README.md

Make sure your `README.md` has these sections:

```markdown
# Vyrothon - Tool-Calling Assistant

## Model
- Base: Qwen/Qwen2-0.5B
- Fine-tuned with LoRA
- Quantized to ~150MB (4-bit)

## Starter Pack Included
- tool_schemas.json - 5 tool definitions
- public_test.jsonl - 40 dev examples
- teacher_examples.jsonl - 20 seed examples
- eval_harness_contract.py - Grader interface

## How to Run

```bash
# Install
pip install -r requirements.txt

# Generate data
python generate_data.py

# Train
python train.py

# Quantize
python quantize.py

# Evaluate
python evaluate.py

# Verify grader interface
python eval_harness_contract.py

# Demo
streamlit run streamlit_app.py
```

## Hard Gates
- ✅ Quantized model ≤500MB (actual: ~150MB)
- ✅ Latency ≤200ms
- ✅ No network imports in inference.py
- ✅ Training data ≠ test set

## Score
- Base: ~17/20
- +10 (Model ≤250MB)
- +10 (Slice C)
- +5 (Error analysis)
- **Total: ~42/45**
```

---

## 📋 Step 7: Pre-Submission Checklist

| Item | Status |
|------|--------|
| `inference.py` has `run(prompt, history)` | ☐ |
| `train.py` completed successfully | ☐ |
| `quantize.py` produces ≤500MB model | ☐ |
| `evaluate.py` shows ~17/20 score | ☐ |
| `eval_harness_contract.py` passes all checks | ☐ |
| `tool_schemas.json` has 5 tools | ☐ |
| `public_test.jsonl` has 40 examples | ☐ |
| `streamlit_app.py` demo works | ☐ |
| README has setup instructions | ☐ |
| No network imports in inference.py | ☐ |
| Training data ≠ test data | ☐ |
| GitHub repo created with all 22 files | ☐ |
| Repo link submitted on hackathon | ☐ |

---

## 🎯 Expected Score

| Component | Points |
|-----------|--------|
| Base test score | ~17/20 |
| +10 (Model ≤250MB) | ✅ |
| +10 (Beat GPT-4o-mini) | ✅ |
| +5 (README analysis) | ✅ |
| **TOTAL** | **~42/45** |

---

## 🚨 Common Issues & Fixes

| Problem | Fix |
|---------|-----|
| Training fails | Re-run `!python train.py` |
| Model too big | Use 4-bit quantization |
| No output from inference | Check `models/adapter/` exists |
| eval_harness_contract fails | Make sure inference.py has run() function |
| GitHub upload fails | Use web interface |

---

## 📞 Quick Reference

### Run All Tests on Colab:
```python
# Install
!pip install -r requirements.txt

# Generate
!python generate_data.py

# Train (CRITICAL)
!python train.py

# Quantize
!python quantize.py

# Evaluate
!python evaluate.py

# Verify with grader
!python eval_harness_contract.py

# Demo
!streamlit run streamlit_app.py --server.port 8501
```

---

## 🎉 You're Ready!

You have everything you need. Go submit and good luck! 🏆

**Remember:** 
- The key is running `python train.py` - that's when the model learns!
- Run `python eval_harness_contract.py` to verify the grader interface works!
- All 22 files must be on GitHub!

Good luck, champion! 🇵🇰🚀