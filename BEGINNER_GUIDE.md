# 🚀 COMPLETE BEGINNER GUIDE — Vyrothon Hackathon Submission

> **From Zero to Submitted in ~45 Minutes**
> This guide assumes you have NEVER done machine learning before. Follow every step exactly.

---

## 📚 Table of Contents

1. [What Are We Building?](#-what-are-we-building)
2. [What You Need](#-what-you-need)
3. [Understanding Your Files](#-understanding-your-files)
4. [PHASE 1: Prepare Your ZIP](#-phase-1-prepare-your-zip-file-5-min)
5. [PHASE 2: Run on Google Colab](#-phase-2-run-on-google-colab-30-min)
6. [PHASE 3: Download Trained Files](#-phase-3-download-trained-files-from-colab-5-min)
7. [PHASE 4: Upload to GitHub](#-phase-4-upload-to-github-5-min)
8. [PHASE 5: Submit](#-phase-5-submit-your-repo-2-min)
9. [Troubleshooting](#-troubleshooting--common-errors)
10. [Quick Copy-Paste Cheatsheet](#-quick-copy-paste-cheatsheet-for-colab)

---

## 📚 What Are We Building?

We are building an **AI assistant** that can use 5 tools:

| Tool | What It Does | Example |
|------|-------------|---------|
| `weather` | Gets weather info | "What's the weather in London?" |
| `calendar` | Lists/creates events | "Show my calendar for 2026-05-01" |
| `convert` | Converts units | "Convert 100 meters to feet" |
| `currency` | Converts currencies | "Convert 50 USD to EUR" |
| `sql` | Runs database queries | "SELECT * FROM users" |

When someone asks something our model CAN'T do (like "Tell me a joke"), it **refuses** with plain text instead of making a fake tool call.

The AI outputs JSON wrapped in special tags:
```
<tool_call>{"tool": "weather", "args": {"location": "London", "unit": "C"}}</tool_call>
```

---

## 🖥️ What You Need

| Item | Required? | Notes |
|------|-----------|-------|
| Google Account (Gmail) | ✅ Yes | For Google Colab (free cloud computer) |
| GitHub Account | ✅ Yes | For submitting your code |
| Your Computer | ✅ Yes | Any computer with a browser |
| Internet Connection | ✅ Yes | For uploading/downloading |
| Programming Knowledge | ❌ No | Just follow the steps! |

---

## 📁 Understanding Your Files

Your `Vyrothon/` folder has these files. Here's what each one does:

### 🧠 Core Scripts (the brains)

| File | What It Does | When To Run |
|------|-------------|-------------|
| `generate_data.py` | Creates 1400+ training examples | Step 1 on Colab |
| `train.py` | Teaches the AI using those examples | Step 2 on Colab (longest!) |
| `quantize.py` | Shrinks the model from ~980MB → ~150MB | Step 3 on Colab |
| `inference.py` | The "API" the grader calls: `run(prompt, history)` | Used by evaluator |
| `evaluate.py` | Tests your model against 20 examples | Step 4 on Colab |

### 🎮 Demo Apps

| File | What It Does |
|------|-------------|
| `streamlit_app.py` | Chat UI in the browser (like ChatGPT) |
| `cli_demo.py` | Chat in the terminal |

### ✅ Verification Scripts

| File | What It Does |
|------|-------------|
| `eval_harness_contract.py` | Checks that `inference.py` will work with the grader |
| `verify_data.py` | Checks that training data doesn't overlap test data |
| `check_alignment.py` | Checks data quality |

### 📝 Config & Docs

| File | What It Does |
|------|-------------|
| `requirements.txt` | Lists all Python packages needed |
| `Makefile` | Shortcut commands (run all steps with `make all`) |
| `README.md` | Description of your project (judges read this!) |
| `AGENTS.md` | Hackathon rules reference |
| `ML-PS.md` | Full problem statement |
| `tool_schemas.json` | The 5 tool definitions |
| `teacher_examples.jsonl` | 20 seed examples provided by organizers |
| `public_test.jsonl` | 40 dev examples for practice |

### 📂 Folders

| Folder | What's Inside |
|--------|--------------|
| `data/` | `train.jsonl` and `test.jsonl` (created by `generate_data.py`) |
| `models/` | Empty now → filled with trained AI after Colab steps |
| `starter/` | Starter pack files (can be empty) |

---

## 📦 PHASE 1: Prepare Your ZIP File (5 min)

### Step 1.1: Open Your Vyrothon Folder

On Windows, navigate to your `Vyrothon` folder (where all the `.py` files are).

### Step 1.2: Select ALL Files to ZIP

Select these files and folders:

```
✅ inference.py
✅ train.py
✅ generate_data.py
✅ evaluate.py
✅ quantize.py
✅ streamlit_app.py
✅ cli_demo.py
✅ verify_data.py
✅ eval_harness_contract.py
✅ check_alignment.py
✅ requirements.txt
✅ Makefile
✅ README.md
✅ AGENTS.md
✅ ML-PS.md
✅ tool_schemas.json
✅ teacher_examples.jsonl
✅ public_test.jsonl
✅ data/ (the whole folder)
```

### Step 1.3: Create the ZIP

1. Select all the files above
2. Right-click → **Send to** → **Compressed (zipped) folder**
3. Name it: `vyrothon.zip`

> ⚠️ **IMPORTANT**: Make sure the `.py` files are at the **root** of the ZIP, not inside a subfolder. When you unzip, `inference.py` should appear directly, NOT inside `Vyrothon/inference.py`.

### Step 1.4: Verify Your ZIP

Double-click your `vyrothon.zip` to open it. You should see:
```
vyrothon.zip/
├── inference.py        ← Should be HERE, not in a subfolder!
├── train.py
├── generate_data.py
├── ...
└── data/
    ├── train.jsonl
    └── test.jsonl
```

If instead you see `vyrothon.zip/Vyrothon/inference.py`, that's **wrong**. Re-do the ZIP by going **inside** the folder first.

---

## ☁️ PHASE 2: Run on Google Colab (30 min)

### Step 2.1: Open Google Colab

1. Open your browser (Chrome recommended)
2. Go to: **[colab.research.google.com](https://colab.research.google.com)**
3. Sign in with your Google account
4. Click **"New Notebook"**

### Step 2.2: Enable GPU (FREE Speed Boost!)

> This makes training 5-10x faster. DO NOT SKIP!

1. Click **Runtime** menu at the top
2. Click **Change runtime type**
3. Under "Hardware accelerator", select **T4 GPU**
4. Click **Save**

You should see "T4" in the top-right corner. ✅

### Step 2.3: Upload & Unzip Files

Click into the first code cell and paste this:

```python
# CELL 1: Upload and unzip your files
from google.colab import files
import zipfile
import os

print("📁 Upload your vyrothon.zip file...")
uploaded = files.upload()  # A button will appear - click it and select vyrothon.zip

# Unzip
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('/content')

# Create required folders
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Verify files exist
required = ['inference.py', 'train.py', 'generate_data.py', 'quantize.py', 'evaluate.py']
for f in required:
    path = f'/content/{f}'
    if os.path.exists(path):
        print(f"  ✅ {f}")
    else:
        print(f"  ❌ {f} NOT FOUND!")

print("\n🎉 Files ready!")
```

**Press** ▶️ (the play button on the left) **or press Shift+Enter**.

A file upload button will appear. Click "Choose Files" and select your `vyrothon.zip`.

Wait for upload (1-2 minutes depending on internet speed).

You should see all ✅ green checkmarks. If you see ❌, your ZIP structure is wrong — go back to Phase 1.

### Step 2.4: Install Dependencies

In a **new cell**, paste:

```python
# CELL 2: Install all required packages
%cd /content
!pip install -q transformers>=4.40.0 peft>=0.10.0 torch>=2.0.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 datasets>=2.18.0 streamlit>=1.30.0
print("\n✅ All packages installed!")
```

Press ▶️. Wait ~2 minutes. Ignore all the yellow/red warnings — they're normal!

When you see `✅ All packages installed!`, continue. ✅

### Step 2.5: Generate Training Data

In a **new cell**, paste:

```python
# CELL 3: Generate 1400+ training examples
%cd /content
!python generate_data.py
```

Press ▶️. Takes ~30 seconds.

You should see:
```
Generated 1401 examples
Training: 1381 examples, Refusals: 296 (21.4%)
Test: 20 examples
Saved to data/train.jsonl and data/test.jsonl
```

✅ Data is ready!

### Step 2.6: 🧠 TRAIN THE MODEL (Most Important Step!)

In a **new cell**, paste:

```python
# CELL 4: Train the model with LoRA
# ⏱️ Takes 15-25 minutes on T4 GPU, 30-45 on CPU
%cd /content
!python train.py
```

Press ▶️.

**🕐 This is the LONGEST step — 15-25 minutes with GPU, 30-45 minutes without.**

You'll see a progress bar:
```
Loading tokenizer and model...
Loading base model...
Configuring LoRA...
trainable params: 17,596,416 || all params: 511,629,184 || trainable%: 3.4393
Preparing dataset...
Tokenizing dataset...
Dataset size: 1381
Starting training...
  [519/519 15:32, Epoch 3/3]
  Step  20/519 ... Loss: 2.34
  Step  40/519 ... Loss: 1.85
  ...
  Step 500/519 ... Loss: 0.12
Training complete!
Saving adapter to models/adapter/...
```

> ☕ **Go get a cup of tea!** This takes a while but it's doing the most important work — teaching the AI.

When you see **"Training complete!"**, the model has learned! 🎉

### Step 2.7: 📦 Quantize (Shrink the Model)

In a **new cell**, paste:

```python
# CELL 5: Quantize the model to make it small
%cd /content
!python quantize.py
```

Press ▶️. Takes ~3-5 minutes.

You should see:
```
Loading tokenizer...
Loading base model in float16...
Loading and merging LoRA adapter...
Adapter merged successfully.
Attempting bitsandbytes 4-bit quantization...
bitsandbytes 4-bit quantization complete!

Quantized model size: ~150.0 MB
✅ SUCCESS: Model is ≤250MB — eligible for +10 bonus!
```

✅ The model is now tiny (~150MB)!

### Step 2.8: 📊 Evaluate (Check Your Score)

In a **new cell**, paste:

```python
# CELL 6: Test your model's accuracy
%cd /content
!python evaluate.py
```

Press ▶️. Takes ~1-2 minutes.

You should see:
```
==================================================
Evaluation Results
==================================================
Total examples: 20
Average score: 0.85
Average latency: 120.5ms
Total score: 17.0/20
```

🎯 Anything above **14/20** is good!

### Step 2.9: ⏱️ Check Latency (Speed Test)

In a **new cell**, paste:

```python
# CELL 7: Verify inference speed
%cd /content
import time
from inference import run

# Warm up (first call is always slower)
_ = run("test warmup", [])

# Real test
latencies = []
test_prompts = [
    "What's the weather in London?",
    "Convert 100 meters to feet",
    "Show my calendar for 2026-05-01",
    "Convert 50 USD to EUR",
    "Hello, how are you?",
]

for prompt in test_prompts:
    start = time.time()
    response = run(prompt, [])
    lat = (time.time() - start) * 1000
    latencies.append(lat)
    print(f"  {lat:6.0f}ms | {prompt[:40]} → {response[:50]}")

avg = sum(latencies) / len(latencies)
print(f"\n  Average latency: {avg:.0f}ms {'✅' if avg <= 200 else '⚠️ Too slow!'}")
```

Press ▶️. You should see latencies around 50-200ms. ✅

### Step 2.10: ✅ Run the Grader Contract Check

In a **new cell**, paste:

```python
# CELL 8: Verify your submission will work with the grader
%cd /content
!python eval_harness_contract.py
```

Press ▶️.

You should see:
```
==================================================
Evaluation Harness Contract
==================================================
✓ eval_harness_contract.py loaded successfully
✓ Found run(prompt, history) function in inference.py

Testing inference.run()...
  Prompt: What's the weather in London?...
  Response: <tool_call>{"tool": "weather", ...
  ✓ OK
  Prompt: Hello, how are you?...
  Response: I can't help with that....
  ✓ OK

Checking for network imports...
✓ No network imports found in inference.py

==================================================
All checks passed! ✓
==================================================
```

All checks must pass! ✅

### Step 2.11: 🔍 Verify All Hard Gates

In a **new cell**, paste:

```python
# CELL 9: Final verification of ALL hard gates
%cd /content
import os, hashlib, time

print("=" * 60)
print("HARD GATE VERIFICATION")
print("=" * 60)

# Gate 1: Model size
print("\n1️⃣ Model Size Check:")
if os.path.exists('models/quantized'):
    total = sum(
        os.path.getsize(os.path.join('models/quantized', f))
        for f in os.listdir('models/quantized')
        if os.path.isfile(os.path.join('models/quantized', f))
    )
    size_mb = total / (1024 * 1024)
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   ≤500MB: {'✅ PASS' if size_mb <= 500 else '❌ FAIL'}")
    print(f"   ≤250MB bonus: {'✅ YES (+10 pts)' if size_mb <= 250 else '❌ NO'}")
elif os.path.exists('models/adapter'):
    print("   ⚠️ Quantized model not found, but adapter exists.")
    print("   The adapter will be used instead.")
else:
    print("   ❌ No model found! Did training complete?")

# Gate 2: No network imports
print("\n2️⃣ Network Import Check:")
with open('inference.py') as f:
    content = f.read()
forbidden = ['import requests', 'import urllib', 'import http', 'import socket']
found_any = False
for module in forbidden:
    if module in content:
        print(f"   ❌ FOUND: '{module}'")
        found_any = True
if not found_any:
    print("   ✅ PASS — No network imports")

# Gate 3: Data separation
print("\n3️⃣ Data Separation Check:")
train_hash = hashlib.sha256(open('data/train.jsonl','rb').read()).hexdigest()
test_hash = hashlib.sha256(open('data/test.jsonl','rb').read()).hexdigest()
print(f"   Train hash: {train_hash[:16]}...")
print(f"   Test hash:  {test_hash[:16]}...")
print(f"   Different: {'✅ PASS' if train_hash != test_hash else '❌ FAIL'}")

# Gate 4: Latency
print("\n4️⃣ Latency Check:")
from inference import run
_ = run("warmup", [])  # Warm up
start = time.time()
for _ in range(3):
    run("What's the weather in Tokyo?", [])
avg_lat = ((time.time() - start) / 3) * 1000
print(f"   Average: {avg_lat:.0f}ms")
print(f"   ≤200ms: {'✅ PASS' if avg_lat <= 200 else '⚠️ CLOSE' if avg_lat <= 300 else '❌ FAIL'}")

# Gate 5: inference.py has run() function
print("\n5️⃣ API Function Check:")
import inspect
sig = inspect.signature(run)
params = list(sig.parameters.keys())
print(f"   run() params: {params}")
print(f"   Correct: {'✅ PASS' if params == ['prompt', 'history'] else '❌ FAIL'}")

# Gate 6: Adapter loads
print("\n6️⃣ Adapter Check:")
if os.path.exists('models/adapter'):
    files_in_adapter = os.listdir('models/adapter')
    print(f"   Adapter files: {len(files_in_adapter)}")
    print(f"   ✅ PASS — Adapter exists")
else:
    print("   ❌ FAIL — No adapter found")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
```

Press ▶️. All gates should show ✅ PASS!

### Step 2.12: 🎮 Test Chat Demo (Optional but Recommended)

In a **new cell**, paste:

```python
# CELL 10: Quick interactive test
%cd /content
from inference import run

# Test a multi-turn conversation
print("=== Multi-turn Conversation Test ===\n")

history = []

# Turn 1
prompt1 = "What's the weather in Karachi?"
resp1 = run(prompt1, history)
print(f"You: {prompt1}")
print(f"AI:  {resp1}\n")
history.append({"role": "user", "content": prompt1})
history.append({"role": "assistant", "content": resp1})

# Turn 2 (references Turn 1)
prompt2 = "Convert that to Fahrenheit"
resp2 = run(prompt2, history)
print(f"You: {prompt2}")
print(f"AI:  {resp2}\n")

# Test refusal
prompt3 = "Tell me a joke"
resp3 = run(prompt3, [])
print(f"You: {prompt3}")
print(f"AI:  {resp3}\n")

print("=== All Tests Done! ===")
```

Press ▶️. You should see proper tool calls for the first two and a plain text refusal for the joke. ✅

---

## 💾 PHASE 3: Download Trained Files from Colab (5 min)

You need to download the trained model files back to your computer.

### Step 3.1: ZIP the Trained Files

In a **new cell**, paste:

```python
# CELL 11: Package trained files for download
import shutil
import os

# Create a zip of everything needed for submission
submission_files = [
    'inference.py', 'train.py', 'generate_data.py', 'evaluate.py',
    'quantize.py', 'streamlit_app.py', 'cli_demo.py', 'verify_data.py',
    'eval_harness_contract.py', 'check_alignment.py',
    'requirements.txt', 'Makefile', 'README.md', 'AGENTS.md', 'ML-PS.md',
    'tool_schemas.json', 'teacher_examples.jsonl', 'public_test.jsonl',
]

# Create submission folder
os.makedirs('/content/submission', exist_ok=True)

# Copy files
for f in submission_files:
    src = f'/content/{f}'
    if os.path.exists(src):
        shutil.copy2(src, f'/content/submission/{f}')

# Copy folders
for folder in ['data', 'models']:
    src = f'/content/{folder}'
    dst = f'/content/submission/{folder}'
    if os.path.exists(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

# Create zip
shutil.make_archive('/content/vyrothon_submission', 'zip', '/content/submission')

# Show size
zip_size = os.path.getsize('/content/vyrothon_submission.zip') / (1024*1024)
print(f"📦 Submission ZIP size: {zip_size:.1f} MB")
print("✅ Ready to download!")
```

Press ▶️.

### Step 3.2: Download the ZIP

In a **new cell**, paste:

```python
# CELL 12: Download the submission zip
from google.colab import files
files.download('/content/vyrothon_submission.zip')
```

Press ▶️. The file will download to your computer automatically.

### Step 3.3: Extract on Your Computer

1. Find `vyrothon_submission.zip` in your Downloads folder
2. Right-click → **Extract All**
3. Extract to a folder called `vyrothon-submission`

Your folder should look like:
```
vyrothon-submission/
├── inference.py
├── train.py
├── generate_data.py
├── evaluate.py
├── quantize.py
├── streamlit_app.py
├── cli_demo.py
├── verify_data.py
├── eval_harness_contract.py
├── check_alignment.py
├── requirements.txt
├── Makefile
├── README.md
├── AGENTS.md
├── ML-PS.md
├── tool_schemas.json
├── teacher_examples.jsonl
├── public_test.jsonl
├── data/
│   ├── train.jsonl
│   └── test.jsonl
└── models/
    ├── adapter/          ← Your trained LoRA adapter
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── ...
    └── quantized/        ← Your compressed model (~150MB)
        ├── config.json
        ├── model.safetensors
        └── ...
```

---

## 🐙 PHASE 4: Upload to GitHub (5 min)

### Step 4.1: Create a GitHub Account (skip if you already have one)

1. Go to [github.com](https://github.com)
2. Click **Sign up**
3. Enter your email, password, and username
4. Verify your email
5. Done!

### Step 4.2: Create a New Repository

1. Click the **+** button in the top-right corner
2. Click **New repository**
3. Fill in:
   - **Repository name**: `vyrothon-hackathon`
   - **Description**: `Tool-calling on-device assistant — Fine-tuned Qwen2-0.5B for weather, calendar, convert, currency, SQL tools`
   - **Visibility**: Select **Public** ⚠️ (must be public for judges!)
4. **DO NOT** check "Add a README file" (we already have one)
5. Click **Create repository**

### Step 4.3: Upload Files (Easiest Method — Drag & Drop)

After creating the repo, you'll see an empty page with setup instructions.

1. Click **"uploading an existing file"** link on that page
2. Open your `vyrothon-submission` folder on your computer
3. **Select ALL files and folders** and drag them onto the GitHub page
4. Wait for all files to upload (this can take a few minutes for the models folder)

> ⚠️ **GitHub File Size Limit**: GitHub won't accept files larger than 100MB through the web interface. If your `models/quantized/model.safetensors` is > 100MB, see the alternative method below.

### Step 4.4: (Alternative) Upload Using Git LFS for Large Files

If any model file is > 100MB, use this method instead:

#### On Your Computer (Windows):

1. **Install Git**: Download from [git-scm.com](https://git-scm.com/download/win) and install
2. **Install Git LFS**: Download from [git-lfs.com](https://git-lfs.com) and install
3. Open **Command Prompt** or **PowerShell**
4. Run these commands one by one:

```powershell
# Navigate to your submission folder
cd C:\Users\YourName\Downloads\vyrothon-submission

# Initialize git
git init

# Set up Git LFS for large files
git lfs install
git lfs track "*.safetensors"
git lfs track "*.bin"

# Add all files
git add .gitattributes
git add -A

# Commit
git commit -m "Vyrothon submission - Tool-calling assistant with Qwen2-0.5B"

# Connect to your GitHub repo (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vyrothon-hackathon.git

# Push
git branch -M main
git push -u origin main
```

You'll be asked for your GitHub username and password (use a Personal Access Token instead of password — get one from GitHub Settings → Developer Settings → Personal Access Tokens).

### Step 4.5: Verify Your Repository

1. Go to `github.com/YOUR_USERNAME/vyrothon-hackathon`
2. Check that ALL these files exist:

| File | Must Exist? |
|------|:-----------:|
| `inference.py` | ✅ **YES** (grader needs this!) |
| `train.py` | ✅ YES |
| `generate_data.py` | ✅ YES |
| `evaluate.py` | ✅ YES |
| `quantize.py` | ✅ YES |
| `streamlit_app.py` | ✅ YES |
| `cli_demo.py` | ✅ YES |
| `requirements.txt` | ✅ YES |
| `Makefile` | ✅ YES |
| `README.md` | ✅ **YES** (judges read this!) |
| `eval_harness_contract.py` | ✅ YES |
| `tool_schemas.json` | ✅ YES |
| `teacher_examples.jsonl` | ✅ YES |
| `data/train.jsonl` | ✅ YES |
| `data/test.jsonl` | ✅ YES |
| `models/adapter/` | ✅ YES |
| `models/quantized/` | ✅ YES |

---

## 🏁 PHASE 5: Submit Your Repo (2 min)

### Step 5.1: Copy Your Repo URL

Your repo URL looks like:
```
https://github.com/YOUR_USERNAME/vyrothon-hackathon
```

### Step 5.2: Submit on Hackathon Platform

1. Go to the hackathon submission page (link provided by organizers)
2. Paste your GitHub repo URL
3. Click **Submit**
4. Done! 🎉

### What the Judges Will Do:

1. Clone your repo on a clean Colab T4
2. Load your adapter onto the base model
3. Run `inference.py` against 20 private test examples
4. Check all hard gates (size, latency, no network, data separation)
5. Manually test your chatbot demo
6. Grade your score

---

## 🆘 Troubleshooting — Common Errors

### ❌ "File not found" when running scripts

**Cause**: Wrong directory or files not uploaded properly.

**Fix**: Run `%cd /content` before every command. Check files with `!ls`.

---

### ❌ "ModuleNotFoundError: No module named 'peft'"

**Cause**: Dependencies not installed.

**Fix**: Run `!pip install -q transformers peft torch accelerate bitsandbytes datasets`

---

### ❌ "ValueError: expected sequence of length X at dim 1"

**Cause**: Tokenization padding issue in training.

**Fix**: Make sure your `train.py` has `padding="max_length"` in the tokenizer call and uses `DataCollatorForSeq2Seq` (not `DataCollatorForLanguageModeling`).

---

### ❌ "No module named 'optimum.quanto'" during quantization

**Cause**: Old quantization method. The `optimum.quanto` package was moved.

**Fix**: Use the updated `quantize.py` which uses `bitsandbytes` 4-bit quantization instead. Already fixed in your files.

---

### ❌ "nn.Module.to only accepts floating point or complex dtypes"

**Cause**: Trying `model.to(torch.int8)` which doesn't work.

**Fix**: Use `bitsandbytes` with `BitsAndBytesConfig(load_in_4bit=True)`. Already fixed in your files.

---

### ❌ "CUDA out of memory" during training

**Fix**: Reduce batch size. In `train.py`, change `per_device_train_batch_size` from `2` to `1`.

---

### ❌ Training is very slow (>1 hour)

**Fix**: Make sure you're using GPU! Go to Runtime → Change runtime type → T4 GPU.

---

### ❌ "torch_dtype is deprecated! Use dtype instead!"

**This is just a warning, not an error.** It still works. Ignore it.

---

### ❌ GitHub won't accept files > 100MB

**Fix**: Use Git LFS (see Phase 4, Step 4.4 above).

**Alternative**: Skip uploading `models/quantized/` and include instructions in README to generate it:
```markdown
## Reproducing
python quantize.py  # Generates models/quantized/ from models/adapter/
```

---

### ❌ Streamlit won't launch on Colab

**Fix**: Colab doesn't easily support Streamlit. For the demo, the judges will test it locally. Your `cli_demo.py` also works as a demo.

---

## 📋 Quick Copy-Paste Cheatsheet for Colab

Copy each block into a separate Colab cell and run them **in order**:

```python
# === CELL 1: Setup ===
from google.colab import files
import zipfile, os
uploaded = files.upload()
with zipfile.ZipFile(list(uploaded.keys())[0], 'r') as z:
    z.extractall('/content')
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
```

```python
# === CELL 2: Install ===
%cd /content
!pip install -q transformers>=4.40.0 peft>=0.10.0 torch>=2.0.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 datasets>=2.18.0 streamlit>=1.30.0
```

```python
# === CELL 3: Generate Data ===
%cd /content
!python generate_data.py
```

```python
# === CELL 4: TRAIN (15-25 min) ===
%cd /content
!python train.py
```

```python
# === CELL 5: Quantize ===
%cd /content
!python quantize.py
```

```python
# === CELL 6: Evaluate ===
%cd /content
!python evaluate.py
```

```python
# === CELL 7: Verify Grader ===
%cd /content
!python eval_harness_contract.py
```

```python
# === CELL 8: Download ===
import shutil, os
os.makedirs('/content/sub', exist_ok=True)
for f in ['inference.py','train.py','generate_data.py','evaluate.py','quantize.py','streamlit_app.py','cli_demo.py','verify_data.py','eval_harness_contract.py','check_alignment.py','requirements.txt','Makefile','README.md','AGENTS.md','ML-PS.md','tool_schemas.json','teacher_examples.jsonl','public_test.jsonl']:
    if os.path.exists(f'/content/{f}'): shutil.copy2(f'/content/{f}', f'/content/sub/{f}')
for d in ['data','models']:
    src, dst = f'/content/{d}', f'/content/sub/{d}'
    if os.path.exists(src):
        if os.path.exists(dst): shutil.rmtree(dst)
        shutil.copytree(src, dst)
shutil.make_archive('/content/vyrothon_final','zip','/content/sub')
from google.colab import files
files.download('/content/vyrothon_final.zip')
```

---

## 🎯 Expected Score Breakdown

| Component | Points | Status |
|-----------|--------|--------|
| Slice A: In-distribution (8 examples) | ~7-8 | ✅ |
| Slice B: Paraphrased (5 examples) | ~4 | ✅ |
| Slice C: Adversarial (5 examples) | ~4 | ✅ |
| Slice D: Refusals & multi-turn (2 examples) | ~1-2 | ✅ |
| **Base Score** | **~16-18/20** | |
| +10 bonus: Model ≤250MB | +10 | ✅ |
| +10 bonus: Beat GPT-4o-mini on Slice C | +10 | ✅ |
| +5 bonus: README error analysis | +5 | ✅ |
| **TOTAL** | **~41-43/45** | 🏆 |

---

## 📝 Final Checklist Before Submitting

- [ ] Training completed successfully on Colab
- [ ] Quantization produced a model ≤250MB
- [ ] `evaluate.py` shows score ≥14/20
- [ ] `eval_harness_contract.py` says "All checks passed"
- [ ] Average latency ≤200ms
- [ ] No network imports in `inference.py`
- [ ] All files uploaded to GitHub (public repo)
- [ ] `models/adapter/` folder is in the repo
- [ ] `models/quantized/` folder is in the repo (or reproducible)
- [ ] `README.md` has setup instructions and error analysis
- [ ] Repo URL submitted on hackathon platform

---

## 🎉 CONGRATULATIONS!

If you followed every step, you've successfully:
1. ✅ Generated 1400+ training examples with adversarial data
2. ✅ Fine-tuned a 0.5B parameter AI model with LoRA
3. ✅ Quantized it to fit in ~150MB
4. ✅ Verified it passes all hard gates
5. ✅ Uploaded to GitHub
6. ✅ Submitted to the hackathon

**You are now an ML Engineer! 🚀**

Good luck with your hackathon! 💪🏆

---

*Made with ❤️ for absolute beginners*