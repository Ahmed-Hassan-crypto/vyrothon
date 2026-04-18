# Step-by-Step Guide to Run Vyrothon on Google Colab

## Phase 1: Prepare Your Files (On Your Computer)

### Step 1: Create a ZIP File
1. Go to the `Vyrothon` folder on your computer
2. Select all these files/folders:
   - `inference.py`
   - `train.py`
   - `generate_data.py`
   - `evaluate.py`
   - `quantize.py`
   - `streamlit_app.py`
   - `requirements.txt`
   - `Makefile`
   - `README.md`
   - `AGENTS.md`
   - `ML-PS.md`
   - `colab_runner.ipynb`
   - `data/` folder (contains train.jsonl and test.jsonl)
3. Right-click → Send to → Compressed (zip)
4. Name it `vyrothon.zip`

---

## Phase 2: Upload to Google Colab

### Step 2: Open Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**

### Step 3: Upload the ZIP File
1. Click the **Files** icon (left sidebar) → click **Upload**
2. Select your `vyrothon.zip` file
3. Wait for upload to complete

### Step 4: Unzip the Files
In a code cell, type:
```python
import zipfile
import os

# Unzip the file
with zipfile.ZipFile('vyrothon.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

# Verify files
for root, dirs, files in os.walk('.'):
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f'{sub_indent}{file}')
    if len(files) > 5:
        print(f'{sub_indent}... and {len(files)-5} more files')
```

---

## Phase 3: Run the Project

### Step 5: Install Dependencies
In a new code cell:
```python
!pip install -r requirements.txt
```
⏱️ **Time: 2-3 minutes**

### Step 6: Generate Training Data
```python
!python generate_data.py
```
⏱️ **Time: 30 seconds**
- Creates ~1400 training examples
- Output should show: "Training: 1398 examples, Refusals: 296 (21.2%)"

### Step 7: Train the Model (IMPORTANT!)
```python
!python train.py
```
⏱️ **Time: 20-30 minutes** (on Colab T4 GPU)

Expected output:
```
Loading tokenizer and model...
Loading base model...
Configuring LoRA...
Preparing dataset...
Dataset size: 1398
Starting training...
Trainable params: 2,097,152 || All params: 494,387,200 || Trainable%: 0.42
...
Training complete!
Saving adapter to models/adapter/...
```

### Step 8: Quantize the Model
```python
!python quantize.py
```
⏱️ **Time: 3-5 minutes**

Expected output:
```
Loading base model...
Loading and merging adapter...
Quantizing to 4-bit...
Quantization complete
Saving to models/quantized/...
Quantized model size: ~150 MB
SUCCESS: Within 250MB bonus threshold!
```

### Step 9: Verify Hard Gates
```python
import os

# Check model size
total_size = sum(os.path.getsize(os.path.join('models/quantized', f)) 
                 for f in os.listdir('models/quantized'))
print(f"Model size: {total_size/(1024*1024):.1f} MB")

# Check no network imports
with open("inference.py") as f:
    content = f.read()
forbidden = ['requests', 'urllib', 'http', 'socket']
has_bad = any(x in content for x in forbidden)
print(f"Network imports: {'FOUND (BAD!)' if has_bad else 'None (Good!)'}")

# Check latency
import time
from inference import run
start = time.time()
run("weather test", [])
latency = (time.time()-start)*1000
print(f"Latency: {latency:.0f}ms")
```

### Step 10: Run Evaluation
```python
!python evaluate.py
```
⏱️ **Time: 1-2 minutes**

Expected output:
```
==================================================
Evaluation Results
==================================================
Total examples: 20
Average score: ~17
Average latency: ~150ms
Total score: 17.0/20
```

### Step 11: Launch Streamlit Demo

```python
!streamlit run streamlit_app.py --server.port 8501
```

This will show a link to open the Streamlit web interface.

---

## Phase 4: Score Summary

| Component | Points |
|-----------|--------|
| Base Score | ~17/20 |
| +10 (Model ≤250MB) | +10 |
| +10 (Slice C) | +10 |
| +5 (Error Analysis) | +5 |
| **TOTAL** | **~42/45** |

---

## Quick Commands Summary

Copy-paste this into Colab cells in order:

```python
# Cell 1: Upload
from google.colab import files
files.upload()

# Cell 2: Unzip
import zipfile, os
with zipfile.ZipFile('vyrothon.zip','r') as z: z.extractall('.')

# Cell 3: Install
!pip install -r requirements.txt

# Cell 4-9: Run pipeline
!python generate_data.py
!python train.py
!python quantize.py
!python evaluate.py

# Cell 10: Demo
!streamlit run streamlit_app.py --server.port 8501
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce batch_size in train.py |
| Model not loading | Check adapter path in models/adapter |
| Latency too high | Reduce max_new_tokens in inference.py |
| Import errors | Reinstall requirements.txt |

---

**Good luck! 🚀**