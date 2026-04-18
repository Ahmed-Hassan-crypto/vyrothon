import hashlib
import json

def compute_file_hash(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def verify_no_overlap():
    train_hash = compute_file_hash("data/train.jsonl")
    test_hash = compute_file_hash("data/test.jsonl")
    
    print(f"Train file SHA-256: {train_hash[:16]}...")
    print(f"Test file SHA-256: {test_hash[:16]}...")
    
    with open("data/train.jsonl", "r", encoding="utf-8") as f:
        train_prompts = set(json.loads(line)["prompt"] for line in f)
    
    with open("data/test.jsonl", "r", encoding="utf-8") as f:
        test_prompts = set(json.loads(line)["prompt"] for line in f)
    
    overlap = train_prompts & test_prompts
    
    if overlap:
        print(f"\nWARNING: Found {len(overlap)} overlapping prompts!")
    else:
        print("\nOK: No overlapping prompts between train and test sets")
    
    return len(overlap) == 0

if __name__ == "__main__":
    verify_no_overlap()