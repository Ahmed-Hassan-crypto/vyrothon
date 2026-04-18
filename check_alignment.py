import json
import os

BASE_DIR = "C:/Vyrothon"

# Check tool schemas
with open(os.path.join(BASE_DIR, 'tool_schemas.json'), encoding='utf-8') as f:
    schemas = json.load(f)
print('=== tool_schemas.json ===')
for tool in schemas['tools']:
    print(f"  - {tool['name']}: {len(tool['parameters']['properties'])} args")

# Check train data tools
with open(os.path.join(BASE_DIR, 'data/train.jsonl'), encoding='utf-8') as f:
    train = [json.loads(line) for line in f]
tools_used = {}
for item in train:
    if item.get('tool'):
        t = item['tool']
        tools_used[t] = tools_used.get(t, 0) + 1

print('\n=== Tools in training data ===')
for t, count in sorted(tools_used.items()):
    print(f"  - {t}: {count} examples")

# Check public test
with open(os.path.join(BASE_DIR, 'public_test.jsonl'), encoding='utf-8') as f:
    public = [json.loads(line) for line in f]
print('\n=== public_test.jsonl ===')
print(f"  Total: {len(public)} examples")

tools_test = {}
refusals = 0
for item in public:
    if item.get('tool'):
        t = item['tool']
        tools_test[t] = tools_test.get(t, 0) + 1
    if item.get('is_refusal'):
        refusals += 1

print(f"  Tool calls: {len(public) - refusals}")
print(f"  Refusals: {refusals}")

# Check eval_harness
print('\n=== eval_harness_contract.py ===')
with open(os.path.join(BASE_DIR, 'eval_harness_contract.py'), encoding='utf-8') as f:
    content = f.read()
    if 'from inference import run' in content:
        print("  OK Imports inference.run")
    if 'def run(' in content:
        print("  OK Has run function")

print('\n=== ALIGNMENT CHECK ===')
schema_tools = set(t['name'] for t in schemas['tools'])
train_tools = set(tools_used.keys())
test_tools = set(tools_test.keys())

print(f"  Schema tools: {schema_tools}")
print(f"  Train tools: {train_tools}")
print(f"  Test tools: {test_tools}")

missing = schema_tools - train_tools
extra = train_tools - schema_tools

if not missing and not extra:
    print('  OK PERFECT ALIGNMENT - All tools match!')
else:
    if missing:
        print(f"  MISSING in train: {missing}")
    if extra:
        print(f"  EXTRA in train: {extra}")

# Check teacher examples
print('\n=== teacher_examples.jsonl ===')
with open(os.path.join(BASE_DIR, 'teacher_examples.jsonl'), encoding='utf-8') as f:
    teacher = [json.loads(line) for line in f]
print(f"  Total: {len(teacher)} examples")
t_tools = {}
t_refuse = 0
for item in teacher:
    if item.get('tool'):
        t_tools[item['tool']] = t_tools.get(item['tool'], 0) + 1
    if item.get('is_refusal'):
        t_refuse += 1
print(f"  Tool calls: {len(teacher) - t_refuse}, Refusals: {t_refuse}")