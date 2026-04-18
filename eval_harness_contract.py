"""
Evaluation Harness Contract
This is the exact interface the grader will call.

The grader will import this module and call:
    inference.run(prompt, history)
    
Where:
    prompt: str - The user's current message
    history: List[dict] - Previous turns [{"role": "user|assistant", "content": str}]
    
Returns: str - Model response (tool call in <tool_call> tags or plain text)
"""

import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the inference module
try:
    from inference import run
except ImportError as e:
    print(f"ERROR: Could not import inference module: {e}")
    print("Make sure inference.py has a run(prompt, history) function")
    sys.exit(1)

# Verify the run function exists and has correct signature
if not callable(run):
    print("ERROR: 'run' is not a callable function in inference.py")
    sys.exit(1)

import inspect
sig = inspect.signature(run)
params = list(sig.parameters.keys())
if params != ['prompt', 'history']:
    print(f"ERROR: run() should have parameters (prompt, history), got: {params}")
    sys.exit(1)

print("✓ eval_harness_contract.py loaded successfully")
print(f"✓ Found run(prompt, history) function in inference.py")

# The grader will call these functions during evaluation
# We provide helper functions for the grader to use

def get_score(prompt: str, expected_tool: str, expected_args: dict, history: list = None) -> float:
    """
    Get score for a single example.
    
    Args:
        prompt: User's message
        expected_tool: Expected tool name or None for refusal
        expected_args: Expected tool arguments
        history: Conversation history
    
    Returns:
        float: Score (+1.0, +0.5, 0.0, or -0.5)
    """
    response = run(prompt, history or [])
    
    # Parse the response to extract tool call
    tool_call = parse_tool_call(response)
    
    # Case 1: Expected refusal
    if expected_tool is None:
        if tool_call is None:
            return 1.0  # Correct refusal
        else:
            return -0.5  # Wrong tool call
    
    # Case 2: Expected tool call
    if tool_call is None:
        return 0.0  # No tool call parsed
    
    # Check tool name
    if tool_call.get("tool") != expected_tool:
        return 0.0  # Wrong tool
    
    # Check arguments
    args = tool_call.get("args", {})
    score = 1.0
    
    for key, expected_val in expected_args.items():
        if key not in args:
            score = 0.5  # Missing arg
        elif isinstance(expected_val, (int, float)) and isinstance(args[key], (int, float)):
            if abs(args[key] - expected_val) > max(abs(expected_val) * 0.01, 1):
                score = 0.5  # Wrong value
        elif str(args[key]).lower() != str(expected_val).lower():
            score = 0.5  # Wrong value
    
    return score

def parse_tool_call(response: str) -> dict | None:
    """Parse tool call from model response."""
    import re
    
    # Try <tool_call> tags
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    # Try any JSON with "tool" key
    json_matches = re.findall(r'\{[^{}]*"tool"\s*:[^{}]*\}', response, re.DOTALL)
    for m in json_matches:
        try:
            parsed = json.loads(m)
            if "tool" in parsed:
                return parsed
        except:
            pass
    
    return None

def verify_no_network_imports() -> bool:
    """Verify inference.py has no network imports."""
    with open("inference.py", "r") as f:
        content = f.read()
    
    forbidden = ["requests", "urllib", "http.client", "socket"]
    for module in forbidden:
        if module in content:
            print(f"ERROR: Found '{module}' in inference.py")
            return False
    
    print("✓ No network imports found in inference.py")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Evaluation Harness Contract")
    print("=" * 50)
    
    # Test the run function
    print("\nTesting inference.run()...")
    
    test_cases = [
        ("What's the weather in London?", []),
        ("Hello, how are you?", []),
    ]
    
    for prompt, history in test_cases:
        try:
            response = run(prompt, history)
            print(f"  Prompt: {prompt[:40]}...")
            print(f"  Response: {response[:60]}...")
            print("  ✓ OK")
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            sys.exit(1)
    
    # Verify no network imports
    print("\nChecking for network imports...")
    verify_no_network_imports()
    
    print("\n" + "=" * 50)
    print("All checks passed! ✓")
    print("=" * 50)