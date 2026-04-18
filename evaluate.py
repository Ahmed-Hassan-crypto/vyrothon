import json
import re
import time
from inference import run

TOOL_SCHEMA = {
    "weather": {"location": str, "unit": lambda x: x in ["C", "F"]},
    "calendar": {"action": lambda x: x in ["list", "create"], "date": str},
    "convert": {"value": (int, float), "from_unit": str, "to_unit": str},
    "currency": {"amount": (int, float), "from": str, "to": str},
    "sql": {"query": str}
}

def extract_tool_call(response):
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except:
            pass
    
    json_matches = re.findall(r'\{[^{}]*"tool"\s*:[^{}]*\}', response, re.DOTALL)
    for m in json_matches:
        try:
            parsed = json.loads(m)
            if "tool" in parsed:
                return parsed
        except:
            pass
    
    response_lower = response.lower()
    weather_kw = ["weather", "temperature", "forecast", "rain", "sunny", "climate"]
    calendar_kw = ["calendar", "schedule", "appointment", "meeting", "event"]
    convert_kw = ["convert", "meters", "feet", "celsius", "fahrenheit", "pounds", "kg"]
    currency_kw = ["currency", "dollar", "euro", "pound", "yen", "usd", "eur", "gbp", "pkr"]
    sql_kw = ["sql", "select", "database"]
    
    for kw in weather_kw:
        if kw in response_lower:
            loc_match = re.search(r'(?:in|at|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', response)
            location = loc_match.group(1) if loc_match else "Unknown"
            return {"tool": "weather", "args": {"location": location, "unit": "C"}}
    
    for kw in calendar_kw:
        if any(k in response_lower for k in ["list", "show", "what", "events"]):
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', response)
            date = date_match.group(1) if date_match else "2026-01-01"
            return {"tool": "calendar", "args": {"action": "list", "date": date}}
    
    for kw in convert_kw:
        if kw in response_lower:
            match = re.search(r'(\d+(?:\.\d+)?)\s*(\w+)\s*(?:to|in|->)\s*(\w+)', response, re.I)
            if match:
                return {"tool": "convert", "args": {"value": float(match.group(1)), "from_unit": match.group(2).lower(), "to_unit": match.group(3).lower()}}
    
    for kw in currency_kw:
        if kw in response_lower:
            match = re.search(r'(\d+(?:\.\d+)?)\s*(\w{3})\s*(?:to|in|->)\s*(\w{3})', response, re.I)
            if match:
                return {"tool": "currency", "args": {"amount": float(match.group(1)), "from": match.group(2).upper(), "to": match.group(3).upper()}}
    
    for kw in sql_kw:
        if kw in response_lower:
            return {"tool": "sql", "args": {"query": "SELECT * FROM users"}}
    
    return None

def score_example(prompt, expected_tool, expected_args, response):
    parsed = extract_tool_call(response)
    
    if expected_tool is None:
        if parsed is None:
            return 1.0, "Correct refusal"
        return -0.5, "Wrong: made tool call when should refuse"
    
    if parsed is None:
        return 0.0, "No tool call parsed"
    
    if parsed.get("tool") != expected_tool:
        return 0.0, f"Wrong tool: {parsed.get('tool')} != {expected_tool}"
    
    args = parsed.get("args", {})
    score = 1.0
    reasons = []
    
    for key, expected_val in expected_args.items():
        if key not in args:
            score = 0.5
            reasons.append(f"Missing: {key}")
        elif isinstance(expected_val, (int, float)) and isinstance(args[key], (int, float)):
            if abs(args[key] - expected_val) > max(abs(expected_val) * 0.01, 1):
                score = 0.5
                reasons.append(f"Wrong {key}")
        elif str(args[key]).lower() != str(expected_val).lower():
            score = 0.5
            reasons.append(f"Wrong {key}")
    
    return score, "; ".join(reasons) if reasons else "Perfect"

def evaluate(test_file):
    with open(test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    
    results = []
    total_score = 0
    
    for item in test_data:
        prompt = item["prompt"]
        expected_tool = item.get("tool")
        expected_args = item.get("args", {})
        history = item.get("history", [])
        
        start_time = time.time()
        response = run(prompt, history)
        latency = (time.time() - start_time) * 1000
        
        score, reason = score_example(prompt, expected_tool, expected_args, response)
        
        results.append({
            "prompt": prompt[:50],
            "expected_tool": expected_tool,
            "score": score,
            "latency_ms": latency,
            "reason": reason
        })
        
        total_score += score
    
    avg_score = total_score / len(test_data)
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    
    print(f"="*50)
    print(f"Evaluation Results")
    print(f"="*50)
    print(f"Total examples: {len(results)}")
    print(f"Average score: {avg_score:.2f}")
    print(f"Average latency: {avg_latency:.1f}ms")
    print(f"Total score: {total_score:.1f}/20")
    print()
    
    for i, r in enumerate(results):
        print(f"{i+1:2d}. Score: {r['score']:+.1f} | Latency: {r['latency_ms']:5.0f}ms | {r['reason']}")
    
    return results

if __name__ == "__main__":
    evaluate("data/test.jsonl")