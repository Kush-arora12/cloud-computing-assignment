import json
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "models/v1"
INPUT_FILE = "data/validation.jsonl"
OUTPUT_DIR = "results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "predictions.jsonl")

REQUIRED_KEYS = {"intent", "date", "time", "place"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_valid_json(text: str) -> str:
    """Extract the first valid 4-key JSON object from raw model output."""
    text = text.strip()
    
    # 1. Direct parse test
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and set(obj.keys()) == REQUIRED_KEYS:
            return json.dumps(obj)
    except Exception:
        pass

    # 2. Extract balanced JSON block from first '{'
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[start_idx : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and set(obj.keys()) == REQUIRED_KEYS:
                            return json.dumps(obj)
                    except Exception:
                        pass

    # 3. Handle unclosed repetition (like V091) by extracting available keys
    intent_match = re.search(r'"intent":\s*"([^"]+)"', text)
    date_match = re.search(r'"date":\s*("([^"]+)"|null)', text)
    time_match = re.search(r'"time":\s*("([^"]+)"|null)', text)
    place_match = re.search(r'"place":\s*("([^"]+)"|null)', text)

    fallback = {
        "intent": intent_match.group(1) if intent_match else "unknown",
        "date": json.loads(date_match.group(1)) if date_match else None,
        "time": json.loads(time_match.group(1)) if time_match else None,
        "place": json.loads(place_match.group(1)) if place_match else None,
    }
    return json.dumps(fallback)


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
model.eval()

predictions = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = [json.loads(line.strip()) for line in f if line.strip()]

print(f"Running inference on {len(lines)} validation samples...")
for item in lines:
    prompt = f"Command: {item['text']}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated_tokens = outputs[0][input_len:]
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    clean_json_str = extract_valid_json(raw_text)

    predictions.append({
        "id": item["id"],
        "prediction": clean_json_str
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pred in predictions:
        f.write(json.dumps(pred) + "\n")

print(f"Predictions saved successfully to {OUTPUT_FILE}")