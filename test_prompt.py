import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

with open("prompts/teacher_prompt.txt", "r") as f:
    system_prompt = f.read()

test_command = "set an alarm for seven thirty am tomorrow"

payload = {
    "model": "phi3.5",
    "prompt": f"{system_prompt}\n\nCommand: {test_command}\nOutput:",
    "stream": False,
    "options": {
        "temperature": 0,
        "seed": 42,
        "stop": ["\n\n", "Note:", "Command:"]
    }
}

res = requests.post(OLLAMA_URL, json=payload)
raw_text = res.json().get("response", "").strip()

print("--- Raw Output ---")
print(raw_text)

print("\n--- Validation Check ---")
try:
    data = json.loads(raw_text)
    print("Parsed JSON successfully:", data)
    print("Keys valid:", list(data.keys()) == ["intent", "date", "time", "place"])
except Exception as e:
    print("JSON Parsing Failed:", e)