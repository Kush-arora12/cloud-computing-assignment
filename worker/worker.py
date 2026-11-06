import os
import json
import time
import requests

SHARD_INDEX = int(os.environ.get("SHARD_INDEX", 0))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", 1))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
INPUT_FILE = os.environ.get("INPUT_FILE", "data/commands_pool.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "datasets/teacher")

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, f"shard-{SHARD_INDEX}.jsonl")

# 1. Check karo pehle se kitne process ho chuke hain (Resume Support)
already_processed_ids = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line.strip())
                    already_processed_ids.add(data["id"])
                except Exception:
                    pass

print("=" * 55)
print(f"Worker {SHARD_INDEX} Started | Total Shards: {SHARD_COUNT}")
print(f"Already completed in this shard: {len(already_processed_ids)} commands")
print(f"Output target: {output_file}")
print("=" * 55)

# 2. Teacher prompt load karo
with open("prompts/teacher_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

count_new = 0

with open(INPUT_FILE, "r", encoding="utf-8") as in_f, open(output_file, "a", encoding="utf-8") as out_f:
    for line_idx, line in enumerate(in_f):
        # Strict Sharding Check
        if line_idx % SHARD_COUNT != SHARD_INDEX:
            continue

        record = json.loads(line.strip())
        cmd_id = record["id"]
        cmd_text = record["text"]

        # Agar pehle se processed hai, toh skip karo (Resume)
        if cmd_id in already_processed_ids:
            continue

        payload = {
            "model": "phi3.5",
            "prompt": f"{system_prompt}\n\nCommand: {cmd_text}\nOutput:",
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "stop": ["\n\n", "Note:", "Command:"]
            }
        }

        start_time = time.time()
        print(f"[Shard {SHARD_INDEX}] Processing {cmd_id}: \"{cmd_text[:30]}...\"", end="", flush=True)

        try:
            res = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=90)
            raw_reply = res.json().get("response", "").strip()
            elapsed = round(time.time() - start_time, 2)
            print(f" -> Done ({elapsed}s)")
        except Exception as e:
            print(f" -> Failed: {e}")
            raw_reply = ""

        out_record = {
            "id": cmd_id,
            "text": cmd_text,
            "teacher_raw": raw_reply
        }
        out_f.write(json.dumps(out_record) + "\n")
        out_f.flush()

        count_new += 1

print(f"\nWorker {SHARD_INDEX} Completed! Newly processed: {count_new}")