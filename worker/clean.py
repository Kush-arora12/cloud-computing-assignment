import os
import glob
import json
import hashlib

ALLOWED_INTENTS = {
    "alarm_set", "alarm_query",
    "calendar_set", "calendar_query", "calendar_remove",
    "weather_query", "datetime_query", "recommendation_events"
}

ALLOWED_KEYS = {"intent", "date", "time", "place"}

SHARDS_DIR = "datasets/teacher"
OUTPUT_DIR = "datasets/v1"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")
SHA_FILE = os.path.join(OUTPUT_DIR, "train.sha256")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Shards read karo
all_records = []
shard_files = sorted(glob.glob(os.path.join(SHARDS_DIR, "shard-*.jsonl")))

print(f"Reading from shard files: {shard_files}")
for s_file in shard_files:
    with open(s_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line.strip()))

# Sort strictly by ID order (e.g. C0001, C0002, ...)
all_records.sort(key=lambda x: x["id"])

total_input = len(all_records)
rejected_a = 0  # not valid JSON
rejected_b = 0  # invalid keys
rejected_c = 0  # invalid intent
rejected_d = 0  # invalid slot values
rejected_e = 0  # duplicate commands
kept_records = []

seen_texts = {}  # text -> smallest id

for item in all_records:
    cmd_id = item["id"]
    text = item["text"]
    raw_reply = item.get("teacher_raw", "").strip()

    # Rule (a): Valid JSON check
    try:
        parsed_json = json.loads(raw_reply)
        if not isinstance(parsed_json, dict):
            rejected_a += 1
            continue
    except Exception:
        rejected_a += 1
        continue

    # Rule (b): Exact keys check
    if set(parsed_json.keys()) != ALLOWED_KEYS:
        rejected_b += 1
        continue

    # Rule (c): Allowed intent check
    if parsed_json["intent"] not in ALLOWED_INTENTS:
        rejected_c += 1
        continue

    # Rule (d): Slot values must be null or verbatim substring of text (case-insensitive check)
    slots_valid = True
    text_lower = text.lower()
    for slot in ["date", "time", "place"]:
        val = parsed_json[slot]
        if val is not None:
            if not isinstance(val, str) or val.lower() not in text_lower:
                slots_valid = False
                break
    if not slots_valid:
        rejected_d += 1
        continue

    # Rule (e): Duplicate command check (keep smallest id)
    if text_lower in seen_texts:
        rejected_e += 1
        continue
    else:
        seen_texts[text_lower] = cmd_id

    # Cleaned record
    clean_record = {
        "id": cmd_id,
        "text": text,
        "intent": parsed_json["intent"],
        "date": parsed_json["date"],
        "time": parsed_json["time"],
        "place": parsed_json["place"],
        "teacher_json": parsed_json
    }
    kept_records.append(clean_record)

# 2. datasets/v1/train.jsonl write karo
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for rec in kept_records:
        f.write(json.dumps(rec) + "\n")

# 3. SHA-256 compute karo
hasher = hashlib.sha256()
with open(OUTPUT_FILE, "rb") as f:
    while chunk := f.read(8192):
        hasher.update(chunk)
sha256_hash = hasher.hexdigest()

with open(SHA_FILE, "w", encoding="utf-8") as f:
    f.write(f"{sha256_hash}  train.jsonl\n")

# 4. Summary Table print karo (Report me exact yeh table banegi)
print("\n" + "="*50)
print("DATA CLEANING SUMMARY REPORT (Task 1)")
print("="*50)
print(f"Total Input Records       : {total_input}")
print(f"Rejected (a) Invalid JSON : {rejected_a}")
print(f"Rejected (b) Wrong Keys   : {rejected_b}")
print(f"Rejected (c) Wrong Intent : {rejected_c}")
print(f"Rejected (d) Invalid Slot : {rejected_d}")
print(f"Rejected (e) Duplicates   : {rejected_e}")
print(f"Records Kept              : {len(kept_records)}")
print("-" * 50)
sum_check = rejected_a + rejected_b + rejected_c + rejected_d + rejected_e + len(kept_records)
print(f"Equation check: {total_input} == {sum_check} -> {'MATCH! (Valid)' if total_input == sum_check else 'MISMATCH!'}")
print(f"SHA-256 Checksum: {sha256_hash}")
print(f"Cleaned dataset saved at: {OUTPUT_FILE}")
print("="*50)