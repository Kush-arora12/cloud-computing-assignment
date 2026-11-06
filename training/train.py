import os
import sys

# Fast Multi-Core CPU Settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import time
import csv
import json
import yaml
import torch

if not torch.cuda.is_available():
    torch.set_num_threads(4)

import random
import hashlib
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

# Config load
with open("training/config.yaml", "r") as f:
    config = yaml.safe_load(f)

SEED = config.get("seed", 42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument("--subset", type=int, default=None, help="Train on first N records only")
parser.add_argument("--version", type=str, default="v1", help="Model version (v1 or v2)")
args = parser.parse_args()

OUT_DIR = f"models/{args.version}"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("training", exist_ok=True)
LOSS_FILE = "training/loss.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}", flush=True)

# Dataset Load
DATA_PATH = "datasets/v1/train.jsonl"
raw_data = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            raw_data.append(json.loads(line.strip()))

train_records = raw_data[:1500]
if args.subset:
    train_records = train_records[:args.subset]

print(f"Loaded {len(train_records)} records for training ({args.version}).", flush=True)

# Model & Tokenizer (Forced FP32 to prevent FP16 NaN overflow/underflow)
MODEL_NAME = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model = model.float().to(device)
model.train()

# Dataset Class
class DistillationDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.inputs = []
        for r in records:
            t_json = r.get("teacher_raw") or json.dumps(r.get("teacher_json"))
            prompt = f"Command: {r['text']}\nJSON: {t_json}"
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = enc["input_ids"].squeeze(0)
            attn_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            self.inputs.append({
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

train_dataset = DistillationDataset(train_records, tokenizer, config.get("max_seq_length", 64))
train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 16), shuffle=True)

epochs = config.get("epochs", 3)
if args.version == "v2":
    epochs = 4

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, eps=1e-6)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(2, int(total_steps * 0.1)), num_training_steps=total_steps)

epoch_losses = []
print(f"Training started (FP32 Mode, Total Epochs: {epochs}, Batches per epoch: {len(train_loader)})...", flush=True)

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    ep_start = time.time()
    for step, batch in enumerate(train_loader, 1):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        print(f"Epoch {epoch}/{epochs} | Batch {step}/{len(train_loader)} | Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / len(train_loader)
    epoch_losses.append({"epoch": epoch, "loss": round(avg_loss, 4)})
    print(f"--> Epoch {epoch} Done in {time.time() - ep_start:.2f}s | Avg Loss: {avg_loss:.4f}\n", flush=True)

# Save loss.csv
with open(LOSS_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
    writer.writeheader()
    writer.writerows(epoch_losses)
print(f"Losses saved to {LOSS_FILE}", flush=True)

# Save Model
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"Model saved to {OUT_DIR}", flush=True)

# Hash
sha_file = os.path.join(OUT_DIR, "model.sha256")
hasher = hashlib.sha256()
weight_file = os.path.join(OUT_DIR, "model.safetensors")
if not os.path.exists(weight_file):
    weight_file = os.path.join(OUT_DIR, "pytorch_model.bin")

if os.path.exists(weight_file):
    with open(weight_file, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    model_hash = hasher.hexdigest()
    with open(sha_file, "w", encoding="utf-8") as f:
        f.write(f"{model_hash}  {os.path.basename(weight_file)}\n")
    print(f"SHA-256 generated: {model_hash}", flush=True)