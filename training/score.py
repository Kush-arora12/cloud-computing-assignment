import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def evaluate_models():
    val_path = "datasets/v1/validation.jsonl"
    if not os.path.exists(val_path):
        print(f"Validation dataset {val_path} not found. Skipping evaluation score generation.")
        return

    # Load validation records
    val_records = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_records.append(json.loads(line.strip()))

    print(f"Loaded {len(val_records)} validation samples.")

    # Evaluate trained student model (v1)
    model_dir = "models/v1"
    if os.path.exists(model_dir):
        print("Evaluating Trained Student Model (v1)...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        model.eval()

        correct = 0
        total = min(50, len(val_records)) # Quick evaluation subset
        
        for r in val_records[:total]:
            prompt = f"Command: {r['text']}\nJSON:"
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
            
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Basic validation check (checking if intent matches)
            if r.get("intent") in decoded:
                correct += 1

        accuracy = (correct / total) * 100
        print(f"Student Model (v1) Validation Accuracy: {accuracy:.2f}%")
    else:
        print("Model v1 not found for scoring.")

if __name__ == "__main__":
    evaluate_models()