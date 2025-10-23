# # run in a small Python shell inside your venv
# from transformers import AutoTokenizer
# import json, os

# BASE = "distilbert-base-uncased"   # the base you trained from
# OUT = "models/best"

# tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
# tok.save_pretrained(OUT)   # writes tokenizer.json, tokenizer_config.json, vocab files

# id2label = {0:'chitchat', 1:'general_knowledge', 2:'internal_knowledge', 3:'unsafe'}
# with open(os.path.join(OUT, "labels.json"), "w") as f:
#     json.dump({"id2label": id2label, "label2id": {v:k for k,v in id2label.items()}}, f, indent=2)
# print("✅ wrote tokenizer + labels.json")

# import torch
# from pathlib import Path

# checkpoint = torch.load("models/best/pytorch_model.bin", map_location="cpu")
# print("Keys in checkpoint:")
# for key in checkpoint.keys():
#     print(f"  {key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}")


# from classifier import QueryClassifier

# model = QueryClassifier()
# print("\nKeys in model:")
# for name, param in model.named_parameters():
#     print(f"  {name}: {param.shape}")


# tt.py
import torch
from pathlib import Path
from safetensors.torch import load_file

# Load the checkpoint
checkpoint = load_file("models/best/model.safetensors")

print("Keys in checkpoint:")
for key in list(checkpoint.keys())[:10]:  # First 10 keys
    print(f"  {key}: {checkpoint[key].shape}")
print(f"  ... ({len(checkpoint)} total keys)")

# Check for your custom heads
print("\nCustom regression heads in checkpoint:")
for key in checkpoint.keys():
    if 'regression_heads' in key:
        print(f"  {key}: {checkpoint[key].shape}")