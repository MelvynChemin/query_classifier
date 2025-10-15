# run in a small Python shell inside your venv
from transformers import AutoTokenizer
import json, os

BASE = "distilbert-base-uncased"   # the base you trained from
OUT = "models/best"

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.save_pretrained(OUT)   # writes tokenizer.json, tokenizer_config.json, vocab files

id2label = {0:'chitchat', 1:'general_knowledge', 2:'internal_knowledge', 3:'unsafe'}
with open(os.path.join(OUT, "labels.json"), "w") as f:
    json.dump({"id2label": id2label, "label2id": {v:k for k,v in id2label.items()}}, f, indent=2)
print("✅ wrote tokenizer + labels.json")
