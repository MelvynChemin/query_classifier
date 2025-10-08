import json
import string
from sklearn.model_selection import train_test_split

def normalize(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation))

# Load from JSONL (one per line)
data = []
with open('data/combined_data.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        item['query'] = normalize(item['query'])  # Normalize
        data.append(item)

# Split 80/10/10 (random, or customize below if public for val/test)
train, temp = train_test_split(data, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Optional Custom Split: If you want synthetic as train, public as val/test
# (Tag 'source' in data if available, then filter: train = [d for d in data if d.get('source') == 'synthetic'], etc.)

# Save as JSON arrays
with open('data/train.json', 'w') as f: json.dump(train, f)
with open('data/val.json', 'w') as f: json.dump(val, f)
with open('data/test.json', 'w') as f: json.dump(test, f)

print(f"Preprocessed: {len(train)} train, {len(val)} val, {len(test)} test.")