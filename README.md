# Query Classifier for RAG Pipelines

[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/MelvynCHEMIN/query_classifier)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

An **intelligent query classifier** designed to optimize Retrieval-Augmented Generation (RAG) pipelines. Acts as a gatekeeper that analyzes user queries to determine intent, ambiguity, safety, and optimal routing strategy—ensuring efficiency, accuracy, and built-in safeguards.

Inspired by modular RAG frameworks (LangChain, Haystack) and research on query decomposition, this classifier is built for **self-hosted setups** using FastAPI, with training pipelines ready for cloud platforms.

---

## 🎯 Goals

- **Efficiency**: Skip retrieval when not needed (e.g., chit-chat, general knowledge)
- **Accuracy**: Route queries to the right retrieval strategy (vanilla, fusion, sub-questions, etc.)
- **Safety**: Detect unsafe or adversarial queries early
- **Observability**: Provide structured metadata for debugging and monitoring
- **Adaptability**: Reusable across projects as a lightweight, open-source module

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/MelvynChemin/query_classifier.git
cd query_classifier
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Example Classification

```python
from transformers import pipeline

pipe = pipeline("text-classification", model="MelvynCHEMIN/query_classifier")
print(pipe("How does retrieval augmentation improve accuracy?"))
```

### 5. Launch FastAPI Server

```bash
uvicorn api:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Model Availability

The trained model is publicly available on **Hugging Face**:

👉 [MelvynCHEMIN/query_classifier](https://huggingface.co/MelvynCHEMIN/query_classifier)

Load it easily with:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("MelvynCHEMIN/query_classifier")
tokenizer = AutoTokenizer.from_pretrained("MelvynCHEMIN/query_classifier")
```

> **🧠 Note**: This is an experimental project—the model was trained locally, and evaluations are ongoing. Upcoming work includes systematic benchmarking and expanded datasets.

---

## 🔄 Pipeline Flow

```
User Query → Normalizer → Query Classifier → Router
├─ LLM only (if no retrieval needed)
├─ Block (if unsafe)
├─ Retrieval strategy (vanilla, fusion, subq, stepback)
    ↓
Retriever → LLM → Final Response
```

---

## 🏗️ Architecture

### Intent Classes

- **`chitchat`**: Casual conversation, no retrieval needed
- **`general_knowledge`**: Factual queries, LLM can answer directly
- **`internal_knowledge`**: Requires retrieval from knowledge base
- **`unsafe`**: Adversarial, abusive, or prompt injection attempts

### Scores & Flags

- `ambiguity_score` (0–1)
- `is_overview` (0–1)
- `is_compositional` (0–1)
- `is_alias_or_underspecified` (0–1)

### Confidence Metrics

- `intent_confidence`: Softmax probability

### Routing Hints

Deterministic rules map outputs to strategies:
- `block`
- `llm_only`
- `vanilla`
- `fusion`
- `subq`
- `stepback`

### Example JSON Output

```json
{
  "intent": "internal_knowledge",
  "intent_confidence": 0.95,
  "ambiguity_score": 0.72,
  "is_overview": 0.1,
  "is_compositional": 0.0,
  "is_alias_or_underspecified": 0.9,
  "route_hint": "fusion"
}
```

---

## 🔧 Technical Details

- **Model**: Transformer encoder (DistilBERT / DeBERTa / MiniLM, ~50–100M params)
- **Multi-task heads**: Classification + regression for scores/flags
- **Loss**: Cross-entropy (intent) + MSE (scores)
- **Training Data**: 5k–20k labeled queries (chitchat, factual, unsafe, multi-hop)
- **Training**: Hugging Face Transformers on GPU (Google Cloud, Alibaba, or local)
- **Inference**: <50ms per query

---

## 🎓 Training & Evaluation

1. **Preprocess**: Normalize and tokenize queries
2. **Train**: Multi-task model with class weighting to prioritize internal knowledge recall
3. **Evaluate**:
   - Intent → Accuracy & F1-score
   - Scores → MAE
   - Pipeline-level metrics → retrieval hit rate, token savings

> **Note**: Model training and dataset management are done within this GitHub repository. Resulting model weights are hosted on Hugging Face for public access.

> **📊 Quantitative evaluation results coming soon** as experiments stabilize.

---

## 🚢 Deployment

### FastAPI Endpoint

```python
@app.post("/classify")
def classify(query: str):
    # Returns JSON with intent, scores, route_hint
    pass
```

- Self-host with Docker or cloud instances
- Monitor with logs/Prometheus
- Active learning loop: Re-train with misclassified or low-confidence queries

---

## 🛠️ Key Components

### `check_query.py`

Validates training query formatting.

```bash
# Validate a single file
python validate_query_jsonl.py /path/to/query_gpt.jsonl

# Output example:
# [PASS] /home/melvyn/Projets/query_classifier/data/query_gpt.jsonl
#   lines: 135
#   intents:
#     - chitchat: 14
#     - general_knowledge: 12
#     - internal_knowledge: 99
#     - unsafe: 10

# Validate all .jsonl files in a folder
python validate_query_jsonl.py /path/to/folder

# Strict mode with error cap
python validate_query_jsonl.py /path/to/folder --strict --max-errors 50
```

---

## 🔮 Future Extensions

- [ ] Add conversation history as context
- [ ] Fine-tune larger models (e.g., LLaMA with PEFT)
- [ ] Integrate external safety tools (OpenAI moderation, HF toxicity models)
- [ ] Package as PyPI module with pre-trained weights
- [ ] Systematic benchmarking suite
- [ ] Expanded multilingual support

---

## 🏷️ Labeling Process

Combination of:
- Manual labeling with **Label Studio**
- LLM-assisted labeling for efficiency

---

## ✨ Benefits

✅ Modular and adaptable for RAG frameworks  
✅ Lightweight and fast inference (<50ms)  
✅ Transparent outputs for debugging and logging  
✅ Ready for fine-tuning or transfer learning  
✅ Open-source and community-friendly  

---

## 📄 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Inspired by **LangChain**, **Haystack**, and the **Hugging Face** ecosystem.  
Special thanks to open-source contributors improving RAG research and tooling.

---

## 📬 Contact & Contributions

Contributions, issues, and feature requests are welcome!

- **GitHub**: [MelvynChemin/query_classifier](https://github.com/MelvynChemin/query_classifier)
- **Hugging Face**: [MelvynCHEMIN/query_classifier](https://huggingface.co/MelvynCHEMIN/query_classifier)

---

**Made with ❤️ for the RAG community**