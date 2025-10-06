# Query Classifier for RAG Pipelines

## Overview
This project provides a **Query Classifier** designed to optimize Retrieval-Augmented Generation (RAG) pipelines.  
It acts as an intelligent gatekeeper, analyzing user queries to determine intent, ambiguity, safety, and routing strategy—ensuring efficiency, accuracy, and built-in safeguards.

Inspired by modular RAG frameworks (e.g., LangChain, Haystack) and research on query decomposition, this classifier is built for **self-hosted setups** using FastAPI, with training pipelines ready for cloud platforms like Google Cloud AI or Alibaba Cloud.

---

## Goals
- **Efficiency**: Skip retrieval when not needed (e.g., chit-chat, general knowledge).
- **Accuracy**: Route queries to the right retrieval strategy (vanilla, fusion, sub-questions, etc.).
- **Safety**: Detect unsafe or adversarial queries early.
- **Observability**: Provide structured metadata for debugging and monitoring.
- **Adaptability**: Reusable across projects as a lightweight, open-source module.

---

## Pipeline Flow
User Query → Normalizer → Query Classifier → Router
├─ LLM only (if no retrieval needed)
├─ Block (if unsafe)
├─ Retrieval strategy (vanilla, fusion, subq, stepback)
Retriever → LLM → Final Response

---

## Key Components

### Intent Classes
- **chitchat**: casual, no retrieval needed  
- **general_knowledge**: factual, LLM can answer directly  
- **internal_knowledge**: requires retrieval from KB  
- **unsafe**: adversarial, abusive, or prompt injection  

### Scores & Flags
- `ambiguity_score` (0–1)  
- `is_overview`, `is_compositional`, `is_alias_or_underspecified` (0–1)  

### Confidence Metrics
- `intent_confidence`: softmax probability  

### Routing Hints
Deterministic rules map outputs to strategies:
- `block`, `llm_only`, `vanilla`, `fusion`, `subq`, `stepback`

**Example JSON Output:**
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
### Architecture

Model: Transformer encoder (DistilBERT / DeBERTa / MiniLM, ~50–100M params).

Multi-task heads: classification + regression for scores/flags.

Loss: cross-entropy (intent) + MSE (scores).

Training Data: 5k–20k labeled queries (chitchat, factual, unsafe, multi-hop).

Training: Hugging Face Transformers on GPU (Google Cloud, Alibaba, Colab).

### Training & Evaluation

Preprocess: normalize, tokenize queries.

Train: multi-task model with class weighting to prioritize internal knowledge recall.

Evaluate:

Intent → Accuracy & F1-score

Scores → MAE

Pipeline-level metrics (retrieval hit rate, token savings)

Deployment: FastAPI endpoint for classification.

### Deployment

Expose via FastAPI:
```py
@app.post("/classify")
def classify(query: str):
    # returns JSON output with intent, scores, route_hint
```

Self-host with Docker or run on cloud instances.

Monitor with logs/Prometheus.

Active learning loop: re-train with misclassified or low-confidence queries.

### Benefits

TBD 
### Check_query
The ```py check_query.py``` file goal is to check that the training querys have the right formating.

Usage :
```
# Validate a single file
python validate_query_jsonl.py /path/to/query_gpt.jsonl

[PASS] /home/melvyn/Projets/query_classifier/data/query_gpt.jsonl
  lines: 135
  intents:
    - chitchat: 14
    - general_knowledge: 12
    - internal_knowledge: 99
    - unsafe: 10
# Validate all .jsonl files in a folder
python validate_query_jsonl.py /path/to/folder

# Be strict and cap reported errors per file
python validate_query_jsonl.py /path/to/folder --strict --max-errors 50


```
### Extensions

Add conversation history as context.

Fine-tune larger models (e.g., LLaMA with PEFT).

Integrate external safety tools (OpenAI moderation, Hugging Face toxicity models).

Package as a PyPI module with pre-trained weights.


---

