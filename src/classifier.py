import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class QueryClassifier(nn.Module):
    def __init__(self, base_model_name="distilbert-base-uncased"):
        super().__init__()
        self.base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=4, output_hidden_states=True
        )  # Enable hidden states
        hidden_size = self.base.config.hidden_size  # 768 for DistilBERT
        self.regression_heads = nn.ModuleDict({
            'ambiguity_score': nn.Linear(hidden_size, 1),
            'is_overview': nn.Linear(hidden_size, 1),
            'is_compositional': nn.Linear(hidden_size, 1),
            'is_alias_or_underspecified': nn.Linear(hidden_size, 1)
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        hidden = outputs.hidden_states[-1][:, 0, :]  # [CLS] token
        scores = {k: torch.sigmoid(v(hidden)) for k, v in self.regression_heads.items()}
        return logits, scores

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")