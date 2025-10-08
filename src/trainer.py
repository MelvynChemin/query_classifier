import json
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss, MSELoss
from classifier import QueryClassifier, tokenizer

# Intent map (adjust if labels differ)
intent_map = {'chitchat': 0, 'general_knowledge': 1, 'internal_knowledge': 2, 'unsafe': 3}

def load_data(split):
    with open(f'data/{split}.json', 'r') as f:
        data = json.load(f)
    # Add intent_id
    for item in data:
        item['intent_id'] = intent_map[item['labels']['intent']]
    return Dataset.from_list(data)

train_ds = load_data('train')
val_ds = load_data('val')

def preprocess(examples):
    # When batched=True, examples is a dict where each value is a list
    tokenized = tokenizer(
        examples['query'], 
        truncation=True, 
        padding='max_length', 
        max_length=128
    )
    
    # Build score lists from the labels (flatten into separate columns)
    ambiguity_scores = []
    is_overview_scores = []
    is_compositional_scores = []
    is_alias_scores = []
    
    for label_dict in examples['labels']:
        ambiguity_scores.append(label_dict['ambiguity_score'])
        is_overview_scores.append(label_dict['is_overview'])
        is_compositional_scores.append(label_dict['is_compositional'])
        is_alias_scores.append(label_dict['is_alias_or_underspecified'])
    
    return {
        **tokenized,
        'intent_labels': examples['intent_id'],
        'ambiguity_score': ambiguity_scores,
        'is_overview': is_overview_scores,
        'is_compositional': is_compositional_scores,
        'is_alias_or_underspecified': is_alias_scores
    }

train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

# Set format to PyTorch tensors
train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'intent_labels', 
                                            'ambiguity_score', 'is_overview', 
                                            'is_compositional', 'is_alias_or_underspecified'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'intent_labels',
                                          'ambiguity_score', 'is_overview',
                                          'is_compositional', 'is_alias_or_underspecified'])

model = QueryClassifier()
label_keys = [
    'intent_labels',
    'ambiguity_score',
    'is_overview',
    'is_compositional',
    'is_alias_or_underspecified'
]
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # --- pull labels
        intent_labels = inputs.pop('intent_labels')
        score_labels = torch.stack([
            inputs.pop('ambiguity_score'),
            inputs.pop('is_overview'),
            inputs.pop('is_compositional'),
            inputs.pop('is_alias_or_underspecified'),
        ], dim=1).float()  # [B,4]

        # --- forward
        logits, scores = model(**inputs)  # your model returns (logits, dict_of_scores)

        device = logits.device
        intent_labels = intent_labels.to(device)
        score_labels = score_labels.to(device)

        # order scores consistently with labels
        pred_scores = torch.stack([
            scores['ambiguity_score'].squeeze(-1),
            scores['is_overview'].squeeze(-1),
            scores['is_compositional'].squeeze(-1),
            scores['is_alias_or_underspecified'].squeeze(-1),
        ], dim=1).float()  # [B,4]
        assert pred_scores.dim() == 2 and pred_scores.size(1) == 4, f"pred_scores shape: {pred_scores.shape}"
        assert score_labels.shape == pred_scores.shape, f"labels {score_labels.shape} vs preds {pred_scores.shape}"
        ce_loss = CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 2.0, 1.0], device=device)
        )(logits, intent_labels)

        mse_loss = MSELoss()(pred_scores, score_labels)

        loss = ce_loss + mse_loss
        return (loss, (logits, scores)) if return_outputs else loss

    # --- OLD HF compatibility: ensure eval/predict don't pass labels to model(**inputs)
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # keep labels aside
        labels = {k: inputs[k] for k in label_keys if k in inputs}

        # strip labels before forward
        for k in label_keys:
            inputs.pop(k, None)

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            outputs = model(**inputs)  # (logits, scores)

        # ALWAYS compute loss if we have labels (so eval_loss exists)
        loss = None
        if labels:
            loss_inputs = dict(inputs)
            loss_inputs.update(labels)
            loss = self.compute_loss(model, loss_inputs)
            if isinstance(loss, tuple):  # (loss, outputs)
                loss = loss[0]
            loss = loss.detach()

        logits, _scores = outputs

        if prediction_loss_only:
            # evaluator only needs the loss; logits/labels can be None
            return (loss, None, None)

        # otherwise also return logits and intent labels for metrics
        intent = labels.get('intent_labels', None)
        return (loss, logits.detach(), None if intent is None else intent.detach())
training_args = TrainingArguments(
    output_dir="models/",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    eval_strategy="epoch",   # <- not eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    logging_steps=10,
    remove_unused_columns=False
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)


trainer.train()
trainer.save_model("models/best")
print("Training complete. Best model saved.")