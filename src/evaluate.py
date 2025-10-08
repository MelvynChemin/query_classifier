from sklearn.metrics import f1_score, mean_absolute_error, recall_score
import torch
from transformers import trainer
from trainer import load_data, preprocess, train_ds, val_ds
test_ds = load_data('test').map(preprocess, batched=True)
predictions = trainer.predict(test_ds)
logits, scores = predictions.predictions
pred_intents = torch.argmax(torch.tensor(logits), dim=1).numpy()
true_intents = predictions.label_ids['intent'].numpy()
pred_scores = torch.cat([scores[k] for k in scores.keys()], dim=1).numpy()  # Concat
true_scores = predictions.label_ids['scores'].numpy()

print("Intent F1:", f1_score(true_intents, pred_intents, average='macro'))
print("Scores MAE:", mean_absolute_error(true_scores, pred_scores))
print("Internal Knowledge Recall:", recall_score(true_intents == 2, pred_intents == 2))  # Aim >0.95
print("Unsafe Recall:", recall_score(true_intents == 3, pred_intents == 3))