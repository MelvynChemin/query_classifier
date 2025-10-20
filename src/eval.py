#!/usr/bin/env python3
"""
Script d'évaluation amélioré pour le classificateur de requêtes
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    f1_score,
    accuracy_score
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple

# Définition des labels
LABELS = ["chitchat", "general_knowledge", "internal_knowledge", "unsafe"]
id2label = {i: l for i, l in enumerate(LABELS)}
label2id = {l: i for i, l in enumerate(LABELS)}


def load_jsonl(path: str) -> List[Dict]:
    """Charge un fichier JSONL"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def predict(model, tokenizer, texts, batch_size=64, device="cpu", max_length=512):
    import torch
    from torch.nn.functional import softmax

    # Récupère le device effectif du modèle (source de vérité)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device(device)

    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            # TOUT sur le même device que le modèle
            encoded = {k: v.to(model_device, non_blocking=True) for k, v in encoded.items()}

            out = model(**encoded)

            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, dict) and "logits" in out:
                logits = out["logits"]
            elif isinstance(out, (list, tuple)) and len(out) > 0:
                logits = out[0]
            else:
                raise TypeError(f"Sortie modèle inattendue: type={type(out)}")

            all_logits.append(logits.detach())

    logits = torch.cat(all_logits, dim=0)
    probs = softmax(logits, dim=-1).cpu().numpy()
    return probs




def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                  probs: np.ndarray, labels: List[str]):
    """Affiche toutes les métriques d'évaluation"""
    
    print("\n" + "="*80)
    print("RAPPORT DE CLASSIFICATION")
    print("="*80)
    print(classification_report(
        y_true, y_pred, 
        target_names=labels, 
        digits=4
    ))
    
    print("\n" + "="*80)
    print("MATRICE DE CONFUSION")
    print("="*80)
    cm = confusion_matrix(y_true, y_pred)
    
    # En-tête
    print(f"\n{'Réel \\ Prédit':<20}", end="")
    for label in labels:
        print(f"{label:<20}", end="")
    print()
    print("-" * (20 + 20 * len(labels)))
    
    # Lignes
    for i, label in enumerate(labels):
        print(f"{label:<20}", end="")
        for j in range(len(labels)):
            print(f"{cm[i, j]:<20}", end="")
        print()
    
    print("\n" + "="*80)
    print("MÉTRIQUES GLOBALES")
    print("="*80)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # F1-scores
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1 Macro:        {f1_macro:.4f}")
    print(f"F1 Weighted:     {f1_weighted:.4f}")
    
    # ROC-AUC
    try:
        auc_macro = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
        auc_weighted = roc_auc_score(y_true, probs, multi_class="ovr", average="weighted")
        print(f"ROC-AUC Macro:   {auc_macro:.4f}")
        print(f"ROC-AUC Weighted: {auc_weighted:.4f}")
    except Exception as e:
        print(f"ROC-AUC: N/A (erreur: {e})")
    
    print("\n" + "="*80)
    print("ANALYSE PAR CLASSE")
    print("="*80)
    
    for i, label in enumerate(labels):
        mask = (y_true == i)
        if mask.sum() == 0:
            continue
        
        class_accuracy = (y_pred[mask] == i).mean()
        print(f"\n{label}:")
        print(f"  Exemples:     {mask.sum()}")
        print(f"  Accuracy:     {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
        
        # Erreurs principales
        errors = y_pred[mask & (y_pred != i)]
        if len(errors) > 0:
            unique, counts = np.unique(errors, return_counts=True)
            print(f"  Erreurs vers:")
            for err_class, count in sorted(zip(unique, counts), key=lambda x: -x[1])[:3]:
                print(f"    - {labels[err_class]}: {count} fois")


def analyze_confidence(probs: np.ndarray, y_true: np.ndarray, 
                       y_pred: np.ndarray, labels: List[str]):
    """Analyse la confiance des prédictions"""
    
    print("\n" + "="*80)
    print("ANALYSE DE CONFIANCE")
    print("="*80)
    
    max_probs = probs.max(axis=1)
    correct = (y_pred == y_true)
    
    print(f"\nConfiance moyenne (toutes): {max_probs.mean():.4f}")
    print(f"Confiance moyenne (correctes): {max_probs[correct].mean():.4f}")
    print(f"Confiance moyenne (erreurs): {max_probs[~correct].mean():.4f}")
    
    # Distribution par quartiles
    print("\nDistribution par confiance:")
    for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
        threshold = np.quantile(max_probs, q)
        above = max_probs >= threshold
        acc = correct[above].mean() if above.sum() > 0 else 0
        print(f"  Top {int((1-q)*100):2d}% (conf >= {threshold:.3f}): "
              f"{above.sum():4d} exemples, accuracy = {acc:.4f}")


def save_errors(test_data: List[Dict], y_true: np.ndarray, 
                y_pred: np.ndarray, probs: np.ndarray, 
                labels: List[str], output_path: str):
    """Sauvegarde les erreurs pour analyse"""
    
    errors = []
    for i, (item, true_idx, pred_idx) in enumerate(zip(test_data, y_true, y_pred)):
        if true_idx != pred_idx:
            errors.append({
                "text": item["text"],
                "true_label": labels[true_idx],
                "pred_label": labels[pred_idx],
                "confidence": float(probs[i, pred_idx]),
                "probabilities": {
                    label: float(probs[i, j]) 
                    for j, label in enumerate(labels)
                }
            })
    
    # Trier par confiance décroissante
    errors.sort(key=lambda x: -x["confidence"])
    
    output_path = Path(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ {len(errors)} erreurs sauvegardées dans {output_path}")


def main(model_name: str, test_path: str, 
         batch_size: int = 64, device: str = None,
         save_errors_path: str = None):
    """Fonction principale d'évaluation"""
    
    # Détection automatique du device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("ÉVALUATION DU CLASSIFICATEUR DE REQUÊTES")
    print("="*80)
    print(f"\nModèle:  {model_name}")
    print(f"Test:    {test_path}")
    print(f"Device:  {device}")
    print(f"Batch:   {batch_size}")
    
    # Charger les données de test
    print("\nChargement des données...")
    test_data = load_jsonl(test_path)
    X = [r["text"] for r in test_data]
    y = np.array([label2id[r["label"]] for r in test_data])
    
    print(f"✓ {len(test_data)} exemples chargés")
    
    # Distribution des classes
    print("\nDistribution des classes dans le test set:")
    for label_name in LABELS:
        count = (y == label2id[label_name]).sum()
        print(f"  {label_name:<20}: {count:4d} ({count/len(y)*100:5.1f}%)")
    
    # Charger le modèle
    # --- remplace ce bloc dans eval.py ---
    print(f"\nChargement du modèle {model_name}...")

    try:
        # Tentative: repo HF *complet*
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("✓ Modèle HF chargé")
    except Exception as e:
        print(f"⚠️ Chargement HF impossible ({e}). Fallback vers le modèle custom QueryClassifier.")
        # Fallback: modèle custom + poids locaux (même logique que test_model.py)
        from classifier import QueryClassifier, tokenizer  # ton tokenizer + classe
        # import torch
        from pathlib import Path

        MODEL_DIR = Path("models/best")
        safe = MODEL_DIR / "model.safetensors"
        pt   = MODEL_DIR / "pytorch_model.bin"
        if safe.exists():
            from safetensors.torch import load_file
            state = load_file(str(safe))
        elif pt.exists():
            state = torch.load(str(pt), map_location="cpu")
        else:
            raise FileNotFoundError(f"Aucun poids trouvé dans {MODEL_DIR} (model.safetensors ou pytorch_model.bin).")

        # Nettoyage d'éventuels préfixes 'module.' / state_dict imbriqué
        if isinstance(state, dict) and "state_dict" in state: state = state["state_dict"]
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in state.items() }

        model = QueryClassifier()
        missing, unexpected = model.load_state_dict(state, strict=False)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # IMPORTANT: déplacer le modèle APRÈS avoir chargé les poids
        model.to(device)
        model.eval()

        # (optionnel) forcer return_dict pour la sortie .logits
        if hasattr(model, "config") and hasattr(model.config, "return_dict"):
            model.config.return_dict = True
        if missing or unexpected:
            print("⚠️ Mismatch state_dict:",
                f"\n  Missing: {missing}\n  Unexpected: {unexpected}")
        print("✓ Modèle custom chargé")
    # --- fin patch ---

    
    # Prédiction
    probs = predict(model, tokenizer, X, batch_size=batch_size, device=device)
    y_pred = probs.argmax(axis=1)
    
    # Métriques
    print_metrics(y, y_pred, probs, LABELS)
    
    # Analyse de confiance
    analyze_confidence(probs, y, y_pred, LABELS)
    
    # Sauvegarder les erreurs
    if save_errors_path:
        save_errors(test_data, y, y_pred, probs, LABELS, save_errors_path)
    
    print("\n" + "="*80)
    print("ÉVALUATION TERMINÉE")
    print("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Évalue un classificateur de requêtes"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Nom ou chemin du modèle HuggingFace (ex: MelvynCHEMIN/query_classifier)"
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Chemin vers test.jsonl"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille des batchs pour l'inférence (défaut: 64)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device à utiliser (défaut: auto)"
    )
    parser.add_argument(
        "--save-errors",
        help="Chemin pour sauvegarder les erreurs en JSON"
    )
    
    args = parser.parse_args()
    
    device = None if args.device == "auto" else args.device
    
    main(
        model_name=args.model,
        test_path=args.test,
        batch_size=args.batch_size,
        device=device,
        save_errors_path=args.save_errors
    )