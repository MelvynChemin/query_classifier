#!/usr/bin/env python3
"""
Script pour convertir combined_data.jsonl au format attendu par eval.py
"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_dataset(input_path, output_dir, test_size=0.2, random_state=42):
    """
    Convertit le dataset et le split en train/test
    
    Args:
        input_path: chemin vers combined_data.jsonl
        output_dir: dossier de sortie
        test_size: proportion du test set (0.2 = 20%)
        random_state: seed pour reproductibilité
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Charger les données
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            # Transformer au format attendu
            converted = {
                "text": record["query"],
                "label": record["labels"]["intent"]
            }
            data.append(converted)
    
    print(f"✓ Chargé {len(data)} exemples")
    
    # Afficher la distribution des classes
    label_counts = {}
    for item in data:
        label = item["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\n=== Distribution des classes ===")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:20s}: {count:4d} ({count/len(data)*100:5.1f}%)")
    
    # Split train/test de manière stratifiée
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels  # Préserve les proportions de classes
    )
    
    # Sauvegarder train.jsonl
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for text, label in zip(train_texts, train_labels):
            json.dump({"text": text, "label": label}, f, ensure_ascii=False)
            f.write('\n')
    
    # Sauvegarder test.jsonl
    test_path = output_dir / "test.jsonl"
    with open(test_path, 'w', encoding='utf-8') as f:
        for text, label in zip(test_texts, test_labels):
            json.dump({"text": text, "label": label}, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"\n✓ Train set: {len(train_texts)} exemples → {train_path}")
    print(f"✓ Test set:  {len(test_texts)} exemples → {test_path}")
    
    return train_path, test_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prépare les données pour l'évaluation du classificateur"
    )
    parser.add_argument(
        "--input",
        default="combined_data.jsonl",
        help="Chemin vers le fichier combined_data.jsonl"
    )
    parser.add_argument(
        "--output",
        default="data",
        help="Dossier de sortie pour train.jsonl et test.jsonl"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion du test set (défaut: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed pour reproductibilité"
    )
    
    args = parser.parse_args()
    
    train_path, test_path = convert_dataset(
        args.input,
        args.output,
        test_size=args.test_size,
        random_state=args.seed
    )
    
    print(f"\n✓ Prêt pour l'évaluation !")
    print(f"  Commande: python eval.py --model MelvynCHEMIN/query_classifier --test {test_path}")