"""Classifier Benchmark Script.

L1_baseline_benchmark: Benchmark current all-MiniLM-L6-v2 on test set
L5_finetuned_benchmark: Benchmark fine-tuned model on test set
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

DATA_DIR = Path("data/cuad")
BENCHMARKS_DIR = DATA_DIR / "benchmarks"


def load_test_set():
    """Load test clauses."""
    clauses = []
    with open(DATA_DIR / "splits" / "test.jsonl") as f:
        for line in f:
            clauses.append(json.loads(line))
    return clauses


def load_categories():
    """Load category list."""
    with open(DATA_DIR / "splits" / "split_stats.json") as f:
        stats = json.load(f)
    return stats["categories"]


def create_category_embeddings(model, categories):
    """Create embeddings for category descriptions."""
    category_texts = [f"This clause is about: {cat}" for cat in categories]
    embeddings = model.encode(category_texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def classify_clause(model, clause_text, category_embeddings, categories):
    """Classify a clause by finding most similar category."""
    clause_emb = model.encode([clause_text], convert_to_numpy=True, show_progress_bar=False)[0]

    # Cosine similarity
    similarities = np.dot(category_embeddings, clause_emb) / (
        np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(clause_emb)
    )

    best_idx = np.argmax(similarities)
    return categories[best_idx], float(similarities[best_idx])


def compute_calibration_error(confidences, correct, num_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        bin_size = mask.sum()

        if bin_size > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


def benchmark(model_name_or_path: str, output_prefix: str):
    """Run benchmark on test set."""
    print(f"\nLoading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)

    print("Loading test set...")
    test_clauses = load_test_set()
    categories = load_categories()
    print(f"  {len(test_clauses)} clauses, {len(categories)} categories")

    print("Creating category embeddings...")
    category_embeddings = create_category_embeddings(model, categories)

    print("Classifying test clauses...")
    y_true = []
    y_pred = []
    confidences = []

    for i, clause in enumerate(test_clauses):
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(test_clauses)}")

        true_cat = clause["category"]
        pred_cat, confidence = classify_clause(model, clause["text"], category_embeddings, categories)

        y_true.append(true_cat)
        y_pred.append(pred_cat)
        confidences.append(confidence)

    # Compute metrics
    print("\nComputing metrics...")
    confidences = np.array(confidences)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    ece = compute_calibration_error(confidences, correct)

    # Per-category metrics
    per_category = {}
    for cat in categories:
        cat_mask = np.array([t == cat for t in y_true])
        if cat_mask.sum() > 0:
            cat_correct = correct[cat_mask]
            cat_conf = confidences[cat_mask]
            per_category[cat] = {
                "count": int(cat_mask.sum()),
                "accuracy": float(cat_correct.mean()),
                "avg_confidence": float(cat_conf.mean()),
            }

    results = {
        "model": model_name_or_path,
        "test_size": len(test_clauses),
        "num_categories": len(categories),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "expected_calibration_error": ece,
        "mean_confidence": float(confidences.mean()),
        "per_category": per_category,
    }

    # Save results
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARKS_DIR / f"{output_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save per-category CSV
    with open(BENCHMARKS_DIR / f"{output_prefix}_per_category.csv", "w") as f:
        f.write("category,count,accuracy,avg_confidence\n")
        for cat, metrics in sorted(per_category.items()):
            f.write(f'"{cat}",{metrics["count"]},{metrics["accuracy"]:.4f},{metrics["avg_confidence"]:.4f}\n')

    print("\n" + "=" * 60)
    print(f"RESULTS: {output_prefix}")
    print("=" * 60)
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Macro F1:     {macro_f1:.4f}")
    print(f"  Weighted F1:  {weighted_f1:.4f}")
    print(f"  ECE:          {ece:.4f}")
    print(f"  Mean Conf:    {confidences.mean():.4f}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_classifier.py [baseline|finetuned|MODEL_PATH]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "baseline":
        benchmark("sentence-transformers/all-MiniLM-L6-v2", "baseline")
    elif mode == "finetuned":
        benchmark("models/cuad-MiniLM-L6-v2-finetuned", "finetuned")
    else:
        # Custom model path
        benchmark(mode, "custom")


if __name__ == "__main__":
    main()
