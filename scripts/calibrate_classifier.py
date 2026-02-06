"""Confidence Calibration via Platt Scaling.

L2_calibration_analysis: Analyze confidence distribution on validation set
L2_platt_scaling: Fit Platt scaling parameters
L4_recalibrate: Fit Platt scaling to fine-tuned model
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/cuad")
CALIBRATION_DIR = DATA_DIR / "calibration"


def load_validation_set():
    """Load validation clauses."""
    clauses = []
    with open(DATA_DIR / "splits" / "val.jsonl") as f:
        for line in f:
            clauses.append(json.loads(line))
    return clauses


def load_categories():
    """Load category list."""
    with open(DATA_DIR / "splits" / "split_stats.json") as f:
        stats = json.load(f)
    return stats["categories"]


def get_predictions(model, clauses, categories):
    """Get predictions and confidences for all clauses."""
    category_texts = [f"This clause is about: {cat}" for cat in categories]
    category_embeddings = model.encode(category_texts, convert_to_numpy=True, show_progress_bar=False)

    predictions = []
    confidences = []
    correct = []

    for clause in clauses:
        clause_emb = model.encode([clause["text"]], convert_to_numpy=True, show_progress_bar=False)[0]

        similarities = np.dot(category_embeddings, clause_emb) / (
            np.linalg.norm(category_embeddings, axis=1) * np.linalg.norm(clause_emb)
        )

        best_idx = np.argmax(similarities)
        pred_cat = categories[best_idx]
        confidence = float(similarities[best_idx])

        predictions.append(pred_cat)
        confidences.append(confidence)
        correct.append(1 if pred_cat == clause["category"] else 0)

    return predictions, np.array(confidences), np.array(correct)


def compute_ece(confidences, correct, num_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total = len(confidences)

    bin_data = []
    for i in range(num_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lower) & (confidences < upper)
        bin_size = mask.sum()

        if bin_size > 0:
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_data.append({
                "bin": i,
                "lower": lower,
                "upper": upper,
                "count": int(bin_size),
                "accuracy": float(bin_acc),
                "avg_confidence": float(bin_conf),
            })
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece, bin_data


def platt_scaling_nll(params, confidences, labels):
    """Negative log-likelihood for Platt scaling."""
    A, B = params
    # Apply sigmoid: P = 1 / (1 + exp(A * conf + B))
    probs = 1.0 / (1.0 + np.exp(A * confidences + B))
    probs = np.clip(probs, 1e-10, 1 - 1e-10)

    # Binary cross-entropy
    nll = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return nll


def fit_platt_scaling(confidences, correct):
    """Fit Platt scaling parameters."""
    # Initial guess
    A0, B0 = -1.0, 0.0

    result = minimize(
        platt_scaling_nll,
        [A0, B0],
        args=(confidences, correct),
        method="L-BFGS-B",
    )

    A, B = result.x
    return A, B


def apply_platt_scaling(confidences, A, B):
    """Apply Platt scaling to confidences."""
    return 1.0 / (1.0 + np.exp(A * confidences + B))


def calibrate(model_name_or_path: str, output_prefix: str):
    """Run calibration analysis and fit Platt scaling."""
    print(f"\nLoading model: {model_name_or_path}")
    model = SentenceTransformer(model_name_or_path)

    print("Loading validation set...")
    val_clauses = load_validation_set()
    categories = load_categories()
    print(f"  {len(val_clauses)} clauses, {len(categories)} categories")

    print("Getting predictions...")
    predictions, confidences, correct = get_predictions(model, val_clauses, categories)

    print("\nAnalyzing raw calibration...")
    raw_ece, raw_bins = compute_ece(confidences, correct)
    raw_accuracy = correct.mean()

    print(f"  Raw accuracy: {raw_accuracy:.4f}")
    print(f"  Raw ECE: {raw_ece:.4f}")
    print(f"  Mean confidence: {confidences.mean():.4f}")

    # Save raw calibration stats
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    raw_stats = {
        "model": model_name_or_path,
        "val_size": len(val_clauses),
        "accuracy": float(raw_accuracy),
        "ece": float(raw_ece),
        "mean_confidence": float(confidences.mean()),
        "bins": raw_bins,
    }
    with open(CALIBRATION_DIR / f"{output_prefix}_calibration_stats.json", "w") as f:
        json.dump(raw_stats, f, indent=2)

    print("\nFitting Platt scaling...")
    A, B = fit_platt_scaling(confidences, correct)
    print(f"  Platt params: A={A:.4f}, B={B:.4f}")

    # Apply Platt scaling
    calibrated_conf = apply_platt_scaling(confidences, A, B)
    calibrated_ece, calibrated_bins = compute_ece(calibrated_conf, correct)

    print(f"\nCalibrated ECE: {calibrated_ece:.4f}")
    print(f"ECE improvement: {(raw_ece - calibrated_ece) / raw_ece * 100:.1f}%")

    # Save Platt parameters
    platt_params = {
        "A": float(A),
        "B": float(B),
        "raw_ece": float(raw_ece),
        "calibrated_ece": float(calibrated_ece),
        "ece_improvement": float((raw_ece - calibrated_ece) / raw_ece),
    }
    with open(CALIBRATION_DIR / f"{output_prefix}_platt_params.json", "w") as f:
        json.dump(platt_params, f, indent=2)

    print("\n" + "=" * 60)
    print(f"CALIBRATION COMPLETE: {output_prefix}")
    print("=" * 60)
    print(f"  Raw ECE:        {raw_ece:.4f}")
    print(f"  Calibrated ECE: {calibrated_ece:.4f}")
    print(f"  Platt A:        {A:.4f}")
    print(f"  Platt B:        {B:.4f}")

    return platt_params


def compare_calibrations():
    """Compare baseline vs fine-tuned calibration."""
    baseline_path = CALIBRATION_DIR / "baseline_platt_params.json"
    finetuned_path = CALIBRATION_DIR / "finetuned_platt_params.json"

    if not baseline_path.exists() or not finetuned_path.exists():
        print("Both baseline and finetuned calibration files required for comparison")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(finetuned_path) as f:
        finetuned = json.load(f)

    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 60)
    print(f"{'Raw ECE':<25} {baseline['raw_ece']:>12.4f} {finetuned['raw_ece']:>12.4f} {finetuned['raw_ece'] - baseline['raw_ece']:>+12.4f}")
    print(f"{'Calibrated ECE':<25} {baseline['calibrated_ece']:>12.4f} {finetuned['calibrated_ece']:>12.4f} {finetuned['calibrated_ece'] - baseline['calibrated_ece']:>+12.4f}")

    comparison = {
        "baseline_raw_ece": baseline["raw_ece"],
        "baseline_calibrated_ece": baseline["calibrated_ece"],
        "finetuned_raw_ece": finetuned["raw_ece"],
        "finetuned_calibrated_ece": finetuned["calibrated_ece"],
        "raw_ece_improvement": baseline["raw_ece"] - finetuned["raw_ece"],
        "calibrated_ece_improvement": baseline["calibrated_ece"] - finetuned["calibrated_ece"],
    }

    with open(CALIBRATION_DIR / "calibration_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)


def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_classifier.py [baseline|finetuned|compare]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "baseline":
        calibrate("sentence-transformers/all-MiniLM-L6-v2", "baseline")
    elif mode == "finetuned":
        calibrate("models/cuad-MiniLM-L6-v2-finetuned", "finetuned")
    elif mode == "compare":
        compare_calibrations()
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
