"""Fine-tune sentence-transformer on CUAD training pairs.

L3_finetune: Fine-tune all-MiniLM-L6-v2 using MultipleNegativesRankingLoss
"""

import json
import os
from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

DATA_DIR = Path("data/cuad")
MODELS_DIR = Path("models")
TRAINING_DIR = DATA_DIR / "training"


def load_training_pairs(max_pairs: int | None = None):
    """Load training pairs for contrastive learning."""
    pairs = []
    with open(DATA_DIR / "pairs" / "train_pairs.jsonl") as f:
        for line in f:
            pairs.append(json.loads(line))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def load_validation_pairs(max_pairs: int | None = None):
    """Load validation pairs."""
    pairs = []
    with open(DATA_DIR / "pairs" / "val_pairs.jsonl") as f:
        for line in f:
            pairs.append(json.loads(line))
            if max_pairs and len(pairs) >= max_pairs:
                break
    return pairs


def create_training_examples(pairs):
    """Convert pairs to InputExamples for sentence-transformers."""
    examples = []
    for pair in pairs:
        # For MultipleNegativesRankingLoss, we need (anchor, positive) pairs
        # Negatives are sampled from other positives in the batch
        if pair["label"] == 1.0:  # Only use positive pairs
            examples.append(InputExample(
                texts=[pair["anchor"], pair["positive"]]
            ))
    return examples


def create_eval_examples(pairs):
    """Create evaluation examples with similarity scores."""
    examples = []
    for pair in pairs:
        examples.append(InputExample(
            texts=[pair["anchor"], pair["positive"]],
            label=pair["label"]
        ))
    return examples


def main():
    print("=" * 60)
    print("L3_finetune: Fine-tuning sentence-transformer on CUAD")
    print("=" * 60)

    # Config - optimized for CPU training
    config = {
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "output_model": str(MODELS_DIR / "cuad-MiniLM-L6-v2-finetuned"),
        "epochs": 1,  # Reduced for CPU
        "batch_size": 16,  # Smaller batch for CPU memory
        "learning_rate": 2e-5,
        "warmup_ratio": 0.1,
        "max_train_pairs": 10000,  # Limit for faster training
        "max_val_pairs": 1000,
        "evaluation_steps": 500,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print(f"\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(TRAINING_DIR / "finetune_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nLoading base model: {config['base_model']}")
    model = SentenceTransformer(config["base_model"], device=config["device"])

    print(f"\nLoading training pairs (max {config['max_train_pairs']})...")
    train_pairs = load_training_pairs(config["max_train_pairs"])
    train_examples = create_training_examples(train_pairs)
    print(f"  Created {len(train_examples)} training examples")

    print(f"\nLoading validation pairs (max {config['max_val_pairs']})...")
    val_pairs = load_validation_pairs(config["max_val_pairs"])
    val_examples = create_eval_examples(val_pairs)
    print(f"  Created {len(val_examples)} validation examples")

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=config["batch_size"]
    )

    # Use MultipleNegativesRankingLoss - effective for contrastive learning
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Create evaluator
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        val_examples,
        name="cuad-val"
    )

    # Calculate training steps
    num_training_steps = len(train_dataloader) * config["epochs"]
    warmup_steps = int(num_training_steps * config["warmup_ratio"])

    print(f"\nTraining:")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Evaluation every {config['evaluation_steps']} steps")
    print(f"  Device: {config['device']}")

    # Training log
    training_log = {
        "config": config,
        "start_time": datetime.now().isoformat(),
        "num_train_examples": len(train_examples),
        "num_val_examples": len(val_examples),
    }

    print("\n" + "=" * 60)
    print("Starting training... (this may take 1-2 hours on CPU)")
    print("=" * 60)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=config["epochs"],
        evaluation_steps=config["evaluation_steps"],
        warmup_steps=warmup_steps,
        output_path=config["output_model"],
        show_progress_bar=True,
        save_best_model=True,
    )

    training_log["end_time"] = datetime.now().isoformat()

    # Save training log
    with open(TRAINING_DIR / "training_log.json", "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 60)
    print("L3_finetune COMPLETE")
    print("=" * 60)
    print(f"  Model saved to: {config['output_model']}")
    print(f"  Training log: {TRAINING_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
