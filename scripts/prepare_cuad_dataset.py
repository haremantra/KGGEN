"""CUAD Dataset Preparation for Classifier Fine-Tuning.

L0_dataset_split: Create train/val/test splits (70/15/15)
L0_training_pairs: Generate sentence pairs with hard negatives for contrastive learning
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_DIR = Path("data/cuad")
RAW_FILE = DATA_DIR / "raw" / "CUADv1.json"
SPLITS_DIR = DATA_DIR / "splits"
PAIRS_DIR = DATA_DIR / "pairs"


def load_cuad():
    """Load CUAD dataset and extract clause-label pairs."""
    with open(RAW_FILE) as f:
        data = json.load(f)

    contracts = []
    all_categories = set()

    for entry in data["data"]:
        contract_id = entry["title"]
        context = entry["paragraphs"][0]["context"]
        qas = entry["paragraphs"][0]["qas"]

        clauses = []
        for qa in qas:
            question = qa["question"]
            # Extract category from question (e.g., "Highlight the parts...")
            # The question contains the category name
            category = extract_category(question)
            all_categories.add(category)

            # Get answers (clause texts)
            answers = qa.get("answers", [])
            if answers:
                for ans in answers:
                    clause_text = ans.get("text", "").strip()
                    if clause_text and len(clause_text) > 20:
                        clauses.append({
                            "text": clause_text,
                            "category": category,
                            "contract_id": contract_id,
                        })

        contracts.append({
            "contract_id": contract_id,
            "context": context,
            "clauses": clauses,
        })

    return contracts, sorted(all_categories)


def extract_category(question: str) -> str:
    """Extract CUAD category from question text."""
    # Questions are like: "Highlight the parts (if any) of this contract related to..."
    # followed by the category description
    q_lower = question.lower()

    # Map question patterns to CUAD labels
    category_patterns = {
        "document name": "Document Name",
        "parties": "Parties",
        "agreement date": "Agreement Date",
        "effective date": "Effective Date",
        "expiration date": "Expiration Date",
        "renewal term": "Renewal Term",
        "notice period to terminate renewal": "Notice Period To Terminate Renewal",
        "governing law": "Governing Law",
        "most favored nation": "Most Favored Nation",
        "non-compete": "Non-Compete",
        "exclusivity": "Exclusivity",
        "no-solicit of customers": "No-Solicit Of Customers",
        "no-solicit of employees": "No-Solicit Of Employees",
        "non-disparagement": "Non-Disparagement",
        "termination for convenience": "Termination For Convenience",
        "rofr/rofo/rofn": "Rofr/Rofo/Rofn",
        "change of control": "Change Of Control",
        "anti-assignment": "Anti-Assignment",
        "revenue/profit sharing": "Revenue/Profit Sharing",
        "price restriction": "Price Restrictions",
        "minimum commitment": "Minimum Commitment",
        "volume restriction": "Volume Restriction",
        "ip ownership assignment": "IP Ownership Assignment",
        "joint ip ownership": "Joint IP Ownership",
        "license grant": "License Grant",
        "non-transferable license": "Non-Transferable License",
        "affiliate license-licensor": "Affiliate License-Licensor",
        "affiliate license-licensee": "Affiliate License-Licensee",
        "unlimited/all-you-can-eat": "Unlimited/All-You-Can-Eat-License",
        "irrevocable or perpetual": "Irrevocable Or Perpetual License",
        "source code escrow": "Source Code Escrow",
        "post-termination services": "Post-Termination Services",
        "audit rights": "Audit Rights",
        "uncapped liability": "Uncapped Liability",
        "cap on liability": "Cap On Liability",
        "liquidated damages": "Liquidated Damages",
        "warranty duration": "Warranty Duration",
        "insurance": "Insurance",
        "covenant not to sue": "Covenant Not To Sue",
        "third party beneficiary": "Third Party Beneficiary",
        "competitive restriction exception": "Competitive Restriction Exception",
    }

    for pattern, label in category_patterns.items():
        if pattern in q_lower:
            return label

    # Fallback: extract from question
    return question[:50]


def create_splits(contracts: list, train_ratio=0.7, val_ratio=0.15):
    """Create stratified train/val/test splits."""
    random.shuffle(contracts)

    n = len(contracts)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = contracts[:train_end]
    val = contracts[train_end:val_end]
    test = contracts[val_end:]

    return train, val, test


def flatten_clauses(contracts: list) -> list:
    """Flatten contracts to list of (clause_text, category) pairs."""
    clauses = []
    for contract in contracts:
        for clause in contract["clauses"]:
            clauses.append({
                "text": clause["text"],
                "category": clause["category"],
                "contract_id": clause["contract_id"],
            })
    return clauses


def generate_training_pairs(clauses: list, categories: list, num_negatives=4) -> list:
    """Generate sentence pairs with hard negatives for contrastive learning."""
    # Group clauses by category
    by_category = defaultdict(list)
    for clause in clauses:
        by_category[clause["category"]].append(clause["text"])

    pairs = []

    for clause in clauses:
        text = clause["text"]
        category = clause["category"]

        # Positive pair: (clause, category_description)
        pairs.append({
            "anchor": text,
            "positive": f"This clause is about: {category}",
            "label": 1.0,
        })

        # Hard negatives: similar categories
        other_categories = [c for c in categories if c != category]
        negative_cats = random.sample(other_categories, min(num_negatives, len(other_categories)))

        for neg_cat in negative_cats:
            pairs.append({
                "anchor": text,
                "positive": f"This clause is about: {neg_cat}",
                "label": 0.0,
            })

    return pairs


def save_splits(train, val, test, categories):
    """Save splits to JSONL files."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    train_clauses = flatten_clauses(train)
    val_clauses = flatten_clauses(val)
    test_clauses = flatten_clauses(test)

    save_jsonl(train_clauses, SPLITS_DIR / "train.jsonl")
    save_jsonl(val_clauses, SPLITS_DIR / "val.jsonl")
    save_jsonl(test_clauses, SPLITS_DIR / "test.jsonl")

    # Save stats
    stats = {
        "train_contracts": len(train),
        "val_contracts": len(val),
        "test_contracts": len(test),
        "train_clauses": len(train_clauses),
        "val_clauses": len(val_clauses),
        "test_clauses": len(test_clauses),
        "categories": categories,
        "num_categories": len(categories),
    }

    with open(SPLITS_DIR / "split_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return train_clauses, val_clauses, test_clauses, stats


def save_pairs(train_clauses, val_clauses, categories):
    """Save training pairs for contrastive learning."""
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs = generate_training_pairs(train_clauses, categories)
    val_pairs = generate_training_pairs(val_clauses, categories, num_negatives=2)

    def save_jsonl(data, path):
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    save_jsonl(train_pairs, PAIRS_DIR / "train_pairs.jsonl")
    save_jsonl(val_pairs, PAIRS_DIR / "val_pairs.jsonl")

    stats = {
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "positive_pairs_train": sum(1 for p in train_pairs if p["label"] == 1.0),
        "negative_pairs_train": sum(1 for p in train_pairs if p["label"] == 0.0),
    }

    with open(PAIRS_DIR / "pair_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    print("=" * 60)
    print("L0_dataset_download: Loading CUAD dataset...")
    print("=" * 60)
    contracts, categories = load_cuad()
    print(f"  Loaded {len(contracts)} contracts")
    print(f"  Found {len(categories)} categories")

    print("\n" + "=" * 60)
    print("L0_dataset_split: Creating train/val/test splits...")
    print("=" * 60)
    train, val, test = create_splits(contracts)
    train_clauses, val_clauses, test_clauses, split_stats = save_splits(train, val, test, categories)
    print(f"  Train: {split_stats['train_contracts']} contracts, {split_stats['train_clauses']} clauses")
    print(f"  Val:   {split_stats['val_contracts']} contracts, {split_stats['val_clauses']} clauses")
    print(f"  Test:  {split_stats['test_contracts']} contracts, {split_stats['test_clauses']} clauses")

    print("\n" + "=" * 60)
    print("L0_training_pairs: Generating training pairs...")
    print("=" * 60)
    pair_stats = save_pairs(train_clauses, val_clauses, categories)
    print(f"  Train pairs: {pair_stats['train_pairs']} ({pair_stats['positive_pairs_train']} pos, {pair_stats['negative_pairs_train']} neg)")
    print(f"  Val pairs:   {pair_stats['val_pairs']}")

    print("\n" + "=" * 60)
    print("L0 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
