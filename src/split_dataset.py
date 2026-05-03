"""
Encode category labels and create stratified 70/15/15 train/val/test splits.

Critical: synthetic rows are EXCLUDED from val and test sets.
They are appended only to train. This prevents the model from being
evaluated on data it has effectively memorized.

Inputs:  data/text/complaints_labeled.csv (with `source` column)
Outputs: data/processed/{train,val,test}.csv  +  data/processed/label_map.json
"""
import csv
import json
import os
import random
import sys
from collections import defaultdict, Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

IN = "data/text/complaints_labeled.csv"
OUT_DIR = "data/processed"


def main():
    # load
    with open(IN, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {IN}")

    train_only_sources = {"synthetic", "augmented_bt", "chatgpt_synthetic", "pseudo_labeled", "eda_augmented"}
    real = [r for r in rows if r.get("source") not in train_only_sources]
    synth = [r for r in rows if r.get("source") in train_only_sources]
    print(f"  real (split into train/val/test): {len(real)}")
    print(f"  train-only (synthetic + augmented): {len(synth)}")

    # build label map from ALL categories
    categories = sorted({r["category"] for r in rows})
    label_map = {cat: i for i, cat in enumerate(categories)}
    print(f"\nCategories ({len(categories)}):")
    for cat, idx in label_map.items():
        print(f"  {idx} → {cat}")

    for r in rows:
        r["label"] = label_map[r["category"]]

    # stratified split on REAL only
    by_cat = defaultdict(list)
    for r in real:
        by_cat[r["category"]].append(r)

    train, val, test = [], [], []
    for cat, items in by_cat.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    # append ALL synthetic to train
    train.extend(synth)

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    print(f"\nSplit sizes:")
    print(f"  train: {len(train)} (real + synthetic)")
    print(f"  val:   {len(val)} (real only)")
    print(f"  test:  {len(test)} (real only)")

    print("\nTrain per-category:")
    for cat, n in Counter(r["category"] for r in train).most_common():
        print(f"  {cat}: {n}")
    print("\nTest per-category (REAL ONLY):")
    for cat, n in Counter(r["category"] for r in test).most_common():
        print(f"  {cat}: {n}")

    os.makedirs(OUT_DIR, exist_ok=True)
    fields = ["text", "category", "label", "priority", "source"]
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = f"{OUT_DIR}/{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"  wrote {path}")

    map_path = f"{OUT_DIR}/label_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  wrote {map_path}")


if __name__ == "__main__":
    main()
