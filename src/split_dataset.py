"""
Encode category labels and create stratified 70/15/15 train/val/test splits.

Critical: synthetic rows are EXCLUDED from val and test sets.
They are appended only to train. This prevents the model from being
evaluated on data it has effectively memorized.

Schema awareness: pass --schema-version to control which categories are
allowed. Default '8class' is the production schema (drops the v3 ambience
class if it appears in input). '9class_ambience' is the v5 experiment
that keeps الجو والمكان.

Inputs:  data/text/complaints_labeled.csv (with `source` column)
Outputs: data/processed/{train,val,test}.csv  +  data/processed/label_map.json
"""
import argparse
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

AMBIENCE_LABEL = "الجو والمكان"

# Train-only sources: ambience_synthetic and ambience_eda_augmented are the
# v5-specific ones; everything else predates this experiment.
TRAIN_ONLY_SOURCES = {
    "synthetic", "augmented_bt", "chatgpt_synthetic",
    "pseudo_labeled", "eda_augmented",
    "ambience_synthetic", "ambience_eda_augmented",
}

# Per schema, the categories we expect to see in the final label map.
SCHEMA_CATEGORIES = {
    "8class": {
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    },
    "9class_ambience": {
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
        AMBIENCE_LABEL,
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--schema-version",
        choices=sorted(SCHEMA_CATEGORIES.keys()),
        default="8class",
        help="Which class schema to enforce. Default '8class' (production); "
             "use '9class_ambience' for the v5 experiment.",
    )
    p.add_argument("--input", default=IN, help=f"Input labeled CSV (default: {IN})")
    p.add_argument("--output-dir", default=OUT_DIR, help=f"Output dir (default: {OUT_DIR})")
    args = p.parse_args()

    allowed = SCHEMA_CATEGORIES[args.schema_version]

    # load
    with open(args.input, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Schema: {args.schema_version} ({len(allowed)} categories allowed)")

    # Drop rows whose category isn't in the active schema. For 8class, this
    # silently drops any leftover ambience rows. For 9class_ambience, all
    # 8 production categories + ambience are allowed.
    before = len(rows)
    rows = [r for r in rows if r["category"] in allowed]
    dropped = before - len(rows)
    if dropped:
        print(f"  dropped {dropped} rows whose category is not in the {args.schema_version} schema")

    real = [r for r in rows if r.get("source") not in TRAIN_ONLY_SOURCES]
    synth = [r for r in rows if r.get("source") in TRAIN_ONLY_SOURCES]
    print(f"  real (split into train/val/test): {len(real)}")
    print(f"  train-only (synthetic + augmented): {len(synth)}")

    # build label map from categories actually present, in the schema's canonical
    # sort order so the map is reproducible across runs
    present_categories = sorted({r["category"] for r in rows})
    label_map = {cat: i for i, cat in enumerate(present_categories)}
    print(f"\nCategories ({len(label_map)}):")
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

    os.makedirs(args.output_dir, exist_ok=True)
    fields = ["text", "category", "label", "priority", "source"]
    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = f"{args.output_dir}/{name}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print(f"  wrote {path}")

    map_path = f"{args.output_dir}/label_map.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  wrote {map_path}")

    # Also write a tiny manifest so downstream code can verify the schema
    manifest = {
        "schema_version": args.schema_version,
        "num_classes": len(label_map),
        "categories": list(label_map.keys()),
        "source_counts": dict(Counter(r.get("source", "?") for r in rows)),
    }
    manifest_path = f"{args.output_dir}/split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"  wrote {manifest_path}")


if __name__ == "__main__":
    main()
