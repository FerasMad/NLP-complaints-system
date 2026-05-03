"""
Targeted EDA augmentation for specific weak categories.

Augments existing high-quality (real / chatgpt / pseudo) rows for the named
categories using:
  - Random word swap
  - Random word deletion
  - Random word insertion (intensifiers)
  - Sentence-fragment extraction (slice random 5-12 word window)

Marks new rows with source="eda_augmented" → train-only (per split_dataset.py).

Usage:
    py src/eda_augment_classes.py --categories "دقة الطلب" "عامة" --target-each 1500
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "text" / "complaints_labeled.csv"

# Sources we trust as input for augmentation
GOOD_SOURCES = {"production", "play_store", "res1", "chatgpt_synthetic", "augmented_bt", "pseudo_labeled"}

INTENSIFIERS = ["مرره", "مره", "جدا", "جدا جدا", "والله", "يا اخي", "بصراحه", "للاسف", "نهائي", "ابدا", "كثير"]
DROPPABLE = {"و", "في", "من", "على", "ال", "بس"}


def existing_texts():
    s = set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            s.add(r.get("text", "").strip())
    return s


def load_pool(category):
    rows = []
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("category") != category:
                continue
            if r.get("source") not in GOOD_SOURCES:
                continue
            t = r.get("text", "").strip()
            if t and len(t.split()) >= 3:
                rows.append(t)
    return rows


def random_swap(words):
    if len(words) < 2:
        return words
    out = list(words)
    a, b = random.sample(range(len(out)), 2)
    out[a], out[b] = out[b], out[a]
    return out


def random_delete(words, p=0.10):
    if len(words) < 4:
        return words
    out = []
    for w in words:
        if w in DROPPABLE and random.random() < p:
            continue
        out.append(w)
    return out


def random_insert(words):
    if not words:
        return words
    out = list(words)
    pos = random.randint(0, len(out))
    out.insert(pos, random.choice(INTENSIFIERS))
    return out


def fragment(words):
    if len(words) < 8:
        return words
    n = random.randint(5, min(12, len(words) - 1))
    start = random.randint(0, len(words) - n)
    return words[start : start + n]


def augment(text):
    words = text.split()
    op = random.choice(["swap", "delete", "insert", "fragment"])
    if op == "swap":
        new_words = random_swap(words)
    elif op == "delete":
        new_words = random_delete(words)
    elif op == "insert":
        new_words = random_insert(words)
    else:
        new_words = fragment(words)
    return " ".join(new_words).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--target-each", type=int, default=1500)
    args = ap.parse_args()

    seen = existing_texts()
    print(f"Existing texts: {len(seen)}")

    all_new = []
    for cat in args.categories:
        pool = load_pool(cat)
        print(f"\n{cat}: {len(pool)} source rows from {GOOD_SOURCES}")
        if not pool:
            print(f"  no rows in good sources — skipping")
            continue

        new_rows = []
        attempts = 0
        max_attempts = args.target_each * 30
        while len(new_rows) < args.target_each and attempts < max_attempts:
            attempts += 1
            src = random.choice(pool)
            out = augment(src)
            if not out or len(out) < 5 or out in seen:
                continue
            seen.add(out)
            new_rows.append({
                "text": out,
                "category": cat,
                "priority": "متوسطة",
                "source": "eda_augmented",
            })
        print(f"  generated {len(new_rows)} ({attempts} attempts)")
        all_new.extend(new_rows)

    if not all_new:
        print("\nNothing new. Exiting.")
        return

    # Append
    with open(LABELED, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(all_new)
    print(f"\nAppended {len(all_new)} rows -> {LABELED}")


if __name__ == "__main__":
    main()
