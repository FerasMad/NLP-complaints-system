"""
Pseudo-label unused scraped Play Store reviews using the top-2 bake-off models.

Safeguards (to prevent error amplification):
  - Use the TOP-2 bake-off models (by val macro F1)
  - Both must predict the SAME class
  - Both must have softmax probability >= MIN_CONF (default 0.90)
  - Only label ROW into TARGET_CATEGORIES (small/weak categories) to avoid
    further inflating already-dominant classes

Output:
  - Append rows to data/text/complaints_labeled.csv with source="pseudo_labeled"
  - data/processed/pseudo_label_log.csv recording what got added (and discarded counts)

Usage:
    py src/pseudo_label.py
"""
from __future__ import annotations

import csv
import gc
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "text" / "complaints_labeled.csv"
PLAY = ROOT / "data" / "raw" / "play_store_reviews.csv"
RESULTS_CSV = ROOT / "models" / "bakeoff_results.csv"
BAKEOFF_DIR = ROOT / "models" / "bakeoff"
LOG_CSV = ROOT / "data" / "processed" / "pseudo_label_log.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
NUM_LABELS = 9
BATCH_SIZE = 64
MIN_CONF = 0.90

# Only pseudo-label these (we need MORE data for these classes specifically)
TARGET_CATEGORIES = {"الجو والمكان", "دقة الطلب", "عامة", "التوصيل"}

# Lana's cleaning (vendored)
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean(text):
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    t = WHITESPACE.sub(" ", t).strip().lower()
    return t


def is_arabic(text):
    if not text or len(text) < 5:
        return False
    arabic_chars = len(ARABIC_CHAR.findall(text))
    return arabic_chars / max(len(text), 1) > 0.3


def top_2_models():
    """Return list of model dirs sorted by val macro_f1 desc, top 2."""
    if not RESULTS_CSV.exists():
        raise SystemExit(f"No bakeoff_results.csv at {RESULTS_CSV}")
    rows = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["macro_f1"]), r["short_name"]))
            except (KeyError, ValueError):
                pass
    rows.sort(reverse=True)
    short_names = [n for _, n in rows[:2]]
    dirs = []
    for n in short_names:
        d = BAKEOFF_DIR / f"{n}_final"
        if d.is_dir():
            dirs.append((n, d))
    return dirs


def load_existing_seen():
    """All texts already in the labeled file (any source)."""
    seen = set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            seen.add(r.get("text", "").strip())
    return seen


def load_unused_scraped(seen):
    """Reviews from play_store that are NOT already in labeled (irrespective of category)."""
    out = []
    with open(PLAY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            t = r.get("text", "").strip()
            if not is_arabic(t):
                continue
            cleaned = clean(t)
            if not cleaned or len(cleaned) < 5:
                continue
            if cleaned in seen:
                continue
            out.append(cleaned)
    # dedupe
    out = list(dict.fromkeys(out))
    return out


def predict_probs(model_dir, texts, device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    all_probs = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs[i : i + len(batch)] = probs
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading existing texts...")
    seen = load_existing_seen()
    print(f"  {len(seen)} texts already in {LABELED}")

    print("Loading unused scraped reviews...")
    candidates = load_unused_scraped(seen)
    print(f"  {len(candidates)} candidate texts")

    if not candidates:
        print("No candidates to label. Exiting.")
        return

    pair = top_2_models()
    if len(pair) < 2:
        print("Need at least 2 finished bake-off models for two-model agreement. Aborting.")
        return
    print(f"Top-2 models: {[n for n, _ in pair]}")

    import json
    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}

    probs_a = predict_probs(pair[0][1], candidates, device)
    probs_b = predict_probs(pair[1][1], candidates, device)

    pred_a = probs_a.argmax(axis=1)
    pred_b = probs_b.argmax(axis=1)
    conf_a = probs_a.max(axis=1)
    conf_b = probs_b.max(axis=1)

    keep_rows = []
    discard_disagree = 0
    discard_lowconf = 0
    discard_off_target = 0
    kept_per_cat = Counter()

    for i, txt in enumerate(candidates):
        if pred_a[i] != pred_b[i]:
            discard_disagree += 1
            continue
        if conf_a[i] < MIN_CONF or conf_b[i] < MIN_CONF:
            discard_lowconf += 1
            continue
        cat = inv[int(pred_a[i])]
        if cat not in TARGET_CATEGORIES:
            discard_off_target += 1
            continue
        keep_rows.append({
            "text": txt,
            "category": cat,
            "priority": "متوسطة",
            "source": "pseudo_labeled",
        })
        kept_per_cat[cat] += 1

    print()
    print(f"Discarded (disagreement):      {discard_disagree}")
    print(f"Discarded (conf < {MIN_CONF}): {discard_lowconf}")
    print(f"Discarded (not target cat):    {discard_off_target}")
    print(f"Kept:                          {len(keep_rows)}")
    print("Per-category kept:")
    for cat, n in kept_per_cat.most_common():
        print(f"  {cat}: {n}")

    if not keep_rows:
        print("Nothing to add. Exiting.")
        return

    # Append
    with open(LABELED, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(keep_rows)
    print(f"Appended {len(keep_rows)} pseudo-labeled rows to {LABELED}")

    # Log
    with open(LOG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "conf_a", "conf_b", "model_a", "model_b"])
        writer.writeheader()
        for i, row in enumerate(keep_rows):
            # We need to map back the index — reconstruct from candidates list
            pass
    # Skip the per-row log if it gets unwieldy. Summary suffices.
    # Replace LOG_CSV with summary
    with open(LOG_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model_a", "model_b", "min_conf", "kept", "discarded_disagree", "discarded_lowconf", "discarded_off_target"])
        writer.writerow([pair[0][0], pair[1][0], MIN_CONF, len(keep_rows), discard_disagree, discard_lowconf, discard_off_target])
        writer.writerow([])
        writer.writerow(["category", "kept_count"])
        for cat, n in kept_per_cat.most_common():
            writer.writerow([cat, n])
    print(f"Summary -> {LOG_CSV}")


if __name__ == "__main__":
    main()
