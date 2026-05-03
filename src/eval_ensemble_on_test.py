"""
Evaluate the bake-off ensemble on the held-out test set.
Mirrors src/ensemble_eval.py but uses test.csv.

Usage:
    py src/eval_ensemble_on_test.py
    py src/eval_ensemble_on_test.py --weighted
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
BAKEOFF_DIR = ROOT / "models" / "bakeoff"
RESULTS_CSV = ROOT / "models" / "bakeoff_results.csv"
ENSEMBLE_DIR = ROOT / "models" / "ensemble"
with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    import json as _json
    NUM_LABELS = len(_json.load(_f))
BATCH_SIZE = 64


def list_final_models():
    return sorted([p for p in BAKEOFF_DIR.glob("*_final") if p.is_dir()])


def model_macro_f1(short_name):
    if not RESULTS_CSV.exists():
        return 1.0
    rows = []
    with open(RESULTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("short_name") == short_name:
                try:
                    rows.append(float(row["macro_f1"]))
                except (KeyError, ValueError):
                    pass
    if not rows:
        return 1.0
    return rows[-1]  # use latest if multiple


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
            all_probs[i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--dirs", nargs="+", default=None, help="Explicit list of model dirs (overrides auto-discovery)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()
    print(f"Test rows: {len(texts)}")

    if args.dirs:
        final_dirs = [Path(d) for d in args.dirs]
    else:
        final_dirs = list_final_models()
    print(f"Models: {[d.name for d in final_dirs]}")

    stack = np.zeros((len(final_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    weights = np.ones(len(final_dirs), dtype=np.float32)

    for k, d in enumerate(final_dirs):
        short = d.name.replace("_final", "")
        print(f"\n[{k+1}/{len(final_dirs)}] {short}")
        stack[k] = predict_probs(d, texts, device)
        if args.weighted:
            weights[k] = model_macro_f1(short)

    weights = weights / weights.sum()

    avg_probs = (stack * weights[:, None, None]).sum(axis=0)
    preds = avg_probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    f1_w = f1_score(labels, preds, average="weighted")
    f1_m = f1_score(labels, preds, average="macro")
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    names = [inv[i] for i in range(NUM_LABELS)]
    report = classification_report(labels, preds, target_names=names, zero_division=0, digits=3)

    print()
    print(f"Ensemble TEST accuracy:    {acc:.4f}")
    print(f"Ensemble TEST weighted F1: {f1_w:.4f}")
    print(f"Ensemble TEST macro F1:    {f1_m:.4f}")
    print(f"Min class F1:              {f1_per_class.min():.4f}")
    print()
    print(report)

    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(ENSEMBLE_DIR / "test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Ensemble of: {[d.name for d in final_dirs]}\n")
        f.write(f"Weighted: {args.weighted}\n")
        f.write(f"Per-model weights: {weights.tolist()}\n\n")
        f.write(f"Test accuracy:    {acc:.4f}\n")
        f.write(f"Test weighted F1: {f1_w:.4f}\n")
        f.write(f"Test macro F1:    {f1_m:.4f}\n")
        f.write(f"Min class F1:     {f1_per_class.min():.4f}\n\n")
        f.write(report)
    print(f"Saved -> {ENSEMBLE_DIR}/test_report.txt")


if __name__ == "__main__":
    main()
