"""
Evaluate a saved model directory on the held-out test set.
Test set was never used for selection, so this is the honest headline number.

Usage:
    py src/eval_on_test.py models/bakeoff/camelbert-mix_final
    py src/eval_on_test.py models/camelbert_final
"""
from __future__ import annotations

import argparse
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
with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    import json as _json
    NUM_LABELS = len(_json.load(_f))
BATCH_SIZE = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir)).to(device)
    model.eval()

    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()
    print(f"Test rows: {len(texts)}")

    # Optional per-class thresholds
    biases = np.zeros(NUM_LABELS, dtype=np.float32)
    th_path = args.model_dir / "thresholds.json"
    if th_path.exists():
        with open(th_path, encoding="utf-8") as f:
            tj = json.load(f)
        with open(LABEL_MAP_JSON, encoding="utf-8") as f:
            lm = json.load(f)
        for cat, idx in lm.items():
            biases[idx] = float(tj.get(cat, 0.0))
        print(f"Applying per-class biases from {th_path}: {biases.tolist()}")

    all_probs = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=args.max_length, padding=True).to(device)
            logits = model(**enc).logits
            all_probs[i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"  {i + len(batch)}/{len(texts)}")

    preds = (all_probs + biases).argmax(axis=1)
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
    print(f"Test accuracy:    {acc:.4f}")
    print(f"Test weighted F1: {f1_w:.4f}")
    print(f"Test macro F1:    {f1_m:.4f}")
    print(f"Min class F1:     {f1_per_class.min():.4f}")
    print()
    print(report)

    out = args.model_dir / "test_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.model_dir}\n")
        f.write(f"Test accuracy:    {acc:.4f}\n")
        f.write(f"Test weighted F1: {f1_w:.4f}\n")
        f.write(f"Test macro F1:    {f1_m:.4f}\n")
        f.write(f"Min class F1:     {f1_per_class.min():.4f}\n\n")
        f.write(report)
    print(f"Saved test report -> {out}")


if __name__ == "__main__":
    main()
