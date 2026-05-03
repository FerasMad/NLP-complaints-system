"""
Per-source evaluation on the test set.

Splits test by `source` column (production vs play_store vs res1) and reports
per-source accuracy + macro F1. Diagnoses generalization across data origins.

Outputs models/per_source_eval.txt.
"""
from __future__ import annotations

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
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_TXT = ROOT / "models" / "per_source_eval.txt"
BATCH_SIZE = 64

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))


def predict_probs(model_dir, texts, device, max_length):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    all_probs = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            logits = model(**enc).logits
            all_probs[i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_probs


def main():
    test_df = pd.read_csv(TEST_CSV)
    if "source" not in test_df.columns:
        print("ERROR: test.csv has no 'source' column")
        sys.exit(1)
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()
    sources = test_df["source"].astype(str).to_numpy()
    print(f"Test rows: {len(texts)}")
    print("Source distribution:")
    for src, n in test_df["source"].value_counts().items():
        print(f"  {src}: {n}")

    with open(ENSEMBLE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    model_dirs = [ROOT / m for m in cfg["models"]]
    max_length = int(cfg.get("max_length", 192))
    biases_dict = cfg.get("tuned_biases", {})

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    names = [inv[i] for i in range(NUM_LABELS)]

    biases = np.zeros(NUM_LABELS, dtype=np.float32)
    for cat, idx in label_map.items():
        biases[idx] = float(biases_dict.get(cat, 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    stack = np.zeros((len(model_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        print(f"  [{k+1}/{len(model_dirs)}] {d.name}")
        stack[k] = predict_probs(d, texts, device, max_length)

    avg = stack.mean(axis=0) + biases
    preds = avg.argmax(axis=1)

    lines = ["# Per-source evaluation", ""]
    lines.append(f"Total test rows: {len(labels)}")
    lines.append(f"Overall accuracy: {accuracy_score(labels, preds):.4f}")
    lines.append(f"Overall weighted F1: {f1_score(labels, preds, average='weighted'):.4f}")
    lines.append(f"Overall macro F1:    {f1_score(labels, preds, average='macro'):.4f}")
    lines.append("")
    lines.append("## Per-source breakdown")
    lines.append("")
    lines.append("| Source | Rows | Accuracy | Weighted F1 | Macro F1 |")
    lines.append("|---|---:|---:|---:|---:|")

    unique_sources = sorted(test_df["source"].unique())
    for src in unique_sources:
        mask = sources == src
        if mask.sum() < 2:
            continue
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        if len(np.unique(sub_labels)) < 2:
            # only one class — F1 not meaningful
            acc = accuracy_score(sub_labels, sub_preds)
            lines.append(f"| {src} | {int(mask.sum())} | {acc:.4f} | (single class) | (single class) |")
            continue
        acc = accuracy_score(sub_labels, sub_preds)
        wf = f1_score(sub_labels, sub_preds, average="weighted", zero_division=0)
        mf = f1_score(sub_labels, sub_preds, average="macro", zero_division=0)
        lines.append(f"| {src} | {int(mask.sum())} | {acc:.4f} | {wf:.4f} | {mf:.4f} |")

    lines.append("")
    lines.append("## Per-source per-class F1")
    lines.append("")
    for src in unique_sources:
        mask = sources == src
        if mask.sum() < 5:
            continue
        sub_labels = labels[mask]
        sub_preds = preds[mask]
        present = sorted(set(sub_labels))
        if len(present) < 2:
            continue
        present_names = [names[i] for i in present]
        rep = classification_report(sub_labels, sub_preds, labels=present, target_names=present_names, zero_division=0, digits=3)
        lines.append(f"### {src} ({int(mask.sum())} rows)")
        lines.append("```")
        lines.append(rep)
        lines.append("```")
        lines.append("")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
