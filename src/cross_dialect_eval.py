"""Cross-dialect generalization eval.

Reads `data/canary/cross_dialect.csv` (Egyptian / Levantine / MSA / Saudi
baseline samples) and reports per-dialect accuracy.

Outputs models/cross_dialect_report.txt.

The canary set is small (~30 samples) and AI-generated for first-pass
measurement. For publishable numbers, expand to ~50 per dialect with
native-speaker review.
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
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
CANARY_CSV = ROOT / "data" / "canary" / "cross_dialect.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_TXT = ROOT / "models" / "cross_dialect_report.txt"
BATCH_SIZE = 32

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))


def predict_argmax(model_dirs, texts, device, max_length, biases):
    stack = np.zeros((len(model_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        tokenizer = AutoTokenizer.from_pretrained(str(d))
        model = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
                logits = model(**enc).logits
                stack[k, i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return (stack.mean(axis=0) + biases).argmax(axis=1)


def main():
    if not CANARY_CSV.exists():
        print(f"ERROR: {CANARY_CSV} not found")
        sys.exit(1)

    canary = pd.read_csv(CANARY_CSV)
    print(f"Canary samples: {len(canary)}")
    print(f"Per-dialect:")
    for d, n in canary["dialect"].value_counts().items():
        print(f"  {d}: {n}")

    with open(ENSEMBLE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    model_dirs = [ROOT / m for m in cfg["models"]]
    max_length = int(cfg.get("max_length", 192))
    biases_dict = cfg.get("tuned_biases", {})

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    biases = np.zeros(NUM_LABELS, dtype=np.float32)
    for cat, idx in label_map.items():
        biases[idx] = float(biases_dict.get(cat, 0.0))

    canary["gold_label"] = canary["gold_category"].map(label_map)
    if canary["gold_label"].isna().any():
        bad = canary[canary["gold_label"].isna()]
        print(f"WARN: {len(bad)} canary rows have unknown category")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    texts = canary["text"].tolist()
    preds = predict_argmax(model_dirs, texts, device, max_length, biases)

    inv = {v: k for k, v in label_map.items()}
    canary["pred_category"] = [inv[int(p)] for p in preds]
    canary["correct"] = canary["pred_category"] == canary["gold_category"]

    # Per-dialect accuracy
    lines = ["# Cross-dialect generalization", ""]
    lines.append(f"Canary set: {len(canary)} samples ({CANARY_CSV.name})")
    lines.append(f"Note: first-pass measurement; expand with native-speaker review before publication.")
    lines.append("")
    lines.append("| Dialect | N | Accuracy |")
    lines.append("|---|---:|---:|")
    overall_acc = canary["correct"].mean()
    for d in canary["dialect"].unique():
        sub = canary[canary["dialect"] == d]
        acc = sub["correct"].mean()
        lines.append(f"| {d} | {len(sub)} | {acc:.2%} |")
    lines.append(f"| **Overall** | **{len(canary)}** | **{overall_acc:.2%}** |")
    lines.append("")

    # Show misclassifications
    misses = canary[~canary["correct"]]
    if len(misses) > 0:
        lines.append("## Misclassifications")
        lines.append("")
        for _, r in misses.iterrows():
            lines.append(f"- **{r['dialect']}** \"{r['text']}\" → predicted {r['pred_category']} (gold: {r['gold_category']}). {r.get('note', '')}")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
