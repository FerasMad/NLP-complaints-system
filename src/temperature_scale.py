"""Temperature scaling for ensemble calibration.

Optimizes a single scalar T on val to minimize NLL of ensemble logits.
Applies T at inference: probs = softmax(logits / T).

Outputs:
  - models/ensemble_final/temperature.json — the optimal T
  - models/temperature_scaling_report.txt — before/after ECE

Usage:
    py src/temperature_scale.py
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
import torch.nn as nn
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_T = ROOT / "models" / "ensemble_final" / "temperature.json"
OUT_RPT = ROOT / "models" / "temperature_scaling_report.txt"
BATCH_SIZE = 64

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))


def get_logits(model_dir, texts, device, max_length):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    out = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            logits = model(**enc).logits
            out[i : i + len(batch)] = logits.cpu().numpy()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def ece(probs, labels, n_bins=10):
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        e += mask.sum() / len(labels) * abs(bin_acc - bin_conf)
    return e


def main():
    with open(ENSEMBLE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    model_dirs = [ROOT / m for m in cfg["models"]]
    max_length = int(cfg.get("max_length", 192))

    val_df = pd.read_csv(VAL_CSV)[["text", "label"]]
    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    val_texts = val_df["text"].astype(str).tolist()
    val_labels = val_df["label"].to_numpy()
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_stack = np.zeros((len(model_dirs), len(val_texts), NUM_LABELS), dtype=np.float32)
    test_stack = np.zeros((len(model_dirs), len(test_texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        print(f"  [{k+1}/{len(model_dirs)}] {d.name}")
        val_stack[k] = get_logits(d, val_texts, device, max_length)
        test_stack[k] = get_logits(d, test_texts, device, max_length)

    # Average logits then apply T
    val_logits = val_stack.mean(axis=0)
    test_logits = test_stack.mean(axis=0)

    # Untuned (T=1)
    val_probs_T1 = np.exp(val_logits) / np.exp(val_logits).sum(axis=1, keepdims=True)
    test_probs_T1 = np.exp(test_logits) / np.exp(test_logits).sum(axis=1, keepdims=True)
    ece_val_T1 = ece(val_probs_T1, val_labels)
    ece_test_T1 = ece(test_probs_T1, test_labels)
    acc_test_T1 = accuracy_score(test_labels, test_probs_T1.argmax(axis=1))

    # Optimize T to minimize NLL on val
    print("\nOptimizing temperature on val...")
    val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long)
    T = nn.Parameter(torch.ones(1) * 1.0)
    optimizer = torch.optim.LBFGS([T], lr=0.05, max_iter=100)
    nll_loss = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_loss(val_logits_t / T, val_labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_opt = float(T.item())
    print(f"Optimal T = {T_opt:.4f}")

    # Re-evaluate
    val_probs_T = torch.softmax(val_logits_t / T_opt, dim=-1).numpy()
    test_probs_T = torch.softmax(torch.tensor(test_logits) / T_opt, dim=-1).numpy()
    ece_val_T = ece(val_probs_T, val_labels)
    ece_test_T = ece(test_probs_T, test_labels)
    acc_test_T = accuracy_score(test_labels, test_probs_T.argmax(axis=1))

    lines = ["# Temperature scaling", ""]
    lines.append(f"Optimal T (on val NLL): {T_opt:.4f}")
    lines.append(f"  T < 1 → makes model MORE confident (was under-confident)")
    lines.append(f"  T > 1 → makes model LESS confident (was over-confident)")
    lines.append("")
    lines.append("|  | Val ECE | Test ECE | Test Acc |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| T=1 (untuned) | {ece_val_T1:.4f} | {ece_test_T1:.4f} | {acc_test_T1:.4f} |")
    lines.append(f"| T={T_opt:.3f} (tuned) | {ece_val_T:.4f} | {ece_test_T:.4f} | {acc_test_T:.4f} |")
    lines.append("")
    if ece_test_T < ece_test_T1:
        lines.append(f"Temperature scaling **reduced test ECE** from {ece_test_T1:.4f} → {ece_test_T:.4f} ({(ece_test_T1-ece_test_T)/ece_test_T1*100:.1f}% improvement). Apply T at inference.")
    else:
        lines.append(f"Temperature scaling didn't help test ECE (val→test gap). Keep T=1.")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_RPT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RPT.write_text(txt + "\n", encoding="utf-8")
    print(f"\nReport -> {OUT_RPT}")

    # Save T to JSON for inference use
    OUT_T.write_text(json.dumps({"temperature": T_opt, "fitted_on": "val", "method": "LBFGS NLL"}, indent=2), encoding="utf-8")
    print(f"Temperature -> {OUT_T}")


if __name__ == "__main__":
    main()
