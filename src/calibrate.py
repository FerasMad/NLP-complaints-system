"""
Calibration analysis on the test set using the winning ensemble.

Outputs:
  - models/calibration_report.txt — text summary (ECE, per-bucket accuracy)
  - models/calibration_plot.png   — reliability diagram (predicted-confidence vs actual-accuracy)

Tells you what confidence threshold makes the model trustworthy.
E.g., "model is X% accurate when confidence ≥ 0.8".

Usage:
    py src/calibrate.py
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_REPORT = ROOT / "models" / "calibration_report.txt"
OUT_PLOT = ROOT / "models" / "calibration_plot.png"
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

    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, test rows: {len(texts)}")

    stack = np.zeros((len(model_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        print(f"  [{k+1}/{len(model_dirs)}] {d.name}")
        stack[k] = predict_probs(d, texts, device, max_length)

    avg = stack.mean(axis=0) + biases
    preds = avg.argmax(axis=1)
    confidences = avg.max(axis=1)
    correct = (preds == labels).astype(np.float32)

    # Reliability diagram
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc[i] = correct[mask].mean()
            bin_conf[i] = confidences[mask].mean()
            bin_count[i] = mask.sum()

    # Expected Calibration Error: weighted avg of |bin_acc - bin_conf|
    ece = np.sum(bin_count / bin_count.sum() * np.abs(bin_acc - bin_conf))

    # Trustworthy thresholds: at confidence >= T, what's the accuracy?
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    trust_table = []
    for T in thresholds:
        mask = confidences >= T
        if mask.sum() > 0:
            cov = mask.mean()
            acc_at = correct[mask].mean()
        else:
            cov = 0.0
            acc_at = float("nan")
        trust_table.append((T, cov, acc_at, int(mask.sum())))

    # Print + save report
    lines = []
    lines.append(f"Calibration analysis — 8-class ensemble on test set ({len(texts)} rows)")
    lines.append(f"Overall accuracy: {correct.mean():.4f}")
    lines.append(f"Expected Calibration Error (ECE, 10 bins): {ece:.4f}")
    lines.append("")
    lines.append("Reliability table (10 bins):")
    lines.append(f"  {'bin':>10s}  {'count':>8s}  {'avg_conf':>10s}  {'accuracy':>10s}  {'gap':>8s}")
    for i in range(n_bins):
        if bin_count[i] == 0:
            continue
        gap = bin_acc[i] - bin_conf[i]
        lines.append(f"  {bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}   {bin_count[i]:>8d}  {bin_conf[i]:>10.4f}  {bin_acc[i]:>10.4f}  {gap:>+8.4f}")
    lines.append("")
    lines.append("Trustworthy-prediction table (model accuracy at each confidence threshold):")
    lines.append(f"  {'threshold':>10s}  {'coverage':>10s}  {'accuracy':>10s}  {'count':>8s}")
    for T, cov, acc, n in trust_table:
        lines.append(f"  >= {T:.2f}     {cov:>10.4f}  {acc:>10.4f}  {n:>8d}")
    lines.append("")
    lines.append("Interpretation:")
    if ece < 0.03:
        lines.append("  Model is WELL CALIBRATED (ECE < 0.03). Trust the confidence values directly.")
    elif ece < 0.06:
        lines.append("  Model is REASONABLY calibrated (0.03 ≤ ECE < 0.06).")
    else:
        lines.append("  Model is POORLY calibrated (ECE ≥ 0.06). Consider temperature scaling.")
    over_under = "over-confident" if (bin_acc - bin_conf).mean() < -0.01 else ("under-confident" if (bin_acc - bin_conf).mean() > 0.01 else "well-balanced")
    lines.append(f"  Direction: model is {over_under} (avg gap = {(bin_acc - bin_conf).mean():+.4f}).")

    report = "\n".join(lines)
    print(report)

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nSaved -> {OUT_REPORT}")

    # Plot reliability diagram (matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.6)
        valid = bin_count > 0
        ax.bar(bin_centers[valid], bin_acc[valid], width=0.09, alpha=0.7, edgecolor="black", label="Actual accuracy")
        ax.scatter(bin_conf[valid], bin_acc[valid], color="red", zorder=5, label="Per-bin (avg conf, accuracy)")
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Reliability diagram — Arabic Complaints Ensemble\nECE = {ece:.4f}, overall acc = {correct.mean():.4f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOT, dpi=120)
        print(f"Saved -> {OUT_PLOT}")
    except Exception as e:
        print(f"  ! plot failed: {e}")


if __name__ == "__main__":
    main()
