"""Bootstrap confidence intervals on test-set metrics.

Resamples the test set 1000× with replacement, recomputes metrics each time,
reports 95% CI on accuracy / weighted F1 / macro F1 / per-class F1.

Outputs models/eval_with_ci.txt.
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
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_TXT = ROOT / "models" / "eval_with_ci.txt"

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))
BATCH_SIZE = 64
N_BOOTSTRAP = 1000
RNG = np.random.RandomState(42)


def predict_probs(model_dir, texts, device, max_length):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    out = np.zeros((len(texts), NUM_LABELS), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
            logits = model(**enc).logits
            out[i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
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

    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()
    n = len(labels)
    print(f"Test rows: {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Predict once
    stack = np.zeros((len(model_dirs), n, NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        print(f"  [{k+1}/{len(model_dirs)}] {d.name}")
        stack[k] = predict_probs(d, texts, device, max_length)
    avg = stack.mean(axis=0) + biases
    preds = avg.argmax(axis=1)

    # Point estimates
    acc_pt = accuracy_score(labels, preds)
    wf_pt = f1_score(labels, preds, average="weighted")
    mf_pt = f1_score(labels, preds, average="macro")
    pc_pt = f1_score(labels, preds, average=None, zero_division=0)

    # Bootstrap
    print(f"\nBootstrap ({N_BOOTSTRAP} resamples)...")
    accs, wfs, mfs = [], [], []
    pcs = [[] for _ in range(NUM_LABELS)]
    for b in range(N_BOOTSTRAP):
        idx = RNG.randint(0, n, size=n)
        sub_l = labels[idx]
        sub_p = preds[idx]
        accs.append(accuracy_score(sub_l, sub_p))
        wfs.append(f1_score(sub_l, sub_p, average="weighted", zero_division=0))
        mfs.append(f1_score(sub_l, sub_p, average="macro", zero_division=0))
        per = f1_score(sub_l, sub_p, average=None, labels=list(range(NUM_LABELS)), zero_division=0)
        for i in range(NUM_LABELS):
            pcs[i].append(per[i])

    def ci(arr):
        a = np.sort(np.array(arr))
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    acc_lo, acc_hi = ci(accs)
    wf_lo, wf_hi = ci(wfs)
    mf_lo, mf_hi = ci(mfs)

    lines = ["# Bootstrap 95% CI on test metrics", ""]
    lines.append(f"Test rows: {n}, Bootstrap resamples: {N_BOOTSTRAP}, RNG seed: 42")
    lines.append("")
    lines.append("## Aggregate (95% CI)")
    lines.append("")
    lines.append("| Metric | Point | 95% CI |")
    lines.append("|---|---:|---|")
    lines.append(f"| Accuracy | {acc_pt:.4f} | [{acc_lo:.4f}, {acc_hi:.4f}] |")
    lines.append(f"| Weighted F1 | {wf_pt:.4f} | [{wf_lo:.4f}, {wf_hi:.4f}] |")
    lines.append(f"| Macro F1 | {mf_pt:.4f} | [{mf_lo:.4f}, {mf_hi:.4f}] |")
    lines.append("")
    lines.append("## Per-class F1 (95% CI)")
    lines.append("")
    lines.append("| Category | Point | 95% CI | Width |")
    lines.append("|---|---:|---|---:|")
    for i in range(NUM_LABELS):
        lo, hi = ci(pcs[i])
        lines.append(f"| {names[i]} | {pc_pt[i]:.4f} | [{lo:.4f}, {hi:.4f}] | {hi-lo:.4f} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"- **Headline accuracy: {acc_pt:.4f} ± {(acc_hi-acc_lo)/2:.4f}** (95% CI half-width)")
    lines.append(f"- **Macro F1: {mf_pt:.4f} ± {(mf_hi-mf_lo)/2:.4f}**")
    lines.append("")
    # Significance vs prior no-EDA baseline
    PRIOR_ACC = 0.9486  # ensemble before EDA boost on دقة الطلب + عامة
    if acc_lo <= PRIOR_ACC <= acc_hi:
        lines.append(f"Prior baseline accuracy ({PRIOR_ACC:.4f}) falls **inside** the 95% CI — accuracy improvement is not distinguishable from noise on this sample size. Macro F1 improvement is more credible (per-class metric).")
    else:
        lines.append(f"Prior baseline accuracy ({PRIOR_ACC:.4f}) falls **outside** the 95% CI — accuracy improvement is statistically significant.")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
