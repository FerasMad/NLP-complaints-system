"""
Threshold-tune the ensemble: per-class additive biases on softmax probs,
optimized for val ACCURACY (not macro F1, since user wants >94% accuracy).

Then apply tuned biases on test and report.

Usage:
    py src/tune_ensemble.py --dirs models/bakeoff/X_final ...
"""
from __future__ import annotations

import argparse
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
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_DIR = ROOT / "models" / "ensemble"
with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    import json as _json
    NUM_LABELS = len(_json.load(_f))
BATCH_SIZE = 64


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


def search_biases(probs, labels, metric="accuracy"):
    """Coordinate descent on per-class additive bias."""
    n_classes = probs.shape[1]
    bias = np.zeros(n_classes, dtype=np.float32)

    def eval_bias(b):
        adjusted = probs + b
        preds = adjusted.argmax(axis=1)
        if metric == "accuracy":
            return accuracy_score(labels, preds)
        elif metric == "macro_f1":
            return f1_score(labels, preds, average="macro")
        else:
            raise ValueError(metric)

    base = eval_bias(bias)
    print(f"Baseline {metric}: {base:.4f}")

    grid = np.arange(-0.30, 0.31, 0.01)
    for it in range(5):
        improved = False
        for c in range(n_classes):
            best_b = bias[c]
            best_f = eval_bias(bias)
            for v in grid:
                trial = bias.copy()
                trial[c] = v
                f = eval_bias(trial)
                if f > best_f + 1e-6:
                    best_f = f
                    best_b = v
            if best_b != bias[c]:
                bias[c] = best_b
                improved = True
        cur = eval_bias(bias)
        print(f"  iter {it+1}: {metric} {cur:.4f}, biases {bias.round(2).tolist()}")
        if not improved:
            break
    return bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--metric", choices=["accuracy", "macro_f1"], default="accuracy")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_df = pd.read_csv(VAL_CSV)[["text", "label"]]
    val_texts = val_df["text"].astype(str).tolist()
    val_labels = val_df["label"].to_numpy()

    test_df = pd.read_csv(TEST_CSV)[["text", "label"]]
    test_texts = test_df["text"].astype(str).tolist()
    test_labels = test_df["label"].to_numpy()

    # Get probs from each model on both val and test
    val_stack = np.zeros((len(args.dirs), len(val_texts), NUM_LABELS), dtype=np.float32)
    test_stack = np.zeros((len(args.dirs), len(test_texts), NUM_LABELS), dtype=np.float32)

    for k, d in enumerate(args.dirs):
        print(f"\n[{k+1}/{len(args.dirs)}] {d}")
        val_stack[k] = predict_probs(Path(d), val_texts, device)
        test_stack[k] = predict_probs(Path(d), test_texts, device)

    # Average (uniform)
    val_probs = val_stack.mean(axis=0)
    test_probs = test_stack.mean(axis=0)

    print(f"\n=== Tuning biases on VAL for {args.metric} ===")
    biases = search_biases(val_probs, val_labels, metric=args.metric)

    # Evaluate on test
    test_preds_untuned = test_probs.argmax(axis=1)
    test_preds_tuned = (test_probs + biases).argmax(axis=1)

    val_preds_untuned = val_probs.argmax(axis=1)
    val_preds_tuned = (val_probs + biases).argmax(axis=1)

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    names = [inv[i] for i in range(NUM_LABELS)]

    def metrics(labels, preds, label):
        acc = accuracy_score(labels, preds)
        f1_w = f1_score(labels, preds, average="weighted")
        f1_m = f1_score(labels, preds, average="macro")
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
        return acc, f1_w, f1_m, f1_per_class.min()

    v_acc_u, v_w_u, v_m_u, v_min_u = metrics(val_labels, val_preds_untuned, "val")
    v_acc_t, v_w_t, v_m_t, v_min_t = metrics(val_labels, val_preds_tuned, "val")
    t_acc_u, t_w_u, t_m_u, t_min_u = metrics(test_labels, test_preds_untuned, "test")
    t_acc_t, t_w_t, t_m_t, t_min_t = metrics(test_labels, test_preds_tuned, "test")

    print()
    print(f"VAL  untuned: acc={v_acc_u:.4f}  wF1={v_w_u:.4f}  mF1={v_m_u:.4f}  min={v_min_u:.4f}")
    print(f"VAL  tuned  : acc={v_acc_t:.4f}  wF1={v_w_t:.4f}  mF1={v_m_t:.4f}  min={v_min_t:.4f}")
    print(f"TEST untuned: acc={t_acc_u:.4f}  wF1={t_w_u:.4f}  mF1={t_m_u:.4f}  min={t_min_u:.4f}")
    print(f"TEST tuned  : acc={t_acc_t:.4f}  wF1={t_w_t:.4f}  mF1={t_m_t:.4f}  min={t_min_t:.4f}")

    print()
    print(classification_report(test_labels, test_preds_tuned, target_names=names, zero_division=0, digits=3))

    # Save tuned biases + probs
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ENSEMBLE_DIR / "tuned_val_probs.npy", val_probs)
    np.save(ENSEMBLE_DIR / "tuned_test_probs.npy", test_probs)
    biases_dict = {names[i]: float(biases[i]) for i in range(NUM_LABELS)}
    with open(ENSEMBLE_DIR / "tuned_biases.json", "w", encoding="utf-8") as f:
        json.dump(biases_dict, f, ensure_ascii=False, indent=2)
    with open(ENSEMBLE_DIR / "tuned_test_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Ensemble of: {args.dirs}\n")
        f.write(f"Tuning metric: {args.metric}\n")
        f.write(f"Tuned biases: {biases_dict}\n\n")
        f.write(f"VAL  untuned: acc={v_acc_u:.4f}  wF1={v_w_u:.4f}  mF1={v_m_u:.4f}  min={v_min_u:.4f}\n")
        f.write(f"VAL  tuned  : acc={v_acc_t:.4f}  wF1={v_w_t:.4f}  mF1={v_m_t:.4f}  min={v_min_t:.4f}\n")
        f.write(f"TEST untuned: acc={t_acc_u:.4f}  wF1={t_w_u:.4f}  mF1={t_m_u:.4f}  min={t_min_u:.4f}\n")
        f.write(f"TEST tuned  : acc={t_acc_t:.4f}  wF1={t_w_t:.4f}  mF1={t_m_t:.4f}  min={t_min_t:.4f}\n\n")
        f.write(classification_report(test_labels, test_preds_tuned, target_names=names, zero_division=0, digits=3))
    print(f"Saved -> {ENSEMBLE_DIR}/tuned_test_report.txt")


if __name__ == "__main__":
    main()
