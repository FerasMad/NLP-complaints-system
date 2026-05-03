"""
Error analysis on the test set using the winning ensemble.

Outputs:
  1. Confusion matrix (CSV + text)
  2. Top failure modes (gold→pred pairs by count)
  3. For each top failure mode: 5-10 sample errors, with text and confidences
  4. Heuristic error classification:
       - "label_noise":  text has strong keywords for the predicted class, almost none for the gold
       - "multi_aspect": text has strong keywords for both gold and predicted class
       - "model_error":  text has clear gold-class keywords but model picked wrong
       - "ambiguous":    nothing clearly dominant
       - "ood":          short or unusual length, no clear keywords
  5. A reviewable CSV at data/processed/error_analysis.csv

Use this to decide where to invest next data effort.
"""
from __future__ import annotations

import csv
import gc
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_DIR = ROOT / "models" / "error_analysis"
with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    import json as _json
    NUM_LABELS = len(_json.load(_f))
BATCH_SIZE = 64

# Reuse the audit script's keyword sets
sys.path.insert(0, str(ROOT / "src"))
from audit_ambiance_eval import KW, count_hits


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


def classify_error(text, gold_cat, pred_cat):
    gold_hits = count_hits(text, KW.get(gold_cat, set()))
    pred_hits = count_hits(text, KW.get(pred_cat, set()))
    word_count = len(text.split())

    if word_count < 5:
        return "ood_short"
    if gold_hits == 0 and pred_hits >= 1:
        return "label_noise"   # gold has no supporting keywords; pred does
    if gold_hits >= 2 and pred_hits >= 2:
        return "multi_aspect"
    if gold_hits >= 1 and pred_hits == 0:
        return "model_error"   # gold supported, pred unsupported
    if gold_hits == 0 and pred_hits == 0:
        return "ambiguous"
    if pred_hits > gold_hits:
        return "label_noise_likely"
    return "ambiguous"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(TEST_CSV)[["text", "label", "category"]]
    texts = test_df["text"].astype(str).tolist()
    labels = test_df["label"].to_numpy()
    print(f"Test rows: {len(texts)}")

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
    print(f"Device: {device}")

    stack = np.zeros((len(model_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        print(f"  [{k+1}/{len(model_dirs)}] {d.name}")
        stack[k] = predict_probs(d, texts, device, max_length)

    avg = stack.mean(axis=0)
    adjusted = avg + biases
    preds = adjusted.argmax(axis=1)
    confidences = avg.max(axis=1)

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_LABELS)))
    cm_df = pd.DataFrame(cm, index=[f"gold:{n}" for n in names], columns=[f"pred:{n}" for n in names])
    cm_path = OUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path, encoding="utf-8-sig")
    print(f"\nConfusion matrix -> {cm_path}")

    # Find errors
    err_mask = preds != labels
    print(f"\nTotal errors: {err_mask.sum()}/{len(labels)} ({err_mask.mean():.2%})")

    # Top failure modes
    fail_pairs = Counter()
    for gold, pred in zip(labels[err_mask], preds[err_mask]):
        fail_pairs[(int(gold), int(pred))] += 1
    print("\nTop 15 failure modes (gold → pred, count):")
    for (g, p), c in fail_pairs.most_common(15):
        print(f"  {names[g]:>20s}  →  {names[p]:<20s}  : {c}")

    # Detailed CSV for human review
    rows = []
    err_idx = np.where(err_mask)[0]
    for i in err_idx:
        text = texts[i]
        gold_cat = names[int(labels[i])]
        pred_cat = names[int(preds[i])]
        err_type = classify_error(text, gold_cat, pred_cat)
        rows.append({
            "idx": i,
            "text": text,
            "gold": gold_cat,
            "pred": pred_cat,
            "confidence": float(confidences[i]),
            "error_type": err_type,
            "gold_keyword_hits": count_hits(text, KW.get(gold_cat, set())),
            "pred_keyword_hits": count_hits(text, KW.get(pred_cat, set())),
            "word_count": len(text.split()),
        })
    err_df = pd.DataFrame(rows)
    err_df = err_df.sort_values(["error_type", "gold", "pred"]).reset_index(drop=True)
    err_path = OUT_DIR / "errors.csv"
    err_df.to_csv(err_path, index=False, encoding="utf-8-sig")
    print(f"\nAll {len(err_df)} errors -> {err_path}")

    # Error type breakdown
    print("\nError type distribution:")
    for et, c in err_df["error_type"].value_counts().items():
        print(f"  {et}: {c} ({c / len(err_df):.1%})")

    # Per-failure-mode samples
    print("\n5 sample errors per top-10 failure mode:")
    for (g, p), c in fail_pairs.most_common(10):
        sub = err_df[(err_df.gold == names[g]) & (err_df.pred == names[p])].head(5)
        print(f"\n  ── {names[g]} → {names[p]} ({c} errors) ──")
        for _, r in sub.iterrows():
            print(f"    [{r['error_type']}] conf={r['confidence']:.2f} : {r['text'][:120]}")

    # Markdown report
    md = OUT_DIR / "error_analysis_report.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Error Analysis — winning ensemble on test set\n\n")
        f.write(f"Test rows: {len(labels)}\n")
        f.write(f"Errors: {err_mask.sum()} ({err_mask.mean():.2%})\n\n")
        f.write("## Error type distribution\n\n")
        for et, c in err_df["error_type"].value_counts().items():
            f.write(f"- **{et}**: {c} ({c / len(err_df):.1%})\n")
        f.write("\n## Top 15 failure modes\n\n")
        f.write("| Gold | → | Predicted | Count |\n|---|---|---|---:|\n")
        for (g, p), c in fail_pairs.most_common(15):
            f.write(f"| {names[g]} | → | {names[p]} | {c} |\n")
        f.write("\n## Sample errors per top-10 failure mode\n\n")
        for (g, p), c in fail_pairs.most_common(10):
            f.write(f"### {names[g]} → {names[p]} ({c} errors)\n\n")
            sub = err_df[(err_df.gold == names[g]) & (err_df.pred == names[p])].head(5)
            for _, r in sub.iterrows():
                f.write(f"- `[{r['error_type']}, conf={r['confidence']:.2f}]` {r['text'][:200]}\n")
            f.write("\n")
    print(f"\nMarkdown report -> {md}")
    print("\nNext: open errors.csv for full list, or read the markdown report.")


if __name__ == "__main__":
    main()
