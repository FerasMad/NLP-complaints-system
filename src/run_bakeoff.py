"""
Train and evaluate multiple Arabic BERT variants on the rebuilt dataset.
Records per-model results in models/bakeoff_results.csv.

Usage:
    py src/run_bakeoff.py                    # all 5 models, default config
    py src/run_bakeoff.py --aggressive       # 7 epochs, DOMINANT_CAP=5000
    py src/run_bakeoff.py --models camelbert-mix arabertv02   # subset
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
MODELS_DIR = ROOT / "models"
RESULTS_CSV = MODELS_DIR / "bakeoff_results.csv"
DOMINANT_CATEGORY = "جودة الطعام"

# Dynamically derive NUM_LABELS from label_map.json (handles 8/9-class transitions)
with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))

MODELS = {
    "camelbert-mix": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "camelbert-da":  "CAMeL-Lab/bert-base-arabic-camelbert-da",
    "arabertv02":    "aubmindlab/bert-base-arabertv02",
    "marbert":       "UBC-NLP/MARBERT",
    "xlm-r-base":    "xlm-roberta-base",
}


def load_data(dominant_cap: int):
    train_df = pd.read_csv(TRAIN_CSV)[["text", "label", "category"]]
    val_df = pd.read_csv(VAL_CSV)[["text", "label", "category"]]

    # subsample dominant class
    dom = train_df[train_df["category"] == DOMINANT_CATEGORY]
    other = train_df[train_df["category"] != DOMINANT_CATEGORY]
    if len(dom) > dominant_cap:
        dom = dom.sample(dominant_cap, random_state=42)
    train_df = pd.concat([dom, other]).sample(frac=1, random_state=42).reset_index(drop=True)

    return train_df, val_df


def make_weighted_trainer(class_weights):
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss
    return WeightedTrainer


def compute_metrics_factory():
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }
    return compute_metrics


def train_one(short_name, model_id, epochs, batch_size, dominant_cap, label_smoothing, max_length=128, seed=42, save_suffix=""):
    print(f"\n{'=' * 70}")
    print(f"TRAINING: {short_name}{save_suffix}  ({model_id})")
    print(f"  epochs={epochs}  batch_size={batch_size}  dominant_cap={dominant_cap}  ls={label_smoothing}  max_length={max_length}  seed={seed}")
    print("=" * 70)

    train_df, val_df = load_data(dominant_cap)
    print(f"  train rows: {len(train_df)}, val rows: {len(val_df)}")

    labels_arr = train_df["label"].values
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=labels_arr)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    # Set torch seed for this run (model init, dropout, etc.)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=NUM_LABELS)

    name_with_suffix = f"{short_name}{save_suffix}"
    out_dir = MODELS_DIR / "bakeoff" / name_with_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=max(1, 32 // batch_size),  # keep effective ~32
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=200,
        report_to="none",
        dataloader_num_workers=2,
        label_smoothing_factor=label_smoothing,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    WeightedTrainer = make_weighted_trainer(class_weights)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(),
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    print(f"  train time: {train_seconds:.0f}s")

    # Final val predictions
    preds_output = trainer.predict(val_ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels_v = preds_output.label_ids

    acc = accuracy_score(labels_v, preds)
    f1_w = f1_score(labels_v, preds, average="weighted")
    f1_m = f1_score(labels_v, preds, average="macro")
    f1_per_class = f1_score(labels_v, preds, average=None, zero_division=0)

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    inv = {v: k for k, v in label_map.items()}
    names = [inv[i] for i in range(NUM_LABELS)]
    report = classification_report(labels_v, preds, target_names=names, zero_division=0, digits=3)

    print(f"\n  Val accuracy:    {acc:.4f}")
    print(f"  Val weighted F1: {f1_w:.4f}")
    print(f"  Val macro F1:    {f1_m:.4f}")
    print(report)

    # Save final model + report
    final_dir = MODELS_DIR / "bakeoff" / f"{name_with_suffix}_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(final_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    with open(final_dir / "eval_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"epochs={epochs} batch={batch_size} dominant_cap={dominant_cap} ls={label_smoothing} max_length={max_length} seed={seed}\n")
        f.write(f"Val accuracy:    {acc:.4f}\n")
        f.write(f"Val weighted F1: {f1_w:.4f}\n")
        f.write(f"Val macro F1:    {f1_m:.4f}\n\n")
        f.write(report)
    print(f"  saved -> {final_dir}")

    # Free up disk: delete checkpoints (final is in *_final)
    for ck in out_dir.glob("checkpoint-*"):
        shutil.rmtree(ck, ignore_errors=True)

    # Free VRAM before next model
    del trainer, model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "short_name": name_with_suffix,
        "model_id": model_id,
        "epochs": epochs,
        "batch_size": batch_size,
        "dominant_cap": dominant_cap,
        "label_smoothing": label_smoothing,
        "max_length": max_length,
        "seed": seed,
        "train_seconds": int(train_seconds),
        "accuracy": acc,
        "weighted_f1": f1_w,
        "macro_f1": f1_m,
        **{f"f1_{names[i]}": f1_per_class[i] for i in range(NUM_LABELS)},
        "min_class_f1": float(min(f1_per_class)),
        "all_classes_above_70": bool((f1_per_class >= 0.70).all()),
    }


def append_result(row, fieldnames):
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressive", action="store_true", help="7 epochs, DOMINANT_CAP=5000, ls=0.1")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()), help="Subset of model short names")
    ap.add_argument("--small-batch", action="store_true", help="Force batch_size=8 (for big models / OOM)")
    ap.add_argument("--epochs", type=int, default=None, help="Override epochs")
    ap.add_argument("--dominant-cap", type=int, default=None, help="Override dominant class cap")
    ap.add_argument("--label-smoothing", type=float, default=None, help="Override label smoothing")
    ap.add_argument("--max-length", type=int, default=128, help="Tokenizer max length")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for model init")
    ap.add_argument("--save-suffix", type=str, default="", help="Suffix to append to saved model dir name")
    args = ap.parse_args()

    epochs = args.epochs if args.epochs is not None else (7 if args.aggressive else 5)
    dominant_cap = args.dominant_cap if args.dominant_cap is not None else (5000 if args.aggressive else 8000)
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else (0.1 if args.aggressive else 0.0)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames_set = set()
    completed = []
    failed = []

    for short_name in args.models:
        if short_name not in MODELS:
            print(f"  ! Unknown model {short_name}, skipping")
            continue
        model_id = MODELS[short_name]
        # Pick batch size: big models drop to 8 to fit RTX 4070
        if args.small_batch or short_name in {"xlm-r-base", "marbert"}:
            bs = 8
        else:
            bs = 16

        try:
            row = train_one(short_name, model_id, epochs, bs, dominant_cap, label_smoothing, max_length=args.max_length, seed=args.seed, save_suffix=args.save_suffix)
            row["aggressive"] = bool(args.aggressive)
            fieldnames_set.update(row.keys())
            completed.append(row)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  ! OOM on {short_name} at batch_size={bs} — retrying at batch_size=4")
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                row = train_one(short_name, model_id, epochs, 4, dominant_cap, label_smoothing, max_length=args.max_length, seed=args.seed, save_suffix=args.save_suffix)
                row["aggressive"] = bool(args.aggressive)
                fieldnames_set.update(row.keys())
                completed.append(row)
            except Exception as e2:
                print(f"  ! failed on retry: {e2}")
                failed.append((short_name, str(e2)))
        except Exception as e:
            print(f"  ! failed: {type(e).__name__}: {e}")
            failed.append((short_name, str(e)))

    if completed:
        # consistent column order
        ordered = [
            "short_name", "model_id", "aggressive", "epochs", "batch_size",
            "dominant_cap", "label_smoothing", "train_seconds",
            "accuracy", "weighted_f1", "macro_f1", "min_class_f1",
            "all_classes_above_70",
        ]
        for row in completed:
            for k in row:
                if k not in ordered:
                    ordered.append(k)
        for row in completed:
            append_result(row, ordered)

    print("\n" + "=" * 70)
    print("BAKEOFF SUMMARY")
    print("=" * 70)
    if completed:
        completed.sort(key=lambda r: (r["all_classes_above_70"], r["macro_f1"], r["weighted_f1"]), reverse=True)
        for r in completed:
            mark = "✓" if r["all_classes_above_70"] else " "
            print(f"  [{mark}] {r['short_name']:<15s}  wF1={r['weighted_f1']:.3f}  mF1={r['macro_f1']:.3f}  minClass={r['min_class_f1']:.3f}  ({r['train_seconds']}s)")
        winner = completed[0]
        print(f"\n  WINNER: {winner['short_name']}  wF1={winner['weighted_f1']:.3f}  mF1={winner['macro_f1']:.3f}")
    if failed:
        print("\nFailed:")
        for name, err in failed:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
