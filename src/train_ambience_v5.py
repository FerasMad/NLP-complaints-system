"""Fine-tune a single CAMeLBERT-mix model for the 9-class ambience experiment.

This is the v5 training script. It does NOT touch the production 8-class
model — output goes to `models/single_ambience_v1/` and the production
artifacts at `models/single_final/` and `models/ensemble_final/` stay intact.

Pipeline:

  1. Load labeled data from data/text/complaints_labeled.csv (8-class baseline)
     plus data/text/ambience_synthetic_v1.csv (contrastive pairs from
     src/data/generate_ambience_synthetic.py).
  2. Filter to the 9-class schema (drops anything outside it).
  3. Run the schema-aware splitter to produce train/val/test under
     data/processed/ambience/. Synthetic and augmented sources land in
     train only — the leakage gate enforces this.
  4. Fine-tune CAMeLBERT-mix on the train split. Class-weighted cross-entropy
     to compensate for ambience being under-represented.
  5. Evaluate on val + test.
  6. Save the model, tokenizer, label_map.json, training_metrics.json,
     and a config.json with schema_version='9class_ambience'.

Usage:

    python src/train_ambience_v5.py \\
        --epochs 4 --batch-size 16 --lr 2e-5

On an RTX 4070 this takes roughly 20-40 minutes depending on how much
real data you've added. With only the synthetic dataset (~250 rows
ambience-related plus the 8-class baseline) it converges fast.

After training:
    - Run src/evaluation/evaluate_ambience.py to score against the
      adversarial fixture in tests/fixtures/ambience_adversarial.csv.
    - If ambience F1 >= 0.85 on the test split AND adversarial pass-rate
      >= 0.85, the v5 experiment has cleared its bar.

What this script does NOT do:
    - Train an ensemble (single model only — ensemble v5 is a follow-up).
    - Tune hyperparameters (defaults are tuned for the 8-class baseline;
      may need adjustment for 9-class).
    - Use the keyword-rescue layer (that's an inference-time post-processor).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def _import_torch_stack():
    """Defer heavy imports so --help works without torch installed."""
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    return {
        "torch": torch,
        "nn": nn,
        "Dataset": Dataset,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "prfs": precision_recall_fscore_support,
    }


CATEGORIES_9CLASS = [
    "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
    "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    "الجو والمكان",
]

DEFAULT_BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "single_ambience_v1"
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "ambience"

TRAIN_ONLY_SOURCES = {
    "synthetic", "augmented_bt", "chatgpt_synthetic",
    "pseudo_labeled", "eda_augmented",
    "ambience_synthetic", "ambience_eda_augmented",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_combined_data(
    baseline_csv: Path,
    ambience_csv: Path | None,
) -> pd.DataFrame:
    """Combine the 8-class baseline and ambience synthetic into one frame.

    The baseline CSV (data/text/complaints_labeled.csv) is the source of
    truth for the 8 production categories. The ambience CSV is the v5-only
    addition. Both must have at minimum: text, category, source.
    """
    if not baseline_csv.exists():
        print(f"[load] WARNING: baseline CSV not found at {baseline_csv}", file=sys.stderr)
        baseline_df = pd.DataFrame(columns=["text", "category", "source"])
    else:
        baseline_df = pd.read_csv(baseline_csv, encoding="utf-8-sig")
        # Older versions may have category in a different column or use 8class only
        if "category" not in baseline_df.columns:
            raise ValueError(f"baseline CSV missing 'category' column; got {list(baseline_df.columns)}")
        baseline_df = baseline_df[baseline_df["category"].isin(CATEGORIES_9CLASS)]

    if ambience_csv and ambience_csv.exists():
        amb_df = pd.read_csv(ambience_csv, encoding="utf-8-sig")
        amb_df = amb_df[amb_df["category"].isin(CATEGORIES_9CLASS)]
    else:
        print(f"[load] WARNING: ambience synthetic CSV not found at {ambience_csv}", file=sys.stderr)
        amb_df = pd.DataFrame(columns=["text", "category", "source"])

    cols_keep = [c for c in ["text", "category", "source"] if c in baseline_df.columns or c in amb_df.columns]
    baseline_df = baseline_df[[c for c in cols_keep if c in baseline_df.columns]]
    amb_df = amb_df[[c for c in cols_keep if c in amb_df.columns]]

    combined = pd.concat([baseline_df, amb_df], ignore_index=True)
    combined = combined.dropna(subset=["text", "category"])
    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]
    if "source" not in combined.columns:
        combined["source"] = "unknown"
    combined["source"] = combined["source"].fillna("unknown")
    return combined.reset_index(drop=True)


def stratified_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split per-category into train/val/test. Train-only sources stay in train."""
    rng = np.random.RandomState(seed)
    real = df[~df["source"].isin(TRAIN_ONLY_SOURCES)].copy()
    train_only = df[df["source"].isin(TRAIN_ONLY_SOURCES)].copy()

    train_parts: list[pd.DataFrame] = [train_only]
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for cat, group in real.groupby("category"):
        group = group.sample(frac=1.0, random_state=rng).reset_index(drop=True)
        n = len(group)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_test - n_val
        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train:n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val:])

    train = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    val = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    test = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return train, val, test


def write_split_artifacts(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
    output_dir: Path,
) -> dict[str, int]:
    """Write train/val/test CSVs + label_map.json + manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_map = {cat: i for i, cat in enumerate(CATEGORIES_9CLASS)}
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df = df.copy()
        df["label"] = df["category"].map(label_map)
        df.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    manifest = {
        "schema_version": "9class_ambience",
        "num_classes": len(label_map),
        "categories": CATEGORIES_9CLASS,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "train_class_distribution": dict(Counter(train["category"])),
    }
    with open(output_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest["split_sizes"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, T) -> dict:
    """Fine-tune CAMeLBERT-mix and return eval metrics. T is the import bundle."""
    label_map = {cat: i for i, cat in enumerate(CATEGORIES_9CLASS)}
    id2label = {i: cat for cat, i in label_map.items()}

    train_df = pd.read_csv(args.data_dir / "train.csv", encoding="utf-8-sig")
    val_df = pd.read_csv(args.data_dir / "val.csv", encoding="utf-8-sig")
    test_df = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")

    train_df["label"] = train_df["category"].map(label_map)
    val_df["label"] = val_df["category"].map(label_map)
    test_df["label"] = test_df["category"].map(label_map)

    print(f"[train] sizes: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print(f"[train] train class distribution: {dict(Counter(train_df['category']))}")

    print(f"[train] loading tokenizer + model from {args.base_model}")
    tokenizer = T["AutoTokenizer"].from_pretrained(args.base_model)
    model = T["AutoModelForSequenceClassification"].from_pretrained(
        args.base_model,
        num_labels=len(CATEGORIES_9CLASS),
        id2label=id2label,
        label2id=label_map,
        ignore_mismatched_sizes=True,
    )

    class TextDataset(T["Dataset"]):
        def __init__(self, df):
            self.texts = df["text"].astype(str).tolist()
            self.labels = df["label"].astype(int).tolist()

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(self.texts[i], truncation=True, max_length=args.max_length)
            enc["labels"] = self.labels[i]
            return enc

    train_ds = TextDataset(train_df)
    val_ds = TextDataset(val_df)
    test_ds = TextDataset(test_df)

    # Class weights — inverse-frequency on train set to compensate for ambience under-representation
    class_counts = np.array([
        max((train_df["label"] == i).sum(), 1)
        for i in range(len(CATEGORIES_9CLASS))
    ], dtype=np.float32)
    class_weights = (class_counts.sum() / (len(class_counts) * class_counts))
    class_weights = T["torch"].tensor(class_weights, dtype=T["torch"].float32)
    print(f"[train] class weights: {dict(zip(CATEGORIES_9CLASS, class_weights.tolist()))}")

    use_focal = bool(args.use_focal_loss)
    focal_gamma = args.focal_gamma

    class WeightedTrainer(T["Trainer"]):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights_on_device = class_weights.to(logits.device)
            if use_focal:
                # Class-weighted focal loss. Designed for high-precision /
                # low-recall imbalance (exactly what v5 ambience hit on
                # run 1: precision 95%, recall 67%). The (1 - p_t)^gamma
                # factor down-weights easy correct examples and up-weights
                # hard wrong ones.
                log_probs = T["nn"].functional.log_softmax(logits, dim=-1)
                probs = log_probs.exp()
                target_log_probs = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                target_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                focal_weight = (1.0 - target_probs).clamp(min=0.0).pow(focal_gamma)
                per_sample = -focal_weight * target_log_probs * weights_on_device[labels]
                loss = per_sample.mean()
            else:
                loss = T["nn"].functional.cross_entropy(
                    logits, labels, weight=weights_on_device
                )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        labels = eval_pred.label_ids
        acc = T["accuracy_score"](labels, preds)
        macro = T["f1_score"](labels, preds, average="macro", zero_division=0)
        weighted = T["f1_score"](labels, preds, average="weighted", zero_division=0)
        per_class = T["prfs"](labels, preds, labels=list(range(len(CATEGORIES_9CLASS))), zero_division=0)
        ambience_idx = label_map["الجو والمكان"]
        amb_p = per_class[0][ambience_idx]
        amb_r = per_class[1][ambience_idx]
        amb_f1 = per_class[2][ambience_idx]
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "ambience_precision": float(amb_p),
            "ambience_recall": float(amb_r),
            "ambience_f1": float(amb_f1),
        }

    # Detect whether val has any ambience rows. If not, ambience_f1 is
    # always 0 and using it for best-model selection would pick a random
    # checkpoint. Codex's first run hit this exact failure mode: trained
    # 4 epochs with ambience F1 = 0 on val (synthetic-only ambience is
    # train-only by leakage gate), and "best" model was worse than the
    # final checkpoint. Fall back to weighted_f1 in that case.
    val_has_ambience = (val_df["category"] == "الجو والمكان").any()
    best_metric = args.best_metric
    if best_metric == "ambience_f1" and not val_has_ambience:
        print(
            "[train] WARNING: val split has zero ambience rows "
            "(synthetic-only training). Switching metric_for_best_model "
            "from 'ambience_f1' to 'weighted_f1' to avoid noise-driven "
            "checkpoint selection. Pass --best-metric weighted_f1 to "
            "silence this warning, or add real ambience rows to val.",
            file=sys.stderr,
        )
        best_metric = "weighted_f1"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = T["TrainingArguments"](
        output_dir=str(args.output_dir / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,  # keep last 2 checkpoints; full 4 ate ~2GB
        load_best_model_at_end=not args.promote_final_checkpoint,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=args.seed,
        fp16=T["torch"].cuda.is_available(),
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=T["DataCollatorWithPadding"](tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"[train] starting training for {args.epochs} epochs...")
    train_result = trainer.train()
    print(f"[train] train metrics: {train_result.metrics}")

    print("[eval] evaluating on val")
    val_metrics = trainer.evaluate(val_ds)
    print(f"  val: {val_metrics}")

    print("[eval] evaluating on test (held-out, not used in training)")
    test_metrics = trainer.evaluate(test_ds)
    print(f"  test: {test_metrics}")

    # Save model + tokenizer + config
    print(f"[save] writing artifacts to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # The HF config.json will already exist; add our schema_version + metrics.
    # Strip the "eval_" prefix that Trainer.evaluate() adds, but KEEP the
    # underlying metrics. Earlier version filtered them out by mistake,
    # leaving v5_metrics with only "epoch".
    def _strip_eval_prefix(d):
        return {(k[len("eval_"):] if k.startswith("eval_") else k): v for k, v in d.items()}

    config_path = args.output_dir / "config.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["schema_version"] = "9class_ambience"
    cfg["v5_metrics"] = {
        "val": _strip_eval_prefix(val_metrics),
        "test": _strip_eval_prefix(test_metrics),
        "best_metric_used": best_metric,
        "promoted_final_checkpoint": args.promote_final_checkpoint,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Also write a small training_metrics.json for downstream tooling
    metrics_path = args.output_dir / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "val": val_metrics,
            "test": test_metrics,
            "train": train_result.metrics,
        }, f, ensure_ascii=False, indent=2)

    return {"val": val_metrics, "test": test_metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                   help=f"HF model ID for the encoder (default: {DEFAULT_BASE_MODEL})")
    p.add_argument("--baseline-csv", type=Path,
                   default=ROOT / "data" / "text" / "complaints_labeled.csv",
                   help="The 8-class production labeled CSV")
    p.add_argument("--ambience-csv", type=Path,
                   default=ROOT / "data" / "text" / "ambience_synthetic_v1.csv",
                   help="The v5 ambience synthetic CSV (output of generate_ambience_synthetic.py)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help=f"Where to write split CSVs (default: {DEFAULT_DATA_DIR})")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Where to save the trained model (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-split", action="store_true",
                   help="Skip the data-loading/splitting step and reuse existing data-dir CSVs")
    p.add_argument(
        "--best-metric",
        default="ambience_f1",
        choices=["ambience_f1", "weighted_f1", "macro_f1", "accuracy"],
        help=(
            "Metric for load_best_model_at_end. Default 'ambience_f1' is "
            "right when val has real ambience rows; auto-falls-back to "
            "'weighted_f1' if val ambience is empty (synthetic-only run)."
        ),
    )
    p.add_argument(
        "--promote-final-checkpoint",
        action="store_true",
        help=(
            "Skip best-model-at-end and keep the final epoch's weights. "
            "Use when val metrics are unreliable (e.g. zero-ambience val). "
            "Codex's first run found the final checkpoint substantially "
            "outperformed the 'best' on the adversarial fixture."
        ),
    )
    p.add_argument(
        "--exclude-source",
        nargs="*",
        default=[],
        metavar="SOURCE",
        help=(
            "Drop ambience rows whose source matches one of these names. "
            "Recommended: --exclude-source synthetic — the 1933 v3-era "
            "templated ambience rows are noisy and likely hurt training. "
            "Non-ambience rows from the same source are kept."
        ),
    )
    p.add_argument(
        "--use-focal-loss",
        action="store_true",
        help=(
            "Use class-weighted focal loss instead of plain weighted CE. "
            "Designed for high-precision/low-recall imbalance — the exact "
            "pattern run 1 hit (95%% precision, 67%% recall on ambience). "
            "Recommended for ambience-experiment runs."
        ),
    )
    p.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss focusing parameter (default 2.0; only used with --use-focal-loss)",
    )
    args = p.parse_args()

    if not args.skip_split:
        print(f"[data] combining baseline + ambience synthetic")
        combined = load_combined_data(args.baseline_csv, args.ambience_csv)
        if combined.empty:
            print("[data] ERROR: no rows loaded. Run generate_ambience_synthetic.py first "
                  "OR pass --skip-split if data is already prepared.", file=sys.stderr)
            return 1

        # --exclude-source filter. Use this to drop the 1933 v3-era
        # `synthetic` ambience rows from the baseline; they're templated
        # and grammatically wonky and likely confuse training.
        if args.exclude_source:
            excluded_set = set(args.exclude_source)
            before = len(combined)
            # Only drop rows where category is ambience AND source is in the
            # excluded set. Keeps the same source name's non-ambience rows
            # intact (e.g. `synthetic` rows for other categories stay).
            mask = ~(
                combined["category"].eq("الجو والمكان")
                & combined["source"].isin(excluded_set)
            )
            combined = combined[mask].reset_index(drop=True)
            print(f"[data] --exclude-source dropped "
                  f"{before - len(combined)} ambience rows from sources: {sorted(excluded_set)}")

        print(f"[data] combined size: {len(combined)} rows; class distribution:")
        for cat, n in Counter(combined["category"]).most_common():
            print(f"  {cat}: {n}")

        train_df, val_df, test_df = stratified_split(combined, seed=args.seed)
        sizes = write_split_artifacts(train_df, val_df, test_df, args.data_dir)
        print(f"[data] wrote splits to {args.data_dir}: {sizes}")

    print(f"[train] importing torch + transformers (this is slow)...")
    T = _import_torch_stack()

    metrics = train(args, T)

    print()
    print("=" * 60)
    print(f"v5 training complete. Artifacts at {args.output_dir}")
    print(f"  test ambience F1:        {metrics['test'].get('eval_ambience_f1', 0):.4f}")
    print(f"  test ambience precision: {metrics['test'].get('eval_ambience_precision', 0):.4f}")
    print(f"  test ambience recall:    {metrics['test'].get('eval_ambience_recall', 0):.4f}")
    print(f"  test macro F1:           {metrics['test'].get('eval_macro_f1', 0):.4f}")
    print(f"  test accuracy:           {metrics['test'].get('eval_accuracy', 0):.4f}")
    target = 0.85
    if metrics['test'].get('eval_ambience_f1', 0) >= target:
        print(f"  -> ambience F1 cleared the {target:.0%} bar.")
    else:
        print(f"  -> ambience F1 below {target:.0%}. Add more real data and re-train.")
    print("=" * 60)
    print()
    print("Next: python src/evaluation/evaluate_ambience.py "
          f"--model-dir {args.output_dir} --adversarial-fixture tests/fixtures/ambience_adversarial.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
