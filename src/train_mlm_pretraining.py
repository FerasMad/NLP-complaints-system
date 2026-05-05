"""Continued pre-training (MLM) of CAMeLBERT-mix on Arabic restaurant + hotel reviews.

This is a domain-adaptation step. The base CAMeLBERT-mix was trained on
mixed MSA + dialectal Arabic, but it doesn't know our specific domain
vocabulary well — Saudi delivery-app slang, hotel-review patterns, etc.

We continue pre-training on raw text from the harvested HF datasets
(no labels needed for MLM), then fine-tune on the labeled task.

Why this helps:
  - The encoder learns ambience-related Arabic phrases ('المكيف ما يبرد',
    'ريحة الزيت في المطعم', 'الكنب مهلوك') in their natural context
  - The downstream classifier head sees a better-conditioned representation
  - Doesn't require a single label — pure unsupervised domain adaptation

Sources:
  data/processed/ambience/huggingface/             qaym restaurant reviews
  data/processed/ambience/huggingface_hard/        HARD hotel reviews
  (Combines all three split files — ambience_candidates, hard_negatives,
   review_needed — to maximize raw text volume)

CLI:

    python src/train_mlm_pretraining.py --epochs 1 --batch-size 16

Output: models/camelbert_arabic_reviews_pretrained/

After pre-training, fine-tune for the 9-class task:

    python src/train_ambience_v5.py \\
        --base-model models/camelbert_arabic_reviews_pretrained \\
        --baseline-csv data/text/complaints_labeled_v5_real.csv \\
        --epochs 4 --batch-size 16 --lr 2e-5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_BASE_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
DEFAULT_OUTPUT = ROOT / "models" / "camelbert_arabic_reviews_pretrained"


def collect_corpus(sources: list[Path]) -> list[str]:
    """Read text columns from each source CSV, return deduplicated list."""
    seen = set()
    texts: list[str] = []
    for src in sources:
        if not src.exists():
            print(f"[corpus] skip (missing): {src}", file=sys.stderr)
            continue
        df = pd.read_csv(src, encoding="utf-8-sig")
        if "text" not in df.columns:
            print(f"[corpus] skip (no text col): {src}", file=sys.stderr)
            continue
        for t in df["text"].astype(str):
            t = t.strip()
            if not t or t in seen:
                continue
            # Keep texts that have at least 3 words (MLM benefits from
            # multi-word context; very short fragments are useless)
            if len(t.split()) < 3:
                continue
            seen.add(t)
            texts.append(t)
        print(f"[corpus] {src.name}: contributed (cumulative unique {len(texts)})")
    return texts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--epochs", type=int, default=1,
                   help="MLM epochs (default 1; 1 epoch on ~100k rows is usually enough for domain adaptation)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5,
                   help="Higher than fine-tuning lr — MLM benefits from more aggressive updates (default 5e-5)")
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--mlm-probability", type=float, default=0.15,
                   help="Fraction of tokens to mask (default 0.15, BERT standard)")
    p.add_argument("--max-rows", type=int, default=0,
                   help="Cap corpus size (default 0 = all). Useful for smoke tests.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    qaym_dir = ROOT / "data" / "processed" / "ambience" / "huggingface"
    hard_dir = ROOT / "data" / "processed" / "ambience" / "huggingface_hard"
    sources = [
        qaym_dir / "ambience_candidates.csv",
        qaym_dir / "hard_negative_candidates.csv",
        qaym_dir / "review_needed.csv",
        hard_dir / "ambience_candidates.csv",
        hard_dir / "hard_negative_candidates.csv",
        hard_dir / "review_needed.csv",
    ]

    print(f"[corpus] collecting raw text from {len(sources)} sources...")
    texts = collect_corpus(sources)
    print(f"[corpus] total unique texts: {len(texts)}")
    if args.max_rows > 0:
        texts = texts[: args.max_rows]
        print(f"[corpus] capped to {len(texts)} rows")
    if not texts:
        print("[corpus] ERROR: no text loaded. Run the HF importer first:", file=sys.stderr)
        print("    python src/data/import_hf_datasets_for_ambience.py "
              "--output-dir data/processed/ambience/huggingface", file=sys.stderr)
        return 1

    print(f"[load] importing torch + transformers...")
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    print(f"[load] loading {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    class TextOnlyDataset(Dataset):
        def __init__(self, texts: list[str]):
            self.texts = texts

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, i):
            enc = tokenizer(
                self.texts[i],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )
            return enc

    ds = TextOnlyDataset(texts)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,  # MLM checkpoints are big; keep just the latest
        logging_steps=100,
        report_to=[],
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print(f"[train] starting MLM pre-training for {args.epochs} epoch(s) on "
          f"{len(texts)} texts (batch={args.batch_size}, lr={args.lr}, "
          f"mask={args.mlm_probability})")
    result = trainer.train()
    print(f"[train] result: {result.metrics}")

    print(f"[save] writing domain-adapted model to {args.output_dir}")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print()
    print("=" * 60)
    print("Domain-adapted CAMeLBERT-mix saved.")
    print()
    print("Next: fine-tune for 9-class ambience classification:")
    print()
    print(f"  python src/train_ambience_v5.py \\")
    print(f"      --base-model {args.output_dir} \\")
    print(f"      --baseline-csv data/text/complaints_labeled_v5_real.csv \\")
    print(f"      --epochs 4 --batch-size 16 --lr 2e-5 \\")
    print(f"      --use-focal-loss")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
