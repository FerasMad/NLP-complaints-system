"""Light data augmentation for ambience training rows.

Append-only script: reads an existing labeled CSV that contains
ambience rows, generates safe perturbations of those rows, and appends
the new rows to a destination CSV with source="ambience_eda_augmented".

Why this exists: the v3 ambience class died from too few clean rows.
Even after the v5 retry collects fresh data via filter_ambience_candidates.py
and labels it, the absolute count is likely to stay small (a few hundred at
best). This script is a low-risk multiplier — 2-4x the volume without
inventing new content, just by perturbing what's already there.

Augmentation strategy: deterministic, dialect-safe perturbations only.
We do NOT use back-translation (Helsinki-NLP MarianMT was tried and
regressed in v4 — see documentation.pdf §3) and we do NOT use synonym
replacement (Arabic synonym dictionaries are weak for Saudi/Gulf dialect).

What we DO use, in order from safest to riskiest:

  1. Random word drop  — drop 1-2 non-anchor words from longer rows.
     Anchor words (ambience keywords, risky tokens) are protected so
     the augmented row keeps its label-bearing signal.

  2. Word reordering  — swap two adjacent non-anchor words. Arabic word
     order is more flexible than English; small swaps usually preserve
     meaning in dialect.

  3. Filler token append  — prepend/append common dialect fillers
     ("والله", "بصراحه", "بجد") that don't change the label.

Each row produces N augmented variants where N is configurable (default 2).
All augmented rows go to TRAIN ONLY — the leakage gate in
tests/test_data_pipeline.py blocks them from val/test via the
'ambience_eda_augmented' source name.

CLI:

    python src/augment_ambience.py \\
        --input data/text/complaints_labeled.csv \\
        --output data/text/complaints_labeled.csv \\
        --variants 2

If --output equals --input, the file is appended in place (after a
backup is written to <input>.bak).
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from utils.arabic_normalization import clean
from config.ambience_keywords import (
    AMBIENCE_HIGH_PRECISION_PHRASES,
    AMBIENCE_SINGLE_TOKENS,
    RISKY_AMBIGUOUS,
)


AMBIENCE_LABEL = "الجو والمكان"
AUGMENTED_SOURCE = "ambience_eda_augmented"

# Anchor tokens (normalized) — never dropped or shuffled, so the row's
# label-bearing signal stays intact.
_ANCHORS: set[str] = (
    {clean(p) for p in AMBIENCE_SINGLE_TOKENS if clean(p)} |
    {clean(p) for p in RISKY_AMBIGUOUS if clean(p)}
)

# Multi-word ambience phrases — if present in a row, the phrase as a whole
# is anchor and we never split it.
_ANCHOR_PHRASES: list[str] = sorted(
    {clean(p) for p in AMBIENCE_HIGH_PRECISION_PHRASES if clean(p)},
    key=len, reverse=True,
)

# Dialect filler tokens that don't shift the label
_FILLERS = ["والله", "بصراحه", "بجد", "صراحه", "للأمانه"]


def _is_anchor(word: str) -> bool:
    return clean(word) in _ANCHORS


def _drop_one(words: list[str], rng: random.Random) -> list[str] | None:
    """Drop one non-anchor word from a list. Returns None if no candidate."""
    candidates = [i for i, w in enumerate(words) if not _is_anchor(w)]
    if not candidates or len(words) < 4:
        return None
    drop_idx = rng.choice(candidates)
    return words[:drop_idx] + words[drop_idx + 1:]


def _swap_adjacent(words: list[str], rng: random.Random) -> list[str] | None:
    """Swap two adjacent non-anchor words. Returns None if no candidate pair."""
    pairs = [
        i for i in range(len(words) - 1)
        if not _is_anchor(words[i]) and not _is_anchor(words[i + 1])
    ]
    if not pairs:
        return None
    i = rng.choice(pairs)
    new = words.copy()
    new[i], new[i + 1] = new[i + 1], new[i]
    return new


def _add_filler(words: list[str], rng: random.Random) -> list[str]:
    """Prepend or append a dialect filler. Always succeeds."""
    filler = rng.choice(_FILLERS)
    return ([filler] + words) if rng.random() < 0.5 else (words + [filler])


def perturb(text: str, rng: random.Random) -> str | None:
    """Return one perturbed variant of `text`, or None if no safe edit applies."""
    words = text.split()
    if len(words) < 3:
        return None
    # Cycle through strategies; first one that produces a different result wins
    for strategy in (_drop_one, _swap_adjacent, _add_filler):
        out = strategy(words, rng)
        if out is not None and out != words:
            return " ".join(out)
    return None


def augment_dataframe(df: pd.DataFrame, variants: int, seed: int = 42) -> pd.DataFrame:
    """Build a new DataFrame containing only augmented ambience rows.

    Returns a frame ready to be concatenated to the original. Empty if
    no ambience rows are present.
    """
    if "category" not in df.columns or "text" not in df.columns:
        raise ValueError("input CSV must have 'category' and 'text' columns")

    amb = df[df["category"] == AMBIENCE_LABEL]
    if amb.empty:
        return pd.DataFrame(columns=df.columns)

    rng = random.Random(seed)
    new_rows: list[dict] = []
    seen_texts = set(df["text"].astype(str))

    for _, row in amb.iterrows():
        base_text = str(row["text"])
        for v in range(variants):
            perturbed = perturb(base_text, rng)
            if perturbed is None or perturbed in seen_texts:
                continue
            seen_texts.add(perturbed)
            new = row.to_dict()
            new["text"] = perturbed
            if "source" in new:
                new["source"] = AUGMENTED_SOURCE
            new_rows.append(new)

    return pd.DataFrame(new_rows, columns=df.columns)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Append augmented ambience rows to a labeled CSV."
    )
    p.add_argument("--input", required=True, type=Path,
                   help="Input labeled CSV (must have 'category' and 'text' columns)")
    p.add_argument("--output", required=True, type=Path,
                   help="Where to write the result. If equal to --input, appends in place after backup.")
    p.add_argument("--variants", type=int, default=2,
                   help="Number of perturbed copies per ambience row (default: 2)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input, encoding="utf-8")
    print(f"Loaded {len(df)} rows from {args.input}")

    n_amb = (df["category"] == AMBIENCE_LABEL).sum() if "category" in df.columns else 0
    print(f"  ambience rows: {n_amb}")

    if n_amb == 0:
        print("No ambience rows to augment. Exiting without changes.")
        return 0

    augmented = augment_dataframe(df, variants=args.variants, seed=args.seed)
    print(f"  generated {len(augmented)} augmented rows ({len(augmented) / max(n_amb, 1):.1f}x)")

    out_df = pd.concat([df, augmented], ignore_index=True)

    if args.output == args.input:
        backup = args.input.with_suffix(args.input.suffix + ".bak")
        shutil.copy2(args.input, backup)
        print(f"  backup -> {backup}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df)} total rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
