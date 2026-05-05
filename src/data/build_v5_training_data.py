"""Build the v5 training CSV by combining baseline + filter-derived weak labels.

This is the "use what we harvested" script. Takes:

  - data/text/complaints_labeled.csv               (95K rows, 8-class baseline)
  - data/text/ambience_synthetic_v1.csv            (250 rows, contrastive pairs)
  - data/processed/ambience/huggingface/
        ambience_candidates.csv                    (556 qaym rows, ~50-70% precision)

Drops the v3-era noisy synthetic ambience (--exclude-source synthetic during
training will handle that), and merges the qaym ambience candidates as
weak labels (source='weak_qaym_ambience', label_confidence='weak').

Skipped on purpose:
  - HARD ambience candidates (23752 rows) — hotel-domain, would teach the
    model to recognize hotel patterns that don't transfer to restaurants.
    Used for MLM pre-training instead (raw text, no labels).
  - filter hard_negative_candidates — duplicates of what's already in the
    baseline. Adding them inflates dataset size without new signal.

Output: data/text/complaints_labeled_v5_real.csv

CLI:

    python src/data/build_v5_training_data.py
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 stdout so Arabic prints don't crash on Windows cp1252.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent.parent

AMBIENCE_LABEL = "الجو والمكان"
WEAK_QAYM_SOURCE = "weak_qaym_ambience"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--baseline-csv",
        type=Path,
        default=ROOT / "data" / "text" / "complaints_labeled.csv",
    )
    p.add_argument(
        "--synthetic-csv",
        type=Path,
        default=ROOT / "data" / "text" / "ambience_synthetic_v1.csv",
    )
    p.add_argument(
        "--qaym-candidates-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "ambience" / "huggingface" / "ambience_candidates.csv",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "text" / "complaints_labeled_v5_real.csv",
    )
    args = p.parse_args()

    # Load baseline (8-class, 95K rows). Drop the v3-era noisy synthetic ambience
    # rows here — the trainer's --exclude-source flag does this too, but doing
    # it here means the output CSV is the cleanest "ready for training" snapshot.
    if not args.baseline_csv.exists():
        print(f"ERROR: baseline not found at {args.baseline_csv}", file=sys.stderr)
        return 2
    baseline = pd.read_csv(args.baseline_csv, encoding="utf-8-sig")
    print(f"[load] baseline: {len(baseline)} rows")
    n_v3_synth_amb = (
        (baseline["category"] == AMBIENCE_LABEL) & (baseline["source"] == "synthetic")
    ).sum()
    baseline = baseline[
        ~((baseline["category"] == AMBIENCE_LABEL) & (baseline["source"] == "synthetic"))
    ].reset_index(drop=True)
    print(f"[load]   dropped {n_v3_synth_amb} v3-era synthetic ambience rows (noisy)")
    print(f"[load]   keeping {len(baseline)} rows")

    # Load v5 synthetic (contrastive pairs, train-only)
    synthetic = pd.DataFrame()
    if args.synthetic_csv.exists():
        synthetic = pd.read_csv(args.synthetic_csv, encoding="utf-8-sig")
        print(f"[load] v5 synthetic: {len(synthetic)} rows")
    else:
        print(f"[load] WARNING: no v5 synthetic at {args.synthetic_csv}")

    # Load qaym ambience candidates as weak-labeled REAL training data.
    # These have ~50-70% precision per sampling — noise averages out in
    # training, model learns more correct ambience patterns than wrong ones.
    qaym_amb = pd.DataFrame()
    if args.qaym_candidates_csv.exists():
        qaym_amb = pd.read_csv(args.qaym_candidates_csv, encoding="utf-8-sig")
        print(f"[load] qaym ambience candidates (weak-labeled): {len(qaym_amb)} rows")
        # Force the category to ambience and re-source so we can track it
        qaym_amb["category"] = AMBIENCE_LABEL
        qaym_amb["source"] = WEAK_QAYM_SOURCE
        # Keep only the schema columns the baseline has
        keep_cols = [c for c in ["text", "category", "priority", "source"] if c in qaym_amb.columns]
        if "priority" not in qaym_amb.columns:
            qaym_amb["priority"] = 1
            keep_cols = ["text", "category", "priority", "source"]
        qaym_amb = qaym_amb[keep_cols]
    else:
        print(f"[load] WARNING: no qaym ambience candidates at {args.qaym_candidates_csv}")

    # Concatenate everything
    parts = [baseline]
    if not synthetic.empty:
        # Synthetic CSV may have different columns — keep just what we need
        synth_keep = pd.DataFrame({
            "text": synthetic["text"].astype(str),
            "category": synthetic["category"].astype(str),
            "priority": 1,
            "source": synthetic["source"].astype(str),
        })
        parts.append(synth_keep)
    if not qaym_amb.empty:
        parts.append(qaym_amb)

    combined = pd.concat(parts, ignore_index=True)
    # Drop empty texts
    combined = combined[combined["text"].astype(str).str.strip().str.len() > 0].reset_index(drop=True)

    # Write the CSV FIRST so a print crash can't lose the result
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[output] wrote {args.output} ({len(combined)} rows)")

    print()
    print(f"[output] category distribution:")
    for cat, n in combined["category"].value_counts().items():
        print(f"  {cat}: {n}")
    print(f"[output] source distribution (top 15):")
    for src, n in combined["source"].value_counts().head(15).items():
        print(f"  {src}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
