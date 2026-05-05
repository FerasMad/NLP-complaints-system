"""Triage Arabic restaurant reviews into ambience review queues.

This CLI is kept for backward compatibility.  The per-row decision logic now
lives in ``src/data/filter_ambience_core.py`` and is reused by the data
importers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from data.filter_ambience_core import (  # noqa: E402
    Triage,
    arabic_mask,
    record_from_filter,
    score_row,
    split_and_write_outputs,
)

__all__ = ["Triage", "score_row"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Triage Arabic restaurant reviews into ambience, hard-negative, and review buckets."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--text-col", default="text", help="Name of the review-text column")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for output CSVs")
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only process the first N rows")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.text_col not in df.columns:
        print(f"Column {args.text_col!r} not in input. Available: {list(df.columns)}", file=sys.stderr)
        return 2
    if args.limit > 0:
        df = df.head(args.limit)

    mask = arabic_mask(df[args.text_col])
    arabic_df = df[mask]
    rows = []
    for idx, row in arabic_df.iterrows():
        metadata = {c: row[c] for c in row.index if c != args.text_col}
        rec = record_from_filter(
            row_id=idx,
            text=str(row[args.text_col]),
            source="manual",
            source_detail=str(args.input),
            metadata=metadata,
        )
        if rec:
            rows.append(rec)

    summary = split_and_write_outputs(
        rows,
        args.output_dir,
        input_path=str(args.input),
        source_name="manual",
        row_count_raw=len(df),
        row_count_arabic=len(arabic_df),
        script_name=Path(__file__).name,
    )

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Arabic rows: {len(arabic_df)}")
    print(f"ambience_candidates.csv: {summary['ambience_candidates_count']}")
    print(f"hard_negative_candidates.csv: {summary['hard_negative_candidates_count']}")
    print(f"review_needed.csv: {summary['review_needed_count']}")
    print(f"Written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
