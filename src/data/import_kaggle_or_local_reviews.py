"""Import generic local Arabic review files into ambience review queues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.filter_ambience_core import (  # noqa: E402
    arabic_mask,
    detect_text_column,
    read_table,
    record_from_filter,
    split_and_write_outputs,
)

PRESERVE_COLUMNS = {"rating", "city", "restaurant_name", "domain", "source_url"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import local Arabic review CSV/XLSX/JSON/JSONL files.")
    parser.add_argument("--input", required=True, type=Path, help="File under data/raw/ or another local path")
    parser.add_argument("--source-name", required=True, help="Source detail/name, e.g. kaggle_arabic_reviews")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-col", required=True, help="Column containing review text")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    df = read_table(args.input)
    if args.limit > 0:
        df = df.head(args.limit)
    text_col = detect_text_column(df, args.text_col)
    arabic_df = df[arabic_mask(df[text_col])]

    source = args.source_name
    rows = []
    for idx, row in arabic_df.iterrows():
        metadata = {c: row[c] for c in row.index if c != text_col and c in PRESERVE_COLUMNS}
        city = str(row["city"]) if "city" in row.index and not row.isna()["city"] else ""
        restaurant = (
            str(row["restaurant_name"])
            if "restaurant_name" in row.index and not row.isna()["restaurant_name"]
            else ""
        )
        rec = record_from_filter(
            row_id=idx,
            text=str(row[text_col]),
            source=source,
            source_detail=args.source_name,
            label_confidence="weak",
            metadata=metadata,
        )
        if rec:
            rec["city"] = city
            rec["restaurant_name"] = restaurant
            rows.append(rec)

    summary = split_and_write_outputs(
        rows,
        args.output_dir,
        input_path=str(args.input),
        source_name=args.source_name,
        row_count_raw=len(df),
        row_count_arabic=len(arabic_df),
        script_name=Path(__file__).name,
    )
    print(f"Wrote {args.output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
