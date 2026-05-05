"""Import locally downloaded AraMA/AraMAMS aspect data for ambience review."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.filter_ambience_core import (  # noqa: E402
    AMBIENCE_CATEGORY,
    FOOD_CATEGORY,
    PRICE_CATEGORY,
    SERVICE_CATEGORY,
    arabic_mask,
    detect_text_column,
    read_table,
    schema_record,
    split_and_write_outputs,
)

ASPECT_ALIASES = {
    "environment": AMBIENCE_CATEGORY,
    "ambience": AMBIENCE_CATEGORY,
    "atmosphere": AMBIENCE_CATEGORY,
    "place": AMBIENCE_CATEGORY,
    "الجو": AMBIENCE_CATEGORY,
    "المكان": AMBIENCE_CATEGORY,
    "food": FOOD_CATEGORY,
    "taste": FOOD_CATEGORY,
    "الطعام": FOOD_CATEGORY,
    "الأكل": FOOD_CATEGORY,
    "service": SERVICE_CATEGORY,
    "staff": SERVICE_CATEGORY,
    "الخدمة": SERVICE_CATEGORY,
    "الموظفين": SERVICE_CATEGORY,
    "price": PRICE_CATEGORY,
    "value": PRICE_CATEGORY,
    "السعر": PRICE_CATEGORY,
    "القيمة": PRICE_CATEGORY,
}


def _split_aspects(value: Any) -> list[str]:
    text = str(value or "").strip().lower()
    if not text or text == "nan":
        return []
    return [part.strip() for part in re.split(r"[,;/|+&]+", text) if part.strip()]


def _map_aspects(value: Any) -> list[str]:
    mapped = []
    for aspect in _split_aspects(value):
        for key, category in ASPECT_ALIASES.items():
            if key in aspect and category not in mapped:
                mapped.append(category)
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description="Import AraMA/AraMAMS local files into ambience queues.")
    parser.add_argument("--input", required=True, type=Path, help="CSV/XLSX/JSON/JSONL downloaded by the user")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--text-col", default=None)
    parser.add_argument("--aspect-col", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    errors: list[str] = []
    df = read_table(args.input)
    if args.limit > 0:
        df = df.head(args.limit)
    text_col = detect_text_column(df, args.text_col)
    aspect_col = args.aspect_col
    if not aspect_col:
        candidates = ["aspect", "aspects", "aspect_label", "category", "label"]
        aspect_col = next((c for c in candidates if c in df.columns), None)
    if not aspect_col or aspect_col not in df.columns:
        print(f"Aspect column not found. Use --aspect-col. Available: {list(df.columns)}", file=sys.stderr)
        return 2

    arabic_df = df[arabic_mask(df[text_col])]
    rows = []
    for idx, row in arabic_df.iterrows():
        categories = _map_aspects(row[aspect_col])
        if not categories:
            continue
        has_env = AMBIENCE_CATEGORY in categories
        is_multi = len(categories) > 1
        metadata = {
            c: row[c]
            for c in row.index
            if c not in {text_col, aspect_col} and c in {"rating", "city", "restaurant_name", "domain", "source_url"}
        }
        metadata.update({"source_aspect": row[aspect_col], "is_multi_aspect": is_multi})

        if has_env:
            rows.append(
                schema_record(
                    row_id=idx,
                    text=str(row[text_col]),
                    source="arama",
                    source_detail=str(args.input),
                    weak_label="ambience_candidate",
                    category=AMBIENCE_CATEGORY,
                    contains_ambience_keyword=True,
                    is_hard_negative=False,
                    label_confidence="source_aspect_label",
                    review_status="needs_discussion" if is_multi else "unreviewed",
                    notes="AraMA/AraMAMS environment aspect label",
                    metadata=metadata,
                )
            )
        else:
            category = categories[0]
            rows.append(
                schema_record(
                    row_id=idx,
                    text=str(row[text_col]),
                    source="arama",
                    source_detail=str(args.input),
                    weak_label="hard_negative_candidate",
                    category=category,
                    contains_ambience_keyword=False,
                    is_hard_negative=True,
                    label_confidence="source_aspect_label",
                    review_status="unreviewed",
                    notes=f"AraMA/AraMAMS non-environment aspect label: {category}",
                    metadata=metadata,
                )
            )

    summary = split_and_write_outputs(
        rows,
        args.output_dir,
        input_path=str(args.input),
        source_name="arama",
        row_count_raw=len(df),
        row_count_arabic=len(arabic_df),
        script_name=Path(__file__).name,
        errors=errors,
    )
    print(f"Wrote {args.output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
