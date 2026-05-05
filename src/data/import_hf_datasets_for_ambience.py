"""Import Hugging Face Arabic review datasets as weak ambience candidates."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.filter_ambience_core import (  # noqa: E402
    arabic_mask,
    detect_text_column,
    record_from_filter,
    split_and_write_outputs,
)

DEFAULT_DATASETS = ["hadyelsahar/ar_res_reviews"]
PRESERVE_COLUMNS = {"polarity", "rating", "restaurant_id", "domain"}


def _dataset_to_frame(dataset_name: str, split: str | None) -> pd.DataFrame:
    from datasets import load_dataset

    loaded = load_dataset(dataset_name, split=split) if split else load_dataset(dataset_name)
    if isinstance(loaded, dict):
        frames = []
        for split_name, ds in loaded.items():
            frame = ds.to_pandas()
            frame["hf_split"] = split_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return loaded.to_pandas()


def main() -> int:
    parser = argparse.ArgumentParser(description="Load vetted HF Arabic review datasets into ambience queues.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", action="append", default=None, help="HF dataset id; repeatable")
    parser.add_argument("--split", default=None, help="Optional split passed to datasets.load_dataset")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    datasets = args.dataset or DEFAULT_DATASETS
    rows = []
    errors: list[str] = []
    total_raw = 0
    total_arabic = 0

    for dataset_name in datasets:
        try:
            df = _dataset_to_frame(dataset_name, args.split)
        except Exception as exc:  # noqa: BLE001 - continue on external dataset failures
            errors.append(f"{dataset_name}: {exc}")
            continue

        if args.limit > 0:
            df = df.head(args.limit)
        total_raw += len(df)
        try:
            text_col = detect_text_column(df)
        except ValueError as exc:
            errors.append(f"{dataset_name}: {exc}")
            continue
        arabic_df = df[arabic_mask(df[text_col])]
        total_arabic += len(arabic_df)

        for idx, row in arabic_df.iterrows():
            metadata = {c: row[c] for c in row.index if c != text_col and c in PRESERVE_COLUMNS | {"hf_split"}}
            rec = record_from_filter(
                row_id=f"{dataset_name}:{idx}",
                text=str(row[text_col]),
                source="huggingface",
                source_detail=dataset_name,
                label_confidence="weak",
                metadata=metadata,
            )
            if rec:
                rows.append(rec)

    summary = split_and_write_outputs(
        rows,
        args.output_dir,
        input_path=",".join(datasets),
        source_name="huggingface",
        row_count_raw=total_raw,
        row_count_arabic=total_arabic,
        script_name=Path(__file__).name,
        errors=errors,
    )
    print(f"Wrote {args.output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
