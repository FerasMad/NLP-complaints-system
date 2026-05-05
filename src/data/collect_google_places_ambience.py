"""Optional Google Places API collector for ambience candidate reviews.

Uses official Places API endpoints only.  It does not scrape Google Maps HTML.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from data.filter_ambience_core import (  # noqa: E402
    arabic_mask,
    record_from_filter,
    split_and_write_outputs,
)

TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def _get_json(url: str, params: dict[str, Any], *, sleep_seconds: float) -> dict[str, Any]:
    import requests

    for attempt in range(4):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(sleep_seconds * (2**attempt))
            continue
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "OVER_QUERY_LIMIT":
            time.sleep(sleep_seconds * (2**attempt))
            continue
        return payload
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect limited Google Places API reviews for ambience queues.")
    parser.add_argument("--query", action="append", default=None, help="Text Search query; repeatable")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/google_places"))
    parser.add_argument("--max-places-per-query", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        print("GOOGLE_PLACES_API_KEY not found. Skipping optional Google Places collector.")
        return 0

    queries = args.query or ["مطاعم الرياض", "مطاعم جدة", "مطاعم الخبر"]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_dir / "google_places_raw.jsonl"
    extracted_rows = []
    errors: list[str] = []

    with raw_path.open("w", encoding="utf-8") as raw_file:
        for query in queries:
            try:
                search_payload = _get_json(
                    TEXT_SEARCH_URL,
                    {"query": query, "key": api_key, "language": "ar"},
                    sleep_seconds=args.sleep,
                )
                raw_file.write(json.dumps({"type": "text_search", "query": query, "payload": search_payload}, ensure_ascii=False) + "\n")
                places = search_payload.get("results", [])[: args.max_places_per_query]
            except Exception as exc:  # noqa: BLE001
                errors.append(f"text search {query}: {exc}")
                continue

            for place in places:
                place_id = place.get("place_id")
                if not place_id:
                    continue
                try:
                    details = _get_json(
                        DETAILS_URL,
                        {
                            "place_id": place_id,
                            "fields": "place_id,name,formatted_address,rating,reviews",
                            "key": api_key,
                            "language": "ar",
                        },
                        sleep_seconds=args.sleep,
                    )
                    raw_file.write(json.dumps({"type": "details", "query": query, "payload": details}, ensure_ascii=False) + "\n")
                    result = details.get("result", {})
                    for review_idx, review in enumerate(result.get("reviews", []) or []):
                        extracted_rows.append(
                            {
                                "id": f"{place_id}:{review_idx}",
                                "text": review.get("text", ""),
                                "restaurant_name": result.get("name", ""),
                                "rating": review.get("rating", ""),
                                "place_rating": result.get("rating", ""),
                                "place_id": place_id,
                                "source_url": review.get("author_url", ""),
                                "domain": "google_places",
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"details {place_id}: {exc}")
                time.sleep(args.sleep)

    rows = []
    arabic_count = 0
    for row in extracted_rows:
        if not arabic_mask(pd.Series([row["text"]])).iloc[0]:
            continue
        arabic_count += 1
        rec = record_from_filter(
            row_id=row["id"],
            text=str(row["text"]),
            source="google_places_api",
            source_detail="; ".join(queries),
            label_confidence="weak",
            metadata={k: v for k, v in row.items() if k not in {"id", "text"}},
        )
        if rec:
            rec["restaurant_name"] = row.get("restaurant_name", "")
            rows.append(rec)

    summary = split_and_write_outputs(
        rows,
        args.output_dir,
        input_path=str(raw_path),
        source_name="google_places_api",
        row_count_raw=len(extracted_rows),
        row_count_arabic=arabic_count,
        script_name=Path(__file__).name,
        errors=errors,
    )
    print("Note: Places API returns only a limited number of reviews per place.")
    print(f"Wrote {args.output_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
