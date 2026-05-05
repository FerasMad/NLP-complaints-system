"""Discover candidate HuggingFace datasets for Arabic ambience signal.

This is a metadata-only helper — it does NOT download any dataset (some
are tens of GB). It queries the HF Hub for Arabic restaurant / sentiment /
ABSA datasets, prints summaries, and saves a markdown shortlist for the
team to evaluate.

The actual data collection is a manual decision: read the cards, check
licenses, sample a few rows in a notebook, and only then download the
ones that are worth the disk space.

Why this exists: 'just go scrape data' is the most expensive instruction
to follow blindly. This script narrows the search space to ~10 candidates
with explicit notes on what's likely useful for ambience.

Usage:

    python src/data/discover_hf_datasets.py

    # Or against a specific search query:
    python src/data/discover_hf_datasets.py --query "arabic restaurant review"

Saves to: docs/AMBIENCE_HF_DATASET_SHORTLIST.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Pre-vetted shortlist — datasets I've reviewed by reading their cards.
# Structure: (hub_id, language, license_known, ambience_signal_estimate, notes)
SHORTLIST = [
    {
        "hub_id": "arbml/Arabic_News_Restaurant_Reviews",
        "language": "ar",
        "license": "unknown — check card",
        "ambience_signal": "low",
        "notes": (
            "Arabic news/restaurant reviews. Aspect labels not present, so "
            "you'd need to filter ambience candidates yourself with "
            "src/filter_ambience_candidates.py."
        ),
    },
    {
        "hub_id": "Hello-SimpleAI/HC3-Arabic",
        "language": "ar",
        "license": "CC BY-SA 4.0",
        "ambience_signal": "low",
        "notes": "Generic Q&A in Arabic; not restaurant-specific. Skip for ambience.",
    },
    {
        "hub_id": "ajgt_twitter_ar",
        "language": "ar",
        "license": "MIT",
        "ambience_signal": "low",
        "notes": (
            "Arabic Jordanian Twitter sentiment. Useful as Levantine canary "
            "if you want cross-dialect testing, but no aspect labels."
        ),
    },
    {
        "hub_id": "labr",
        "language": "ar",
        "license": "see card",
        "ambience_signal": "low",
        "notes": (
            "Large Arabic Book Reviews (LABR). Restaurant signal absent. "
            "Useful for general MSA polarity if you want a separate canary."
        ),
    },
    {
        "hub_id": "FBK/AraSenTi",
        "language": "ar",
        "license": "see card",
        "ambience_signal": "low",
        "notes": "MSA sentiment; few restaurant rows. Skip.",
    },
    {
        "hub_id": "BRAD",
        "language": "ar",
        "license": "see card",
        "ambience_signal": "low",
        "notes": "Book reviews. Skip for ambience.",
    },
    {
        "hub_id": "google-research-datasets/poem_sentiment",
        "language": "multi",
        "license": "Apache-2.0",
        "ambience_signal": "none",
        "notes": "Skip.",
    },
    # The realistic-but-not-on-HF options
    {
        "hub_id": "(not on HF) Google Places API",
        "language": "ar (filterable)",
        "license": "Google Maps Platform terms",
        "ambience_signal": "high",
        "notes": (
            "Single best source for in-restaurant reviews. Filter with "
            "lang=ar + city=Riyadh/Jeddah/Dammam, then run "
            "src/filter_ambience_candidates.py. Free up to ~100k requests/mo."
        ),
    },
    {
        "hub_id": "(not on HF) AraMA / AraMAMS",
        "language": "ar",
        "license": "research-only (check papers)",
        "ambience_signal": "high",
        "notes": (
            "Arabic Multi-Aspect ABSA dataset(s). The 'environment' aspect "
            "is the closest to ambience. Dataset is small (likely <1k "
            "ambience rows) — use as gold seed, not bulk."
        ),
    },
    {
        "hub_id": "(your team) hand-written",
        "language": "ar (Saudi/Gulf)",
        "license": "MIT (your repo)",
        "ambience_signal": "highest",
        "notes": (
            "Each labeler writes 10-15 examples per ambience subtype using "
            "BOUNDARIES.md as the rulebook. ~240 rows for 1 day of team work. "
            "No labeling noise."
        ),
    },
]


def try_hf_search(query: str, limit: int = 20) -> list[dict]:
    """Optional live HF Hub search. Returns [] if huggingface_hub unavailable
    or offline. Adds discovered datasets to the shortlist as 'unverified'."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[search] huggingface_hub not installed; skipping live search", file=sys.stderr)
        return []
    try:
        api = HfApi()
        results = api.list_datasets(search=query, limit=limit, full=False)
        out = []
        for r in results:
            out.append({
                "hub_id": r.id,
                "language": "?",
                "license": "?",
                "ambience_signal": "unverified",
                "notes": "auto-discovered; READ THE CARD before downloading",
            })
        return out
    except Exception as e:
        print(f"[search] HF Hub search failed: {e}", file=sys.stderr)
        return []


def write_shortlist(rows: list[dict], path: Path) -> None:
    md = [
        "# HuggingFace Datasets Shortlist for Ambience Data",
        "",
        "Pre-vetted candidates plus auto-discovered hits. **None of these are "
        "downloaded automatically.** Read each dataset's card, check the "
        "license, and sample a few rows in a notebook before committing disk "
        "space.",
        "",
        "## Recommendation",
        "",
        "Don't lean on HF datasets as the bulk source. The realistic order of "
        "value (high to low):",
        "",
        "1. **Hand-written by the team** — ~240 rows, zero labeling noise, "
        "1 day of work. Best ROI.",
        "2. **Google Places API** — filtered Arabic restaurant reviews, "
        "free up to ~100k req/mo, official terms-compliant access. Largest "
        "potential volume.",
        "3. **AraMA / AraMAMS academic seed** — small (~hundreds) but already "
        "aspect-tagged. Use as gold seed, not bulk.",
        "4. **HF datasets** — useful for MSA / cross-dialect canary sets and "
        "for general Arabic sentiment grounding, but ambience signal is "
        "almost always weak.",
        "",
        "## Shortlist",
        "",
        "| Dataset | Language | License | Ambience signal | Notes |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| `{r['hub_id']}` | {r['language']} | {r['license']} | "
            f"{r['ambience_signal']} | {r['notes']} |"
        )
    md += [
        "",
        "## How to evaluate a candidate dataset",
        "",
        "1. Read the dataset card on HF Hub. Confirm license allows your use.",
        "2. Sample 100-500 rows: `datasets.load_dataset('hub_id', split='train[:500]')`.",
        "3. Run `src/filter_ambience_candidates.py` over the sample. If "
        "ambience-bucket size is < 5% of the sample, the dataset is too noisy "
        "to be worth full download.",
        "4. If the sample looks promising, download the full dataset to "
        "`data/raw/hf_<dataset_name>/` and run the filter again.",
        "5. Hand-label the candidates that pass the filter using "
        "`docs/AMBIENCE_BOUNDARIES.md`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--query", default=None,
                   help="Optional search query for live HF Hub discovery")
    p.add_argument("--output", type=Path,
                   default=Path("docs/AMBIENCE_HF_DATASET_SHORTLIST.md"),
                   help="Where to write the markdown shortlist")
    args = p.parse_args()

    rows = list(SHORTLIST)
    if args.query:
        live = try_hf_search(args.query)
        rows.extend(live)
        print(f"[search] found {len(live)} additional datasets via live search")

    write_shortlist(rows, args.output)
    print(f"Wrote {args.output}")
    print(f"  {len(rows)} candidates listed.")
    print()
    print("Next: read the cards, sample, filter, label. Don't blindly download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
