"""Triage Arabic restaurant reviews into ambience / hard-negative / review buckets.

Reads a CSV of reviews and writes three output CSVs split by signal strength:

  ambience_candidates.csv         — likely ambience, send to labeling
  hard_negative_candidates.csv    — looks ambience-ish but mentions food/staff/
                                    delivery/etc., send to labeling as
                                    confirmed-negative training data
  review_needed.csv               — only ambiguous "risky" tokens (حار, بارد,
                                    قديم, ريحة, زيت, برا, المطعم, etc.) —
                                    needs human judgment

Decision logic is intentionally conservative: when in doubt, push to
review_needed rather than guess. Over-labeling killed v3 ambience.

Reuses the production normalization (`src/utils/arabic_normalization.py`)
and the canonical keyword vocabulary (`src/config/ambience_keywords.py`)
so triage decisions match what downstream training/inference will see.

Usage:

    python src/filter_ambience_candidates.py \\
        --input data/raw/some_reviews.csv \\
        --text-col text \\
        --output-dir data/processed/ambience_triage

CSV inputs are expected to be UTF-8. Outputs are written as utf-8-sig so
they open cleanly in Excel for hand-labeling.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Local imports — make src/ importable when running this script directly
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
from utils.arabic_normalization import clean
from config.ambience_keywords import (
    AMBIENCE_HIGH_PRECISION_PHRASES,
    AMBIENCE_SINGLE_TOKENS,
    ALL_HARD_NEGATIVES,
    RISKY_AMBIGUOUS,
    subtype_for_phrase,
)


# ---------------------------------------------------------------------------
# Pre-normalize keyword lists once at import — orthography-tolerant matching.
# ---------------------------------------------------------------------------

_NORMALIZED_HIGH_PRECISION: list[str] = sorted(
    {clean(p) for p in AMBIENCE_HIGH_PRECISION_PHRASES if clean(p)},
    key=len, reverse=True,
)
_NORMALIZED_SINGLE_TOKENS: set[str] = {clean(t) for t in AMBIENCE_SINGLE_TOKENS if clean(t)}
_NORMALIZED_RISKY: set[str] = {clean(w) for w in RISKY_AMBIGUOUS if clean(w)}
_NORMALIZED_HARD_NEG: set[str] = {clean(w) for w in ALL_HARD_NEGATIVES if clean(w)}

# Phrase → original surface form, so we can report subtype tags
_PHRASE_TO_ORIGINAL: dict[str, str] = {}
for original in AMBIENCE_HIGH_PRECISION_PHRASES:
    n = clean(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)
for original in AMBIENCE_SINGLE_TOKENS:
    n = clean(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)


# ---------------------------------------------------------------------------
# Per-row scoring
# ---------------------------------------------------------------------------

@dataclass
class Triage:
    bucket: str                  # "ambience" | "hard_negative" | "review" | "none"
    high_precision_hits: list[str] = field(default_factory=list)
    single_token_hits: list[str] = field(default_factory=list)
    risky_hits: list[str] = field(default_factory=list)
    hard_negative_hits: list[str] = field(default_factory=list)
    subtype_tags: list[str] = field(default_factory=list)
    score: float = 0.0
    reason: str = ""


def score_row(text: str) -> Triage:
    """Decide which bucket a row belongs to.

    Decision rules in order:

    1. Ambience anchor (multi-word phrase OR strong single token) AND no
       hard-negative cue  -> ambience candidate.

    2. Ambience anchor AND hard-negative cue  -> hard_negative candidate.
       (We deliberately want these as confirmed-negative training data.)

    3. Only risky/ambiguous tokens, no strong anchor  -> review_needed.

    4. Nothing relevant  -> filtered out (not written to any output).

    A "strong" single token excludes risky-ambiguous ones. Place words like
    'المطعم', 'الطاولة', 'المكان' appear in every class — they shouldn't
    anchor a row to ambience by themselves.
    """
    cleaned = clean(text)
    if not cleaned:
        return Triage("none", reason="empty_after_clean")

    words = set(cleaned.split())

    # Multi-word phrase matches (high precision)
    high_precision_hits = [p for p in _NORMALIZED_HIGH_PRECISION if p in cleaned]

    # Single-token matches — whole word
    single_token_hits = sorted(words & _NORMALIZED_SINGLE_TOKENS)

    # Risky tokens
    risky_hits = sorted(words & _NORMALIZED_RISKY)

    # Hard-negative tokens (single-word matches + multi-word substrings)
    hard_negative_hits = sorted(words & _NORMALIZED_HARD_NEG)
    for hn in _NORMALIZED_HARD_NEG:
        if " " in hn and hn in cleaned and hn not in hard_negative_hits:
            hard_negative_hits.append(hn)

    # Subtype tags from any ambience hit
    subtype_tags: list[str] = []
    for hit in high_precision_hits + single_token_hits:
        original = _PHRASE_TO_ORIGINAL.get(hit, hit)
        st = subtype_for_phrase(original)
        if st and st not in subtype_tags:
            subtype_tags.append(st)

    # Coarse score for sorting within each bucket
    score = (
        2.0 * len(high_precision_hits)
        + 1.0 * len(single_token_hits)
        + 0.3 * len(risky_hits)
    )

    # Subtract risky tokens from "strong" single-token signal so generic
    # place words don't false-anchor anything to ambience.
    risky_set = set(risky_hits)
    strong_single_tokens = [t for t in single_token_hits if t not in risky_set]
    has_ambience_anchor = bool(high_precision_hits or strong_single_tokens)
    # Multi-word hard-neg phrases ("الحمام وصخ", "بقع طعام") are high-precision
    # signals of confirmed non-ambience content. Single-word hard-neg matches
    # alone (e.g. "الموظف" appearing in a pure staff complaint) are too
    # noisy to bucket — those rows fall through to "none".
    has_multi_word_hard_neg = any(" " in hn for hn in hard_negative_hits)
    has_hard_negative = bool(hard_negative_hits)
    has_only_risky = (not has_ambience_anchor) and bool(risky_hits or single_token_hits)

    if has_ambience_anchor and has_hard_negative:
        return Triage(
            "hard_negative",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "ambience-words present alongside hard-negative cue — likely a "
            "non-ambience complaint that mentions place-words",
        )
    if has_ambience_anchor and not has_hard_negative:
        return Triage(
            "ambience",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "ambience phrase/token present, no hard-negative cue",
        )
    if has_multi_word_hard_neg:
        # Multi-word hard-neg phrase like "الحمام وصخ" or "بقع طعام" —
        # confirmed non-ambience content even without an ambience anchor.
        # Route to hard_negative so labelers can confirm and use as training
        # data for the model to learn the boundary (bathroom-cleanliness vs
        # bathroom-facilities-ambience).
        return Triage(
            "hard_negative",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "multi-word hard-negative phrase present — confirmed non-ambience",
        )
    if has_only_risky:
        return Triage(
            "review",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "only ambiguous risky tokens present (e.g. حار / بارد / قديم) — "
            "needs human judgment",
        )
    return Triage("none", reason="no relevant signal")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

# Output column order — extends ChatGPT's 19-column ambience schema.
# Caller-provided columns from the input CSV are passed through after these.
_OUTPUT_COLS = [
    "row_id",
    "text",
    "weak_label",                  # ambience | hard_negative | review_needed
    "ambience_subtype",            # primary subtype tag, if any
    "ambience_subtype_all",        # | -separated all detected subtypes
    "contains_ambience_keyword",   # bool
    "is_hard_negative",            # bool
    "risk_keyword",                # | -separated risky words present
    "high_precision_hits",         # | -separated multi-word ambience phrases
    "single_token_hits",           # | -separated single-token ambience matches
    "hard_negative_hits",          # | -separated hard-neg cues
    "score",                       # for sort within bucket
    "review_status",               # always "unreviewed" out of the script
    "notes",                       # auto-generated decision reason
]


def _row_to_record(orig_idx: int, row: pd.Series, text_col: str, t: Triage) -> dict:
    """Translate a triage decision into the canonical output row schema."""
    weak_label = {
        "ambience": "ambience_candidate",
        "hard_negative": "hard_negative_candidate",
        "review": "review_needed",
        "none": "none",
    }[t.bucket]
    rec = {
        "row_id": orig_idx,
        "text": row[text_col],
        "weak_label": weak_label,
        "ambience_subtype": t.subtype_tags[0] if t.subtype_tags else "",
        "ambience_subtype_all": " | ".join(t.subtype_tags),
        "contains_ambience_keyword": bool(t.high_precision_hits or t.single_token_hits),
        "is_hard_negative": t.bucket == "hard_negative",
        "risk_keyword": " | ".join(t.risky_hits),
        "high_precision_hits": " | ".join(t.high_precision_hits),
        "single_token_hits": " | ".join(t.single_token_hits),
        "hard_negative_hits": " | ".join(t.hard_negative_hits),
        "score": round(t.score, 2),
        "review_status": "unreviewed",
        "notes": t.reason,
    }
    # Pass through any other input columns
    for c in row.index:
        if c == text_col:
            continue
        if c not in rec:
            rec[c] = row[c]
    return rec


def write_bucket(rows: list[dict], path: Path) -> int:
    """Write a bucket as a UTF-8-with-BOM CSV (Excel-friendly)."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # Reorder so the ambience-specific columns lead; pass-through columns trail
    leading = [c for c in _OUTPUT_COLS if c in df.columns]
    trailing = [c for c in df.columns if c not in leading]
    df = df[leading + trailing]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Triage Arabic restaurant reviews into ambience candidates, "
            "hard-negatives, and review-needed buckets."
        )
    )
    p.add_argument("--input", required=True, type=Path, help="Input CSV path")
    p.add_argument("--text-col", default="text",
                   help="Name of the column containing review text (default: text)")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to write the three output CSVs")
    p.add_argument("--limit", type=int, default=0,
                   help="If > 0, only process the first N rows (for smoke tests)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.text_col not in df.columns:
        print(f"Column {args.text_col!r} not in input. Available: {list(df.columns)}",
              file=sys.stderr)
        return 2
    if args.limit > 0:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Triaging on column {args.text_col!r}...")

    bucket_rows: dict[str, list[dict]] = {"ambience": [], "hard_negative": [], "review": []}
    for orig_idx, row in df.iterrows():
        result = score_row(row[args.text_col])
        if result.bucket == "none":
            continue
        bucket_rows[result.bucket].append(_row_to_record(orig_idx, row, args.text_col, result))

    # Sort each bucket by score desc — strongest cases at top for the labeler
    for bucket in bucket_rows.values():
        bucket.sort(key=lambda r: -r["score"])

    n_amb = write_bucket(bucket_rows["ambience"], args.output_dir / "ambience_candidates.csv")
    n_neg = write_bucket(bucket_rows["hard_negative"], args.output_dir / "hard_negative_candidates.csv")
    n_rev = write_bucket(bucket_rows["review"], args.output_dir / "review_needed.csv")

    n_in = len(df)
    n_filtered = n_in - n_amb - n_neg - n_rev
    print()
    print(f"  ambience_candidates.csv:       {n_amb:>5}  ({n_amb / n_in:.1%})")
    print(f"  hard_negative_candidates.csv:  {n_neg:>5}  ({n_neg / n_in:.1%})")
    print(f"  review_needed.csv:             {n_rev:>5}  ({n_rev / n_in:.1%})")
    print(f"  filtered out (no signal):      {n_filtered:>5}  ({n_filtered / n_in:.1%})")
    print()
    print(f"Written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
