"""Triage a CSV of Arabic restaurant reviews into three buckets:

  1. ambience_candidates.csv     — likely ambience, send to labeling
  2. hard_negative_candidates.csv — looks ambience-ish but is clearly food/
                                    staff/delivery/etc., send to labeling
                                    as confirmed-negative for training
  3. review_needed.csv            — ambiguous, send to labeling meeting

The decision logic is intentionally conservative: when in doubt, push to
review_needed rather than guess. Over-labeling is what killed v3 ambience.

Reuses the project's `clean()` normalization (same Persian/Gulf char fold-in
as the production inference path). Does not depend on PySarf — keyword
filtering only.

Usage:

    python research/ambience_revival/filter_ambience.py \\
        --input data/raw/some_reviews.csv \\
        --text-col text \\
        --output-dir data/processed/ambience_triage

CSV inputs are expected to be UTF-8. Outputs are written as utf-8-sig so
they open cleanly in Excel.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Local import — the keyword vocab lives next to this script
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from keywords import (  # noqa: E402
    AMBIENCE_HIGH_PRECISION_PHRASES,
    AMBIENCE_SINGLE_TOKENS,
    ALL_HARD_NEGATIVES,
    RISKY_AMBIGUOUS,
    subtype_for_phrase,
)


# ---------------------------------------------------------------------------
# Arabic normalization — kept byte-identical with hf_space/app.py:clean()
# so any text we triage matches downstream production behavior.
# ---------------------------------------------------------------------------

TASHKEEL = re.compile(r"[ً-ٟ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")


def normalize_ar(text: str) -> str:
    """Same normalization the production inference pipeline uses."""
    if not isinstance(text, str) or not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "پ": "ب", "چ": "ج", "گ": "ك", "ک": "ك", "ی": "ي",
    }))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


# Pre-normalize the keyword lists once, so per-row matching is fast and
# orthography-tolerant.
_NORMALIZED_HIGH_PRECISION: list[str] = sorted(
    {normalize_ar(p) for p in AMBIENCE_HIGH_PRECISION_PHRASES if normalize_ar(p)},
    key=len, reverse=True,
)
_NORMALIZED_SINGLE_TOKENS: set[str] = {normalize_ar(t) for t in AMBIENCE_SINGLE_TOKENS if normalize_ar(t)}
_NORMALIZED_RISKY: set[str] = {normalize_ar(w) for w in RISKY_AMBIGUOUS if normalize_ar(w)}
_NORMALIZED_HARD_NEG: set[str] = {normalize_ar(w) for w in ALL_HARD_NEGATIVES if normalize_ar(w)}


# Build phrase → original-form map so we can report subtype tags
_PHRASE_TO_ORIGINAL: dict[str, str] = {}
for original in AMBIENCE_HIGH_PRECISION_PHRASES:
    n = normalize_ar(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)
for original in AMBIENCE_SINGLE_TOKENS:
    n = normalize_ar(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)


# ---------------------------------------------------------------------------
# Per-row scoring
# ---------------------------------------------------------------------------

@dataclass
class Triage:
    bucket: str                  # "ambience" | "hard_negative" | "review"
    high_precision_hits: list[str]
    single_token_hits: list[str]
    risky_hits: list[str]
    hard_negative_hits: list[str]
    subtype_tags: list[str]
    score: float
    reason: str


def score_row(text: str) -> Triage:
    """Decide which bucket a row belongs to.

    Decision rules (in order):

      1. Multi-word ambience phrase present (high precision) AND no
         hard-negative cue → ambience candidate.

      2. Single-word ambience token present AND no hard-negative cue AND
         no risky-ambiguous-only signal → ambience candidate (lower confidence).

      3. Hard-negative cue present (food / staff / delivery / etc.) AND
         any ambience signal present → hard_negative candidate
         (these are the confusable rows we deliberately want as negatives).

      4. Only risky-ambiguous tokens present, no anchor → review_needed.

      5. Nothing relevant → not written to any output (filtered out entirely).
    """
    cleaned = normalize_ar(text)
    if not cleaned:
        return Triage("none", [], [], [], [], [], 0.0, "empty_after_clean")

    words = set(cleaned.split())

    # Phrase matches — multi-word, high precision
    high_precision_hits = [p for p in _NORMALIZED_HIGH_PRECISION if p in cleaned]

    # Single-token matches — whole-word
    single_token_hits = sorted(words & _NORMALIZED_SINGLE_TOKENS)

    # Risky tokens — could be ambience or could be food/etc.
    risky_hits = sorted(words & _NORMALIZED_RISKY)

    # Hard-negative tokens
    hard_negative_hits = sorted(words & _NORMALIZED_HARD_NEG)
    # Also check multi-word hard-negative phrases
    for hn in _NORMALIZED_HARD_NEG:
        if " " in hn and hn in cleaned and hn not in hard_negative_hits:
            hard_negative_hits.append(hn)

    # Subtype tags — derived from any ambience hit
    subtype_tags: list[str] = []
    for hit in high_precision_hits + single_token_hits:
        original = _PHRASE_TO_ORIGINAL.get(hit, hit)
        st = subtype_for_phrase(original)
        if st and st not in subtype_tags:
            subtype_tags.append(st)

    # Score — coarse, just for sorting within each bucket
    score = (
        2.0 * len(high_precision_hits)
        + 1.0 * len(single_token_hits)
        + 0.3 * len(risky_hits)
    )

    # Decision logic
    # A "strong" single-token anchor must NOT also be a risky-ambiguous token.
    # Otherwise generic place words like 'المطعم' / 'الطاولة' (which ARE in
    # the ambience vocab but appear in every class) would falsely anchor any
    # row that mentions them.
    risky_set = set(risky_hits)
    strong_single_tokens = [t for t in single_token_hits if t not in risky_set]
    has_ambience_anchor = bool(high_precision_hits or strong_single_tokens)
    has_hard_negative = bool(hard_negative_hits)
    has_only_risky = (not has_ambience_anchor) and bool(risky_hits or single_token_hits)

    if has_ambience_anchor and has_hard_negative:
        return Triage(
            "hard_negative",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "ambience-words present alongside hard-negative cue — likely a non-ambience complaint that mentions place-words",
        )
    if has_ambience_anchor and not has_hard_negative:
        return Triage(
            "ambience",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "ambience phrase/token present, no hard-negative cue",
        )
    if has_only_risky:
        return Triage(
            "review",
            high_precision_hits, single_token_hits, risky_hits, hard_negative_hits,
            subtype_tags, score,
            "only ambiguous risky tokens present (e.g. حار / بارد / قديم) — needs human judgment",
        )
    return Triage("none", [], [], [], [], [], 0.0, "no relevant signal")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def write_bucket(rows: list[dict], path: Path) -> int:
    """Write a bucket as a UTF-8-with-BOM CSV (Excel-friendly). Returns count."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Triage Arabic restaurant reviews into ambience candidates / hard-negatives / review-needed."
    )
    p.add_argument("--input", required=True, type=Path, help="Input CSV path")
    p.add_argument("--text-col", default="text", help="Column name containing the review text (default: text)")
    p.add_argument("--output-dir", required=True, type=Path, help="Where to write the three output CSVs")
    p.add_argument("--limit", type=int, default=0, help="If > 0, only process the first N rows (for smoke tests)")
    args = p.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input, encoding="utf-8")
    if args.text_col not in df.columns:
        print(f"Column {args.text_col!r} not in input. Available: {list(df.columns)}", file=sys.stderr)
        return 2
    if args.limit > 0:
        df = df.head(args.limit)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Triaging on column {args.text_col!r}...")

    bucket_rows: dict[str, list[dict]] = {"ambience": [], "hard_negative": [], "review": []}
    for orig_idx, row in df.iterrows():
        text = row[args.text_col]
        result = score_row(text)
        if result.bucket == "none":
            continue
        bucket_rows[result.bucket].append({
            "row_id": orig_idx,
            "text": text,
            "high_precision_hits": " | ".join(result.high_precision_hits),
            "single_token_hits": " | ".join(result.single_token_hits),
            "risky_hits": " | ".join(result.risky_hits),
            "hard_negative_hits": " | ".join(result.hard_negative_hits),
            "subtype_tags": " | ".join(result.subtype_tags),
            "score": round(result.score, 2),
            "reason": result.reason,
            # Pass through any other useful columns the caller might want
            **{c: row[c] for c in df.columns if c != args.text_col and c not in {"row_id"}},
        })

    # Sort each bucket by score (highest first) so labelers see strongest cases first
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
