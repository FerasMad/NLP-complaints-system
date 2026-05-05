"""Reusable ambience candidate filtering and output helpers.

This module is the shared core for v5 ambience data acquisition.  The
per-row triage logic is extracted from ``src/filter_ambience_candidates.py``
so importers and the legacy CLI make the same conservative decisions.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.ambience_keywords import (  # noqa: E402
    ALL_HARD_NEGATIVES,
    AMBIENCE_HIGH_PRECISION_PHRASES,
    AMBIENCE_SINGLE_TOKENS,
    RISKY_AMBIGUOUS,
    subtype_for_phrase,
)
from utils.arabic_normalization import clean  # noqa: E402

AMBIENCE_CATEGORY = "الجو والمكان"
FOOD_CATEGORY = "جودة الطعام"
SERVICE_CATEGORY = "خدمة الموظفين"
PRICE_CATEGORY = "السعر والقيمة"

SCHEMA_COLUMNS = [
    "id",
    "text",
    "category",
    "label",
    "source",
    "source_detail",
    "city",
    "restaurant_name",
    "dialect",
    "weak_label",
    "contains_ambience_keyword",
    "ambience_subtype",
    "risk_keyword",
    "is_hard_negative",
    "is_synthetic",
    "label_confidence",
    "review_status",
    "notes",
    "split",
]

TEXT_COLUMN_CANDIDATES = ["text", "review", "review_text", "comment", "content"]
ARABIC_RE = r"[\u0600-\u06FF]"


@dataclass
class Triage:
    bucket: str
    high_precision_hits: list[str] = field(default_factory=list)
    single_token_hits: list[str] = field(default_factory=list)
    risky_hits: list[str] = field(default_factory=list)
    hard_negative_hits: list[str] = field(default_factory=list)
    subtype_tags: list[str] = field(default_factory=list)
    score: float = 0.0
    reason: str = ""


_NORMALIZED_HIGH_PRECISION: list[str] = sorted(
    {clean(p) for p in AMBIENCE_HIGH_PRECISION_PHRASES if clean(p)},
    key=len,
    reverse=True,
)
_NORMALIZED_SINGLE_TOKENS: set[str] = {clean(t) for t in AMBIENCE_SINGLE_TOKENS if clean(t)}
_NORMALIZED_RISKY: set[str] = {clean(w) for w in RISKY_AMBIGUOUS if clean(w)}
_NORMALIZED_HARD_NEG: set[str] = {clean(w) for w in ALL_HARD_NEGATIVES if clean(w)}

_PHRASE_TO_ORIGINAL: dict[str, str] = {}
for original in AMBIENCE_HIGH_PRECISION_PHRASES:
    n = clean(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)
for original in AMBIENCE_SINGLE_TOKENS:
    n = clean(original)
    if n:
        _PHRASE_TO_ORIGINAL.setdefault(n, original)


def score_row(text: str) -> Triage:
    """Decide whether a row is ambience, hard negative, review, or none."""
    cleaned = clean(text)
    if not cleaned:
        return Triage("none", reason="empty_after_clean")

    words = set(cleaned.split())
    high_precision_hits = [p for p in _NORMALIZED_HIGH_PRECISION if p in cleaned]
    single_token_hits = sorted(words & _NORMALIZED_SINGLE_TOKENS)
    risky_hits = sorted(words & _NORMALIZED_RISKY)

    hard_negative_hits = sorted(words & _NORMALIZED_HARD_NEG)
    for hn in _NORMALIZED_HARD_NEG:
        if " " in hn and hn in cleaned and hn not in hard_negative_hits:
            hard_negative_hits.append(hn)

    subtype_tags: list[str] = []
    for hit in high_precision_hits + single_token_hits:
        original = _PHRASE_TO_ORIGINAL.get(hit, hit)
        subtype = subtype_for_phrase(original)
        if subtype and subtype not in subtype_tags:
            subtype_tags.append(subtype)

    score = 2.0 * len(high_precision_hits) + 1.0 * len(single_token_hits) + 0.3 * len(risky_hits)
    risky_set = set(risky_hits)
    strong_single_tokens = [t for t in single_token_hits if t not in risky_set]
    has_ambience_anchor = bool(high_precision_hits or strong_single_tokens)
    has_multi_word_hard_neg = any(" " in hn for hn in hard_negative_hits)
    has_hard_negative = bool(hard_negative_hits)
    has_only_risky = (not has_ambience_anchor) and bool(risky_hits or single_token_hits)

    if has_ambience_anchor and has_hard_negative:
        return Triage(
            "hard_negative",
            high_precision_hits,
            single_token_hits,
            risky_hits,
            hard_negative_hits,
            subtype_tags,
            score,
            "ambience-words present alongside hard-negative cue - likely a non-ambience complaint",
        )
    if has_ambience_anchor and not has_hard_negative:
        return Triage(
            "ambience",
            high_precision_hits,
            single_token_hits,
            risky_hits,
            hard_negative_hits,
            subtype_tags,
            score,
            "ambience phrase/token present, no hard-negative cue",
        )
    if has_multi_word_hard_neg:
        return Triage(
            "hard_negative",
            high_precision_hits,
            single_token_hits,
            risky_hits,
            hard_negative_hits,
            subtype_tags,
            score,
            "multi-word hard-negative phrase present - confirmed non-ambience",
        )
    if has_only_risky:
        return Triage(
            "review",
            high_precision_hits,
            single_token_hits,
            risky_hits,
            hard_negative_hits,
            subtype_tags,
            score,
            "only ambiguous risky tokens present - needs human judgment",
        )
    return Triage("none", reason="no relevant signal")


def classify_ambience_candidate(text: str) -> dict[str, Any]:
    """Return the reusable weak-label decision fields for a single text."""
    triage = score_row(text)
    primary_subtype = _primary_subtype(text, triage.subtype_tags)
    weak_label = {
        "ambience": "ambience_candidate",
        "hard_negative": "hard_negative_candidate",
        "review": "review_needed",
        "none": "none",
    }[triage.bucket]
    cleaned = clean(text)
    if primary_subtype == "smell" and any(place in cleaned for place in ["المطعم", "المكان", "المحل"]):
        weak_label = "ambience_candidate"
    if primary_subtype == "decor_furniture" and triage.bucket == "ambience":
        weak_label = "ambience_candidate"
    suggested_category = AMBIENCE_CATEGORY if weak_label == "ambience_candidate" else ""
    return {
        "weak_label": weak_label,
        "contains_ambience_keyword": bool(triage.high_precision_hits or triage.single_token_hits),
        "ambience_subtype": primary_subtype,
        "risk_keyword": " | ".join(triage.risky_hits),
        "is_hard_negative": weak_label == "hard_negative_candidate",
        "suggested_category": suggested_category,
        "review_status": "unreviewed",
        "notes": triage.reason,
        "_bucket": triage.bucket,
        "_score": round(triage.score, 2),
        "_ambience_subtype_all": " | ".join(triage.subtype_tags),
        "_high_precision_hits": " | ".join(triage.high_precision_hits),
        "_single_token_hits": " | ".join(triage.single_token_hits),
        "_hard_negative_hits": " | ".join(triage.hard_negative_hits),
    }


def _primary_subtype(text: str, subtype_tags: list[str]) -> str:
    """Choose the most actionable subtype when generic place tokens also hit."""
    if not subtype_tags:
        return ""
    cleaned = clean(text)
    if any(token in cleaned for token in ["مكيف", "تكييف", "يبرد", "حار", "بارد"]):
        if "temperature_ac" in subtype_tags:
            return "temperature_ac"
    if any(token in cleaned for token in ["ريحه", "ريحة", "رائحه", "رائحة"]):
        if "smell" in subtype_tags:
            return "smell"
    if any(token in cleaned for token in ["تهتز", "مهزوزه", "مكسوره", "خربانه", "خربان"]):
        if "decor_furniture" in subtype_tags or any(surface in cleaned for surface in ["طاوله", "كرسي", "كنب"]):
            return "decor_furniture"
    for subtype in subtype_tags:
        if subtype != "general_place_vibe":
            return subtype
    return subtype_tags[0]


def detect_text_column(df: pd.DataFrame, explicit: str | None = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"text column {explicit!r} not found. Available: {list(df.columns)}")
        return explicit
    for col in TEXT_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"could not auto-detect text column. Tried: {TEXT_COLUMN_CANDIDATES}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported input extension {path.suffix!r}; use CSV, XLSX, JSON, or JSONL")


def arabic_mask(series: pd.Series) -> pd.Series:
    pattern = re.compile(ARABIC_RE)
    return series.fillna("").astype(str).map(lambda value: bool(pattern.search(value)))


def schema_record(
    *,
    row_id: Any,
    text: str,
    source: str,
    weak_label: str,
    category: str = "",
    source_detail: str = "",
    city: str = "",
    restaurant_name: str = "",
    dialect: str = "unknown",
    contains_ambience_keyword: bool = False,
    ambience_subtype: str = "",
    risk_keyword: str = "",
    is_hard_negative: bool = False,
    label_confidence: str = "weak",
    review_status: str = "unreviewed",
    notes: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = {col: "" for col in SCHEMA_COLUMNS}
    rec.update(
        {
            "id": row_id,
            "text": text,
            "category": category,
            "label": "",
            "source": source,
            "source_detail": source_detail,
            "city": city,
            "restaurant_name": restaurant_name,
            "dialect": dialect or "unknown",
            "weak_label": weak_label,
            "contains_ambience_keyword": bool(contains_ambience_keyword),
            "ambience_subtype": ambience_subtype,
            "risk_keyword": risk_keyword,
            "is_hard_negative": bool(is_hard_negative),
            "is_synthetic": False,
            "label_confidence": label_confidence,
            "review_status": review_status,
            "notes": notes,
            "split": "unassigned",
        }
    )
    if metadata:
        for key, value in metadata.items():
            if key not in rec:
                rec[key] = "" if pd.isna(value) else value
    return rec


def record_from_filter(
    *,
    row_id: Any,
    text: str,
    source: str,
    source_detail: str = "",
    label_confidence: str = "weak",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    decision = classify_ambience_candidate(text)
    if decision["_bucket"] == "none":
        return None
    return schema_record(
        row_id=row_id,
        text=text,
        source=source,
        source_detail=source_detail,
        weak_label=decision["weak_label"],
        category=decision["suggested_category"],
        contains_ambience_keyword=decision["contains_ambience_keyword"],
        ambience_subtype=decision["ambience_subtype"],
        risk_keyword=decision["risk_keyword"],
        is_hard_negative=decision["is_hard_negative"],
        label_confidence=label_confidence,
        review_status=decision["review_status"],
        notes=decision["notes"],
        metadata={
            **(metadata or {}),
            "score": decision["_score"],
            "ambience_subtype_all": decision["_ambience_subtype_all"],
            "high_precision_hits": decision["_high_precision_hits"],
            "single_token_hits": decision["_single_token_hits"],
            "hard_negative_hits": decision["_hard_negative_hits"],
        },
    )


def write_csv(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    leading = [c for c in SCHEMA_COLUMNS if c in df.columns]
    trailing = [c for c in df.columns if c not in leading]
    df = df[leading + trailing].fillna("")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return len(df)


def split_and_write_outputs(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    input_path: str,
    source_name: str,
    row_count_raw: int,
    row_count_arabic: int,
    script_name: str,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    ambience = [r for r in rows if r.get("weak_label") == "ambience_candidate"]
    hard_negative = [r for r in rows if r.get("weak_label") == "hard_negative_candidate"]
    review = [r for r in rows if r.get("weak_label") == "review_needed" or r.get("review_status") == "needs_discussion"]

    write_csv(ambience, output_dir / "ambience_candidates.csv")
    write_csv(hard_negative, output_dir / "hard_negative_candidates.csv")
    write_csv(review, output_dir / "review_needed.csv")

    subtype_counts = Counter(r.get("ambience_subtype", "") for r in ambience if r.get("ambience_subtype"))
    risky_counts: Counter[str] = Counter()
    for r in rows:
        for token in str(r.get("risk_keyword", "")).split(" | "):
            if token:
                risky_counts[token] += 1

    summary = {
        "input_path": input_path,
        "source_name": source_name,
        "row_count_raw": row_count_raw,
        "row_count_arabic": row_count_arabic,
        "ambience_candidates_count": len(ambience),
        "hard_negative_candidates_count": len(hard_negative),
        "review_needed_count": len(review),
        "top_ambience_subtypes": dict(subtype_counts.most_common(10)),
        "top_risky_keywords": dict(risky_counts.most_common(10)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script_name": script_name,
        "errors": errors or [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
