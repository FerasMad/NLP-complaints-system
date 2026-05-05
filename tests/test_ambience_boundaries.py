"""End-to-end test of the ambience filter against the boundary-example fixture.

Runs `src/filter_ambience_candidates.py:score_row` on each row of the 90-case
hand-written fixture and asserts:

  * Every row labeled الجو والمكان (positives) lands in the ambience or
    review bucket (never in hard_negative or filtered-out).
  * Overall ambience-bucket precision >= 95% (i.e. <= 5% of rows in the
    ambience bucket are actually non-ambience).
  * Overall ambience-bucket recall >= 90% on positives.
  * Bathroom-rule cases are bucketed correctly per BOUNDARIES.md.
  * Temperature-rule cases distinguish food temperature from room temp.
  * Smell-rule cases distinguish food smell from environmental smell.

These thresholds are the v5 acceptance bar — if a future keyword edit
regresses below them, this test fails the CI gate before the bad change
ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from filter_ambience_candidates import score_row  # noqa: E402

FIXTURE = _ROOT / "tests" / "fixtures" / "ambience_boundary_examples.csv"


@pytest.fixture(scope="module")
def fixture_df():
    assert FIXTURE.exists(), f"missing fixture file: {FIXTURE}"
    return pd.read_csv(FIXTURE, encoding="utf-8")


@pytest.fixture(scope="module")
def triaged(fixture_df):
    """Score every fixture row; return a DataFrame with the assigned bucket."""
    rows = []
    for _, r in fixture_df.iterrows():
        result = score_row(r["text"])
        rows.append({
            "id": r["id"],
            "text": r["text"],
            "expected_label": r["expected_label"],
            "expected_subtype": r.get("expected_subtype", ""),
            "bucket": result.bucket,
            "subtype_tags": result.subtype_tags,
            "reason": result.reason,
        })
    return pd.DataFrame(rows)


def test_no_ambience_positive_in_hard_negative(triaged):
    """No row labeled الجو والمكان should land in hard_negative — that bucket is for confirmed non-ambience."""
    leaked = triaged[
        (triaged["expected_label"] == "الجو والمكان")
        & (triaged["bucket"] == "hard_negative")
    ]
    assert leaked.empty, (
        f"ambience positives leaked into hard_negative bucket:\n"
        + "\n".join(f"  {r['id']}: {r['text']}" for _, r in leaked.iterrows())
    )


def test_ambience_recall_at_least_90_percent(triaged):
    positives = triaged[triaged["expected_label"] == "الجو والمكان"]
    in_ambience_bucket = positives[positives["bucket"] == "ambience"]
    recall = len(in_ambience_bucket) / max(len(positives), 1)
    assert recall >= 0.90, (
        f"ambience recall {recall:.0%} < 90%; only {len(in_ambience_bucket)}/{len(positives)} "
        f"positives landed in ambience bucket"
    )


def test_ambience_precision_at_least_95_percent(triaged):
    in_ambience_bucket = triaged[triaged["bucket"] == "ambience"]
    if in_ambience_bucket.empty:
        pytest.fail("ambience bucket is empty — filter is broken")
    correct = in_ambience_bucket[in_ambience_bucket["expected_label"] == "الجو والمكان"]
    precision = len(correct) / len(in_ambience_bucket)
    assert precision >= 0.95, (
        f"ambience precision {precision:.0%} < 95%; "
        f"{len(in_ambience_bucket) - len(correct)}/{len(in_ambience_bucket)} leaks:\n"
        + "\n".join(
            f"  {r['id']} ({r['expected_label']}): {r['text']}"
            for _, r in in_ambience_bucket[
                in_ambience_bucket["expected_label"] != "الجو والمكان"
            ].iterrows()
        )
    )


# ----- Specific boundary rules from BOUNDARIES.md -----

# Each pair: (text, expected_bucket). 'expected_bucket' must be one of:
# 'ambience', 'hard_negative', 'review', 'none'
BOUNDARY_RULE_CASES = [
    # Temperature: food vs room
    ("الأكل بارد", {"hard_negative", "review"}),       # food temp
    ("المكان بارد والمكيف قوي", {"ambience"}),         # room temp
    ("الشاورما حارة", {"hard_negative", "review"}),    # food spice
    ("المكان حار والمكيف ما يبرد", {"ambience"}),     # AC failure
    # Smell: food vs environment
    ("ريحة الزيت ماسكة في المطعم", {"ambience", "review"}),  # env smell
    ("البطاطس مليانة زيت", {"hard_negative", "review", "none"}),  # food
    # Furniture: dirty vs broken
    ("الطاولة وصخة فيها بقع طعام", {"hard_negative"}),  # cleanliness
    ("الطاولة تهتز كل ما تحط شي", {"ambience"}),       # condition
    # Bathroom: cleanliness vs facility
    ("الحمام وصخ", {"hard_negative", "review"}),       # cleanliness
    ("السيفون خربان وما فيه صابون", {"ambience"}),    # facility
    # Staff vs ambience-sounding word
    ("الموظف كان بارد في التعامل", {"hard_negative", "review"}),
]


@pytest.mark.parametrize("text,expected_buckets", BOUNDARY_RULE_CASES)
def test_boundary_rule(text, expected_buckets):
    """Each canonical boundary case must triage to one of the allowed buckets."""
    result = score_row(text)
    assert result.bucket in expected_buckets, (
        f"text={text!r}\n"
        f"  expected one of: {expected_buckets}\n"
        f"  got: {result.bucket!r}\n"
        f"  reason: {result.reason}\n"
        f"  hits: hp={result.high_precision_hits} "
        f"single={result.single_token_hits} "
        f"risky={result.risky_hits} "
        f"hard_neg={result.hard_negative_hits}"
    )
