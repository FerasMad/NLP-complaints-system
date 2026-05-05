from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.filter_ambience_core import classify_ambience_candidate  # noqa: E402


BOUNDARY_CASES = [
    ("الأكل بارد", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("المكان بارد والمكيف قوي", {"ambience_candidate"}, "temperature_ac"),
    ("الشاورما حارة", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("المطعم حار والمكيف خربان", {"ambience_candidate"}, "temperature_ac"),
    ("ريحة الزيت ماسكة في المطعم", {"ambience_candidate", "review_needed"}, "smell"),
    ("البطاطس مليانة زيت", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("الموظف كان بارد في التعامل", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("الطاولة وصخة", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("الطاولة تهتز", {"ambience_candidate"}, "decor_furniture"),
    ("الحمام وصخ", {"review_needed", "hard_negative_candidate", "none"}, ""),
    ("الحمام خربان وما فيه صابون", {"ambience_candidate"}, "bathroom_facilities"),
]


@pytest.mark.parametrize("text,expected_weak_labels,expected_subtype", BOUNDARY_CASES)
def test_classify_ambience_candidate_boundaries(text, expected_weak_labels, expected_subtype):
    decision = classify_ambience_candidate(text)
    assert decision["weak_label"] in expected_weak_labels
    if expected_subtype:
        assert decision["ambience_subtype"] == expected_subtype
        assert decision["suggested_category"] == "الجو والمكان"
        assert decision["contains_ambience_keyword"] is True
    else:
        assert decision["weak_label"] != "ambience_candidate"
