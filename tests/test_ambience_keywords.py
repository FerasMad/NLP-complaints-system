"""Structural tests for src/config/ambience_keywords.py.

These guard against silent regressions in the keyword vocabulary:
duplicates, empty subtypes, accidental cross-list contamination
(an ambience word leaking into the hard-negative set, etc.). They
do not require torch / transformers / data — fast CI gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make src/ importable for the test
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from config import ambience_keywords as kw  # noqa: E402


EXPECTED_SUBTYPES = {
    "temperature_ac", "seating_comfort", "space_crowding", "noise_music",
    "lighting", "smell", "decor_furniture", "parking", "bathroom_facilities",
    "outdoor_view", "privacy", "general_place_vibe",
}


def test_all_12_subtypes_defined():
    assert set(kw.AMBIENCE_BY_SUBTYPE.keys()) == EXPECTED_SUBTYPES, (
        f"subtype mismatch — expected {EXPECTED_SUBTYPES}, "
        f"got {set(kw.AMBIENCE_BY_SUBTYPE.keys())}"
    )


def test_every_subtype_has_keywords():
    """No subtype should be empty — that's a sign of an editing accident."""
    for subtype, words in kw.AMBIENCE_BY_SUBTYPE.items():
        assert len(words) >= 5, f"subtype {subtype!r} has only {len(words)} keywords"


def test_high_precision_phrases_are_multi_word():
    """The HIGH_PRECISION_PHRASES list should only contain phrases with spaces."""
    for phrase in kw.AMBIENCE_HIGH_PRECISION_PHRASES:
        assert " " in phrase, f"{phrase!r} in HIGH_PRECISION_PHRASES is single-word"


def test_single_tokens_have_no_spaces():
    for token in kw.AMBIENCE_SINGLE_TOKENS:
        assert " " not in token, f"{token!r} in SINGLE_TOKENS contains a space"


def test_risky_words_are_disjoint_from_hard_negatives():
    """Risky words and hard-negative words must NOT overlap.

    A risky word ("حار") could go either way (food temp vs room temp); it goes
    to review_needed alone. A hard-negative word ("الموظف") is unambiguous
    evidence of a non-ambience class. A word in both would create undefined
    triage behavior.
    """
    overlap = kw.RISKY_AMBIGUOUS & kw.ALL_HARD_NEGATIVES
    assert not overlap, f"these words appear in both RISKY_AMBIGUOUS and ALL_HARD_NEGATIVES: {overlap}"


def test_required_risky_words_present():
    """The known boundary-breaking words must be in RISKY_AMBIGUOUS.

    These are the v3 ambience class's death traps — their absence here would
    let 'الأكل بارد' / 'الشاورما حارة' false-anchor as ambience.
    """
    required = {"حار", "بارد", "قديم", "ريحة", "زيت", "برا", "المطعم", "الفرع"}
    missing = required - kw.RISKY_AMBIGUOUS
    assert not missing, f"required risky words missing from RISKY_AMBIGUOUS: {missing}"


def test_subtype_for_phrase_round_trips():
    """Every phrase in AMBIENCE_BY_SUBTYPE must resolve back to its subtype."""
    for subtype, phrases in kw.AMBIENCE_BY_SUBTYPE.items():
        for phrase in phrases:
            got = kw.subtype_for_phrase(phrase)
            assert got == subtype, (
                f"subtype_for_phrase({phrase!r}) returned {got!r}, expected {subtype!r}"
            )


def test_no_duplicate_phrases_across_subtypes():
    """A phrase should appear in at most one subtype.

    Duplicates would make subtype tagging non-deterministic (which subtype
    does subtype_for_phrase return?). If a word genuinely fits two subtypes,
    pick one and document the choice with a NB comment in keywords.py.
    """
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for subtype, phrases in kw.AMBIENCE_BY_SUBTYPE.items():
        for p in phrases:
            if p in seen and seen[p] != subtype:
                duplicates.append(f"{p!r} in both {seen[p]!r} and {subtype!r}")
            seen[p] = subtype
    assert not duplicates, f"duplicate phrases across subtypes: {duplicates}"
