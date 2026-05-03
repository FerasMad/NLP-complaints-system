"""Property-based tests using Hypothesis.

Tests that should hold for ANY valid input, not just hand-picked canaries.
Marked `gpu` because the underlying classifier needs the ensemble.

Run: pytest tests/test_properties.py -v -m "not gpu" (skips real model)
     pytest tests/test_properties.py -v                (full)
"""
import sys

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from app.ensemble_inference import arabic_ratio, clean_arabic


# Strategy for Arabic-ish text (mix of letters, spaces, punctuation)
arabic_chars = "ابتثجحخدذرزسشصضطظعغفقكلمنهويءأإآى ةى"
arabic_text = st.text(alphabet=arabic_chars, min_size=3, max_size=200)


@given(arabic_text)
def test_clean_arabic_idempotent(text):
    """clean(clean(x)) == clean(x) — applying twice gives same result."""
    once = clean_arabic(text)
    twice = clean_arabic(once)
    assert once == twice


@given(arabic_text)
def test_clean_arabic_no_tashkeel_in_output(text):
    """No tashkeel characters survive cleaning."""
    out = clean_arabic(text)
    tashkeel_chars = "ًٌٍَُِّْ"
    for ch in tashkeel_chars:
        assert ch not in out


@given(arabic_text)
def test_clean_arabic_no_alef_variants_in_output(text):
    """Alef variants are normalized to plain alef."""
    out = clean_arabic(text)
    for ch in "أإآٱ":
        assert ch not in out


@given(arabic_text)
def test_clean_arabic_no_double_spaces(text):
    """Whitespace is collapsed."""
    out = clean_arabic(text)
    assert "  " not in out


@given(arabic_text)
def test_arabic_ratio_in_range(text):
    """arabic_ratio is always in [0, 1]."""
    r = arabic_ratio(text)
    assert 0.0 <= r <= 1.0


@given(st.text(alphabet="abcdefghijklmnop ", min_size=5, max_size=100))
def test_arabic_ratio_zero_for_english(text):
    """English-only text has zero Arabic ratio."""
    assert arabic_ratio(text) == 0.0


# --- Predict-output property tests (need real ensemble) ---

@pytest.fixture(scope="module")
def classifier():
    """Load ensemble once. Slow — ~2 min on CPU."""
    from pathlib import Path

    from app.ensemble_inference import EnsembleClassifier

    cfg = Path(__file__).resolve().parent.parent / "models" / "ensemble_final" / "config.json"
    return EnsembleClassifier(cfg)


@pytest.mark.gpu
@pytest.mark.integration
@given(arabic_text)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_predict_returns_valid_category(classifier, text):
    """For any Arabic-ish input, predict() returns a valid category or abstains."""
    valid_cats = {
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    }
    result = classifier.predict(text)
    assert (result.category is None) or (result.category in valid_cats)


@pytest.mark.gpu
@pytest.mark.integration
@given(arabic_text)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_predict_confidence_in_range(classifier, text):
    """Confidence is always a probability in [0, 1]."""
    result = classifier.predict(text)
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.gpu
@pytest.mark.integration
@given(arabic_text)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_predict_top_k_unique_categories(classifier, text):
    """Top-3 has unique categories (no duplicates)."""
    result = classifier.predict(text, top_k=3)
    if result.top_3:
        cats = [c for c, _ in result.top_3]
        assert len(set(cats)) == len(cats)


@pytest.mark.gpu
@pytest.mark.integration
@given(arabic_text)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_predict_top_k_sorted(classifier, text):
    """Top-3 is sorted descending by confidence."""
    result = classifier.predict(text, top_k=3)
    if len(result.top_3) >= 2:
        confs = [s for _, s in result.top_3]
        for i in range(len(confs) - 1):
            assert confs[i] >= confs[i + 1]
