"""Determinism tests — same input must give same output across calls.

Catches GPU non-determinism, dropout left enabled, race conditions, etc.
Marked `gpu` because real predictions need the ensemble.
"""
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def classifier():
    from app.ensemble_inference import EnsembleClassifier

    cfg = Path(__file__).resolve().parent.parent / "models" / "ensemble_final" / "config.json"
    return EnsembleClassifier(cfg)


CANONICAL_TEXTS = [
    "الاكل بايخ ومالح",
    "الاسعار مبالغ فيها",
    "وصل الطلب بارد جدا والمندوب تاخر",
    "النظافه سيئه الطاولات متسخه",
]


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("text", CANONICAL_TEXTS)
def test_predict_deterministic(classifier, text):
    """Calling predict() N times gives identical category + confidence."""
    r1 = classifier.predict(text)
    r2 = classifier.predict(text)
    r3 = classifier.predict(text)

    assert r1.category == r2.category == r3.category
    # Confidence should be exactly equal (no random sampling at inference)
    assert abs(r1.confidence - r2.confidence) < 1e-6
    assert abs(r2.confidence - r3.confidence) < 1e-6


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_probs_deterministic(classifier):
    """predict_probs() is deterministic across calls."""
    text = "الاكل بايخ"
    p1 = classifier.predict_probs(text)
    p2 = classifier.predict_probs(text)
    import numpy as np
    assert np.allclose(p1, p2, atol=1e-6)
