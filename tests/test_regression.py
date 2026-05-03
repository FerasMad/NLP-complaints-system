"""Regression-gate test — block PRs that drop accuracy.

Loads the production ensemble, evaluates on a fixed 200-row test slice,
and asserts accuracy stays above a hard floor.

Marked `gpu` and `slow` — runs in CI on a special job, not on every push.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Hard floor: if accuracy on the regression slice drops below this, fail CI.
# Set well below current ~95% with margin for the smaller stratified slice.
ACCURACY_FLOOR = 0.92
MACRO_F1_FLOOR = 0.85
MIN_CLASS_F1_FLOOR = 0.65


@pytest.fixture(scope="module")
def classifier():
    from app.ensemble_inference import EnsembleClassifier

    cfg = Path(__file__).resolve().parent.parent / "models" / "ensemble_final" / "config.json"
    return EnsembleClassifier(cfg)


@pytest.fixture(scope="module")
def regression_slice():
    """200-row stratified subset of test.csv."""
    test_csv = Path(__file__).resolve().parent.parent / "data" / "processed" / "test.csv"
    if not test_csv.exists():
        pytest.skip("test.csv not present (CI without data); skipping regression test")
    df = pd.read_csv(test_csv)
    # Stratified sample with fixed seed
    sample = df.groupby("category", group_keys=False).apply(
        lambda x: x.sample(min(len(x), 25), random_state=42)
    )
    return sample.reset_index(drop=True).head(200)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
def test_accuracy_above_floor(classifier, regression_slice):
    """Top-1 accuracy on the regression slice must be >= 92%."""
    correct = 0
    total = 0
    for _, row in regression_slice.iterrows():
        result = classifier.predict(row["text"])
        if result.category is None:
            continue  # abstains don't count for/against accuracy
        total += 1
        if result.category == row["category"]:
            correct += 1

    accuracy = correct / max(total, 1)
    print(f"\n  Regression accuracy: {accuracy:.4f} ({correct}/{total})")
    assert accuracy >= ACCURACY_FLOOR, (
        f"Accuracy {accuracy:.4f} dropped below floor {ACCURACY_FLOOR}. "
        f"Investigate model regression."
    )


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.integration
def test_macro_f1_above_floor(classifier, regression_slice):
    """Macro F1 on regression slice must be >= 85%."""
    from sklearn.metrics import f1_score

    preds: list[str] = []
    golds: list[str] = []
    for _, row in regression_slice.iterrows():
        result = classifier.predict(row["text"])
        if result.category is None:
            continue
        preds.append(result.category)
        golds.append(row["category"])

    mF1 = f1_score(golds, preds, average="macro", zero_division=0)
    per = f1_score(golds, preds, average=None, labels=sorted(set(golds)), zero_division=0)
    min_class = float(per.min()) if len(per) else 0.0

    print(f"\n  Regression macro F1: {mF1:.4f}")
    print(f"  Min class F1:        {min_class:.4f}")
    assert mF1 >= MACRO_F1_FLOOR, f"Macro F1 {mF1:.4f} below floor {MACRO_F1_FLOOR}"
    assert min_class >= MIN_CLASS_F1_FLOOR, f"Min class F1 {min_class:.4f} below floor {MIN_CLASS_F1_FLOOR}"
