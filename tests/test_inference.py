"""
Inference tests — load the ensemble, predict on canary examples.

Marked `gpu` because loading 4 BERT models on CPU is slow (~2 min).
Run with: pytest tests/test_inference.py
Skip GPU tests: pytest -m "not gpu"
"""
import sys

import pytest


# Each tuple: (input_text, expected_top1_category)
CANARIES = [
    ("الاكل بايخ ومالح والطبخ مو متقن", "جودة الطعام"),
    ("الاسعار مبالغ فيها لا تناسب الجوده", "السعر والقيمة"),
    ("النظافه سيئه الطاولات متسخه والارض غير نظيفه", "النظافة"),
    ("طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي", "دقة الطلب"),
    ("الموظف اسلوبه سيء وغير محترم", "خدمة الموظفين"),
    ("انتظرت ساعتين قبل ان ياتي الطلب", "وقت الانتظار"),
    ("تجربه سيئه عموما لن اعود", "عامة"),
]


@pytest.fixture(scope="module")
def classifier(ensemble_config_path):
    """Load ensemble once per test module (slow on CPU)."""
    sys.path.insert(0, str(ensemble_config_path.parent.parent.parent))
    from app.ensemble_inference import EnsembleClassifier
    return EnsembleClassifier(ensemble_config_path)


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.parametrize("text,expected", CANARIES)
def test_canary_predictions(classifier, text, expected):
    """Each canary input must classify to its expected category."""
    result = classifier.predict(text)
    assert result.category == expected, (
        f"text={text!r} expected={expected} got={result.category} "
        f"(conf={result.confidence:.2%}, top_3={result.top_3})"
    )


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_returns_top3_valid_cats(classifier):
    """Top-3 should always have 3 entries with valid category names."""
    result = classifier.predict("الاكل بايخ")
    assert len(result.top_3) == 3
    valid_cats = {
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    }
    for c, p in result.top_3:
        assert c in valid_cats, f"unexpected category {c!r}"
        assert 0.0 <= p <= 1.0, f"invalid probability {p}"


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_too_short(classifier):
    """Very short input abstains via too_short reason; no crash."""
    result = classifier.predict("ا")
    assert result.abstain_reason == "too_short"
    assert result.category is None
