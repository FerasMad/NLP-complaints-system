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
    cat, conf, top3 = classifier.predict(text)
    assert cat == expected, f"text={text!r} expected={expected} got={cat} (conf={conf:.2%}, top3={top3})"


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_returns_8_classes_in_top3(classifier):
    """Top-3 should always have 3 entries with valid category names."""
    cat, conf, top3 = classifier.predict("الاكل بايخ")
    assert len(top3) == 3
    valid_cats = {
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    }
    for c, p in top3:
        assert c in valid_cats, f"unexpected category {c!r}"
        assert 0.0 <= p <= 1.0, f"invalid probability {p}"


@pytest.mark.gpu
@pytest.mark.integration
def test_predict_too_short(classifier):
    """Very short input returns empty / category zero confidence."""
    cat, conf, top3 = classifier.predict("ا")
    # Behavior: ensemble_inference returns zero-prob array → argmax is index 0
    # The predict() function doesn't error; just returns whatever has the highest tied prob
    # We don't assert a specific category, just that it doesn't crash
    assert isinstance(cat, str)
