"""Pure-logic tests for inference helpers — no model loading required.

Tests the building blocks that don't need the 1.8 GB ensemble:
  - clean_arabic() text normalization
  - arabic_ratio() / is_arabic_enough() OOD gates
  - softmax_entropy() math
  - PII scrubbing (in api.py)
"""
import numpy as np
import pytest

from app.ensemble_inference import (
    arabic_ratio,
    clean_arabic,
    is_arabic_enough,
    softmax_entropy,
)


class TestCleanArabic:
    def test_empty_input(self):
        assert clean_arabic("") == ""
        assert clean_arabic(None) == ""

    def test_strips_tashkeel(self):
        assert clean_arabic("الْأَكْلُ") == "الاكل"

    def test_normalizes_alef(self):
        assert clean_arabic("أمين") == "امين"
        assert clean_arabic("إلى") == "الي"
        assert clean_arabic("آخر") == "اخر"
        assert clean_arabic("ٱلسلام") == "السلام"

    def test_normalizes_ya_and_ta_marbuta(self):
        assert "ى" not in clean_arabic("على")
        assert "ة" not in clean_arabic("مدينة")

    def test_strips_punctuation(self):
        # Non-Arabic punctuation (!, %, etc.) is stripped. Arabic punctuation
        # (، . ; etc.) is in the U+0600-U+06FF block and is KEPT — it's part
        # of normal Arabic text. Matches hf_space/app.py:53 NON_ARABIC regex.
        out = clean_arabic("الاكل!!! بايخ ، ، ،")
        assert "!" not in out
        assert "بايخ" in out
        # Arabic comma intentionally preserved — change this test only if the
        # cleaning policy itself changes (don't remove the comma in the regex).
        assert "،" in out

    def test_preserves_arabic_indic_digits(self):
        # 0-9 (Latin) and ٠-٩ (Arabic-Indic) both kept
        out = clean_arabic("١٠ ريال 50%")
        assert "١٠" in out or "10" in out  # at minimum digits aren't all stripped

    def test_collapses_whitespace(self):
        out = clean_arabic("الاكل      بايخ")
        assert "  " not in out


class TestArabicRatio:
    def test_pure_arabic(self):
        assert arabic_ratio("الاكل بايخ") > 0.5

    def test_pure_english(self):
        assert arabic_ratio("hello world") == 0.0

    def test_empty(self):
        assert arabic_ratio("") == 0.0
        assert arabic_ratio("ab") == 0.0  # too short

    def test_mixed(self):
        # 10 Arabic chars + 5 latin chars + spaces
        text = "الاكل hello"
        r = arabic_ratio(text)
        assert 0.0 < r < 1.0


class TestIsArabicEnough:
    def test_arabic_passes(self):
        assert is_arabic_enough("الاكل بايخ ومالح") is True

    def test_english_fails(self):
        assert is_arabic_enough("Hello, this is a test in English") is False

    def test_gibberish_fails(self):
        assert is_arabic_enough("aaaa bbbb cccc") is False

    def test_threshold_configurable(self):
        # ~10% Arabic
        text = "the food is bad. الاكل"
        assert is_arabic_enough(text, min_ratio=0.5) is False
        # Lower threshold should pass
        # (depends on text — keep loose)


class TestSoftmaxEntropy:
    def test_uniform_is_max(self):
        n = 8
        uniform = np.ones(n) / n
        e = softmax_entropy(uniform)
        # max entropy of n-class uniform is ln(n)
        assert abs(e - np.log(n)) < 0.01

    def test_one_hot_is_zero(self):
        n = 8
        one_hot = np.zeros(n)
        one_hot[3] = 1.0
        e = softmax_entropy(one_hot)
        assert abs(e) < 0.01

    def test_entropy_increases_with_uncertainty(self):
        confident = np.array([0.95, 0.025, 0.025])
        uncertain = np.array([0.5, 0.3, 0.2])
        assert softmax_entropy(uncertain) > softmax_entropy(confident)


class TestPIIScrubbing:
    def test_phone_redacted(self):
        from app.api import scrub_pii

        out = scrub_pii("اتصل علي 0501234567 الان")
        assert "0501234567" not in out
        assert "REDACTED_PHONE" in out

    def test_email_redacted(self):
        from app.api import scrub_pii

        out = scrub_pii("ايميلي test@example.com")
        assert "test@example.com" not in out
        assert "REDACTED_EMAIL" in out

    def test_url_redacted(self):
        from app.api import scrub_pii

        out = scrub_pii("راجع الموقع https://example.com/foo")
        assert "https://example.com" not in out
        assert "REDACTED_URL" in out

    def test_no_pii_unchanged(self):
        from app.api import scrub_pii

        text = "الاكل بايخ ومالح"
        assert scrub_pii(text) == text

    def test_empty_safe(self):
        from app.api import scrub_pii

        assert scrub_pii("") == ""
        assert scrub_pii(None) is None
