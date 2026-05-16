"""Pure-logic tests for hf_space/app.py — no model load required.

Covers the functions that ship to the Space and that the prior test suite
didn't touch: clean(), is_arabic_enough(), looks_like_praise(), apply_rescue(),
extract_aspects(), and the rail-composition path in render_result().

These tests run in milliseconds. They are the safety net the Adversarial Q&A
audit (I-1) flagged as missing. Together with the model-dependent suite they
give the Space's logic real coverage.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow `import hf_space.app` from the repo root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Block the model load — these tests touch only deterministic helpers.
# Setting a dummy repo ID still forces transformers to hit the hub, so we
# monkey-patch the heavy loads to no-ops before the module imports.
os.environ.setdefault("HF_REPO_ID", "FerasMad/arabic-complaints-classifier")

# Stub out `gradio` so this test suite runs without the gradio install
# (the Space depends on it at runtime, but the helpers we test do not).
# We use a real ModuleType so hypothesis / pytest's path scrutiny doesn't
# try to introspect a non-module object and fail.
import types as _types


class _GrStub:
    """An inert stand-in for any gradio attribute — callable, indexable,
    context-managed, iterable. All operations return another _GrStub so
    chains like `gr.Blocks(theme=gr.themes.Soft()).launch()` succeed silently."""

    def __getattr__(self, name):
        return _GrStub()

    def __call__(self, *a, **kw):
        return _GrStub()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter([])

    def __getitem__(self, _key):
        return _GrStub()

    def __bool__(self):
        return False


def _build_gradio_stub_module() -> _types.ModuleType:
    mod = _types.ModuleType("gradio")
    mod.__file__ = __file__  # satisfies hypothesis path scrutiny
    # Dynamic attribute access: any gr.X resolves to a _GrStub
    mod.__getattr__ = lambda name: _GrStub()  # type: ignore[attr-defined]
    return mod


if "gradio" not in sys.modules:
    sys.modules["gradio"] = _build_gradio_stub_module()
    themes_mod = _types.ModuleType("gradio.themes")
    themes_mod.__file__ = __file__
    themes_mod.__getattr__ = lambda name: _GrStub()  # type: ignore[attr-defined]
    sys.modules["gradio.themes"] = themes_mod


@pytest.fixture(scope="module")
def app_module():
    """Import hf_space.app with model loads stubbed out."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    orig_tok = AutoTokenizer.from_pretrained
    orig_model = AutoModelForSequenceClassification.from_pretrained

    class _FakeTokenizer:
        def __call__(self, *a, **kw):
            return {"input_ids": torch.zeros(1, 4, dtype=torch.long),
                    "attention_mask": torch.ones(1, 4, dtype=torch.long)}

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

    class _FakeOutput:
        def __init__(self):
            self.logits = torch.zeros(1, 8)

    class _FakeModel:
        config = type("C", (), {"num_labels": 8})()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def __call__(self, *a, **kw):
            return _FakeOutput()

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

    AutoTokenizer.from_pretrained = _FakeTokenizer.from_pretrained
    AutoModelForSequenceClassification.from_pretrained = _FakeModel.from_pretrained
    try:
        if "hf_space.app" in sys.modules:
            del sys.modules["hf_space.app"]
        mod = importlib.import_module("hf_space.app")
    finally:
        AutoTokenizer.from_pretrained = orig_tok
        AutoModelForSequenceClassification.from_pretrained = orig_model
    return mod


# ---------------------------------------------------------------------------
# clean() — normalization is idempotent, strips diacritics, folds alif forms
# ---------------------------------------------------------------------------

class TestClean:
    def test_idempotent(self, app_module):
        s = "الأكل بايخ جداً"
        assert app_module.clean(app_module.clean(s)) == app_module.clean(s)

    def test_strips_tashkeel(self, app_module):
        assert "ً" not in app_module.clean("جداً")
        assert "ٌ" not in app_module.clean("شيءٌ")

    def test_folds_alif_forms(self, app_module):
        # أ، إ، آ all collapse to ا so dialect typos match training tokens
        assert app_module.clean("الأكل") == app_module.clean("الاكل")
        assert app_module.clean("إنتظار") == app_module.clean("انتظار")

    def test_folds_taa_marbuta(self, app_module):
        # ة → ه matches MARBERT / CAMeLBERT pretrain normalization
        assert app_module.clean("نظيفة") == app_module.clean("نظيفه")

    def test_empty_safe(self, app_module):
        assert app_module.clean("") == ""
        assert app_module.clean(None) == ""

    def test_emoji_stripped(self, app_module):
        out = app_module.clean("الاكل بايخ ❤️🔥")
        assert "❤" not in out and "🔥" not in out
        assert "الاكل بايخ" in out


# ---------------------------------------------------------------------------
# is_arabic_enough() — gates non-Arabic input out of the model
# ---------------------------------------------------------------------------

class TestIsArabicEnough:
    def test_pure_arabic_passes(self, app_module):
        assert app_module.is_arabic_enough("الاكل بايخ والمندوب تاخر")

    def test_pure_english_fails(self, app_module):
        assert not app_module.is_arabic_enough("the food was terrible and the driver was late")

    def test_too_short_fails(self, app_module):
        assert not app_module.is_arabic_enough("ال")
        assert not app_module.is_arabic_enough("")

    def test_mixed_above_threshold_passes(self, app_module):
        # 30% Arabic minimum — "الاكل بايخ test" is mostly Arabic
        assert app_module.is_arabic_enough("الاكل بايخ test 123")


# ---------------------------------------------------------------------------
# looks_like_praise() — the praise-screen guard
# ---------------------------------------------------------------------------

class TestLooksLikePraise:
    def test_pure_praise_detected(self, app_module):
        assert app_module.looks_like_praise(app_module.clean("ممتاز جدا شكرا"))

    def test_pure_complaint_not_praise(self, app_module):
        assert not app_module.looks_like_praise(app_module.clean("الاكل بايخ"))

    def test_word_boundary_safe(self, app_module):
        # "الاكل" contains "لا" — must not false-match the negation particle
        assert not app_module.looks_like_praise(app_module.clean("الاكل لذيذ بس بارد"))

    def test_formal_msa_negation_recognized(self, app_module):
        # I-2: ليس must be in NEGATIVE_WORDS so formal complaints reach the model
        assert not app_module.looks_like_praise(app_module.clean("ليس ممتاز ابدا"))

    def test_contrastive_marker_bypasses_screen(self, app_module):
        # I-3: contrastive marker بس means this is a legitimate complaint
        # framed politely, NOT a praise post that should be discarded.
        assert not app_module.looks_like_praise(app_module.clean("ممتاز بس الجو حار"))
        assert not app_module.looks_like_praise(app_module.clean("الموظف رائع لكن الكاشير غلط"))


# ---------------------------------------------------------------------------
# apply_rescue() — the keyword override layer; returns (probs, rescued_idx|None)
# ---------------------------------------------------------------------------

class TestApplyRescue:
    def test_returns_tuple(self, app_module):
        probs = np.array([0.1] * 7 + [0.3])
        result = app_module.apply_rescue(probs, "كلام عادي")
        assert isinstance(result, tuple) and len(result) == 2

    def test_no_match_returns_none_idx(self, app_module):
        probs = np.array([0.1] * 7 + [0.3])
        _new, idx = app_module.apply_rescue(probs, "كلام عادي")
        assert idx is None

    def test_hygiene_phrase_rescues(self, app_module):
        # "الحمام قذر" is a canonical hygiene rescue
        probs = np.array([0.2, 0.1, 0.05, 0.4, 0.1, 0.05, 0.05, 0.05])
        new_probs, idx = app_module.apply_rescue(probs, app_module.clean("الحمام قذر جدا"))
        assert idx is not None
        # The rescued category should be the new argmax
        assert int(new_probs.argmax()) == idx

    def test_probs_renormalize(self, app_module):
        # After rescue + dampening, probabilities still sum to ~1
        probs = np.array([0.2, 0.1, 0.05, 0.4, 0.1, 0.05, 0.05, 0.05])
        new_probs, _ = app_module.apply_rescue(probs, app_module.clean("الحمام قذر جدا"))
        assert abs(new_probs.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# extract_aspects() — single source of truth for the rail
# ---------------------------------------------------------------------------

class TestExtractAspects:
    def test_returns_tuple_of_correct_types(self, app_module):
        matches, by_aspect = app_module.extract_aspects(app_module.clean("الاكل بايخ"))
        assert isinstance(matches, list)
        assert isinstance(by_aspect, dict)

    def test_food_aspect_detected(self, app_module):
        _matches, by_aspect = app_module.extract_aspects(app_module.clean("الاكل بايخ ومالح"))
        assert "جودة الطعام" in by_aspect

    def test_multi_aspect_detected(self, app_module):
        # The exact bug the user reported pre-6f91ad5
        _matches, by_aspect = app_module.extract_aspects(
            app_module.clean("الاكل جودته سيئه والتوصيل اخذ وقت طويل")
        )
        # Expect at least 2 of: food, delivery, wait
        keys = set(by_aspect.keys())
        assert len(keys) >= 2

    def test_empty_input_safe(self, app_module):
        matches, by_aspect = app_module.extract_aspects("")
        assert matches == [] and by_aspect == {}


# ---------------------------------------------------------------------------
# render_result() — aspect-driven rail; the 6f91ad5 ship target
# ---------------------------------------------------------------------------

class TestRenderResult:
    def test_zero_aspects_falls_back_to_top1(self, app_module):
        # No aspects detected → single badge for model top-1
        top = [("جودة الطعام", 0.9), ("عامة", 0.05), ("التوصيل", 0.03)]
        html = app_module.render_result(top, by_aspect={})
        assert "result-cat" in html
        # Multi-aspect badge should NOT appear with one category
        assert "تشمل الشكوى أكثر من جانب" not in html

    def test_two_aspects_render_two_badges(self, app_module):
        top = [("جودة الطعام", 0.6), ("التوصيل", 0.3), ("عامة", 0.05)]
        by_aspect = {"جودة الطعام": ["بايخ"], "التوصيل": ["تاخر"]}
        full_probs = {"جودة الطعام": 0.6, "التوصيل": 0.3, "عامة": 0.05,
                      "السعر والقيمة": 0.01, "النظافة": 0.01, "خدمة الموظفين": 0.01,
                      "دقة الطلب": 0.01, "وقت الانتظار": 0.01}
        html = app_module.render_result(top, by_aspect, full_probs=full_probs)
        # The new rail shows result-rank-top (darker, first) + result-rank-other
        assert "result-rank-top" in html
        assert "result-rank-other" in html
        # Multi-aspect badge fires at ≥2
        assert "تشمل الشكوى أكثر من جانب" in html

    def test_rail_capped_at_four(self, app_module):
        # 5 aspects detected → must cap at 4
        top = [("جودة الطعام", 0.4), ("التوصيل", 0.2), ("النظافة", 0.15),
               ("وقت الانتظار", 0.1), ("السعر والقيمة", 0.1)]
        by_aspect = {c: ["x"] for c, _ in top}
        full_probs = {c: p for c, p in top}
        for c in ["خدمة الموظفين", "دقة الطلب", "عامة"]:
            full_probs[c] = 0.01
        html = app_module.render_result(top, by_aspect, full_probs=full_probs)
        # Count rendered rows (each row has "result-row" class)
        n_rows = html.count("result-row")
        assert n_rows == 4, f"expected 4 badges, got {n_rows}"

    def test_rescued_category_force_included(self, app_module):
        # extract_aspects missed delivery but rescue forced it → must appear
        top = [("جودة الطعام", 0.5), ("التوصيل", 0.3), ("عامة", 0.05)]
        by_aspect = {"جودة الطعام": ["بايخ"]}  # delivery NOT extracted
        full_probs = {c: 0.05 for c in ["التوصيل", "السعر والقيمة", "النظافة",
                                         "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار"]}
        full_probs["جودة الطعام"] = 0.5
        html = app_module.render_result(top, by_aspect, full_probs=full_probs,
                                         rescued_cat="التوصيل")
        # Delivery should appear because rescue forced it
        assert "التوصيل" in html


# ---------------------------------------------------------------------------
# Determinism — same input → same output across calls
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_clean_deterministic(self, app_module):
        s = "الأكل بايخ والمندوب تاخر"
        assert app_module.clean(s) == app_module.clean(s)

    def test_extract_aspects_deterministic(self, app_module):
        s = app_module.clean("الاكل بايخ ومالح والمندوب تاخر")
        out1 = app_module.extract_aspects(s)
        out2 = app_module.extract_aspects(s)
        assert out1[1] == out2[1]  # by_aspect dict comparable
