"""Per-category keyword evidence for label-noise auditing.

This module provides the keyword sets and counting helpers that the
project's audit scripts use to detect probable label noise.

History: an earlier file `src/audit_ambiance_eval.py` (note misspelling)
defined `KW`, `count_hits`, and `classify`, then was deleted but the
imports in `src/error_analysis.py` and `src/audit_category.py` were
never updated. Those imports now point here (correct spelling).

Public surface:

  KW            : dict[str, set[str]]  — keyword sets per category
  count_hits    : (text, keyword_set) -> int  — keyword occurrence counter
  classify      : legacy helper kept for backward import compatibility

The keyword sets are intentionally HIGH-PRECISION, not exhaustive. They
exist for noise auditing, not for model decisions. Each keyword should
be a phrase that, when present in a complaint, is strong evidence the
complaint is about that category. False positives here translate
directly into wrong audit conclusions, so we err on the side of
fewer-but-stronger keywords.

The ambience entry pulls from `src.config.ambience_keywords` so it
stays in sync with the rest of the v5 ambience pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/config is importable when this module is run from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.arabic_normalization import clean

try:
    from config.ambience_keywords import (
        AMBIENCE_HIGH_PRECISION_PHRASES,
        AMBIENCE_SINGLE_TOKENS,
    )
    _AMBIENCE_KEYWORDS = set(AMBIENCE_HIGH_PRECISION_PHRASES) | set(AMBIENCE_SINGLE_TOKENS)
except ImportError:
    _AMBIENCE_KEYWORDS = set()


# ---------------------------------------------------------------------------
# KW — high-precision keywords per production category.
# Stored in raw surface form; normalized at module load via _normalize().
# ---------------------------------------------------------------------------

_KW_RAW: dict[str, set[str]] = {
    "جودة الطعام": {
        "الاكل", "الطعام", "الوجبه", "الوجبة",
        "البرجر", "الشاورما", "البيتزا", "السندويش", "الساندويتش",
        "الدجاج", "اللحم", "اللحمه", "الرز", "الارز",
        "السلطه", "السلطة", "البطاطس", "العيش", "الخبز",
        "الطعم", "طعمه", "طعمها", "النكهه", "النكهة",
        "بايخ", "مالح", "محروق", "ني", "نيء",
        "الاكل بارد", "الاكل حار",
    },
    "خدمة الموظفين": {
        "الموظف", "الموظفين", "الموظفه", "الموظفة",
        "الكاشير", "النادل", "الويتر", "المضيف", "المضيفه", "المضيفة",
        "اسلوبه", "اسلوبها", "اسلوب",
        "غير محترم", "ما رحب", "ما رحبت", "متجاهل", "متجاهله",
        "ما رد", "ما ردت", "ما اهتم", "ما اهتمت",
        "تعامله", "تعاملها",
    },
    "التوصيل": {
        "المندوب", "المندوبين", "السائق", "الديلفري", "التوصيل",
        "الطلب وصل", "ما وصل الطلب", "ضاع الطلب",
        "تطبيق", "هنقرستيشن", "جاهز", "مرسول", "طلبات",
        "المندوب تاخر", "المندوب تأخر", "المندوب ما رد",
    },
    "السعر والقيمة": {
        "السعر", "الاسعار", "الأسعار",
        "الفاتوره", "الفاتورة",
        "غالي", "غاليه", "غالية", "مكلف", "مكلفه", "مكلفة",
        "ما يستاهل", "مبالغ فيه", "مبالغ فيها",
        "القيمه", "القيمة",
    },
    "النظافة": {
        "النظافه", "النظافة", "نظيف", "نظيفه", "نظيفة", "متسخ", "متسخه", "وسخ", "وسخه",
        "قذر", "قذره", "قذرة", "مقرف", "مقرفه",
        "ذباب", "صراصير", "حشره", "حشرة",
        "الحمام وصخ", "الحمام قذر",
        "صحن وسخ", "ملعقه وسخه",
        "بقع طعام", "بقع",
    },
    "دقة الطلب": {
        "نسوا", "نسي", "ناقص", "ناقصه", "ناقصة",
        "غلط", "خطا", "خطأ",
        "غير اللي طلبته", "ما طلبت", "ما طلبنا",
        "بدون بصل", "بدون مايونيز", "بدون صوص",
        "وضعوا", "حطوا",
    },
    "وقت الانتظار": {
        "انتظرت", "انتظرنا", "ساعه كامله", "ساعه كاملة", "ساعتين",
        "تاخر الطلب", "تأخر الطلب", "وقت طويل", "بطء", "بطيء",
        "البطء", "نص ساعه", "نص ساعة",
        "انتظار طويل",
    },
    "عامة": {
        "تجربه سيئه", "تجربة سيئة",
        "لن اعود", "اخر مره", "آخر مرة", "اول مره واخر مره",
        "ما عجبني", "ما عجبتني",
        "بشكل عام", "عموما", "عمومًا",
        "محبط", "محبطه",
    },
    # Ambience entry — pulled from src/config/ambience_keywords.py so the
    # audit and the v5 pipeline stay in sync.
    "الجو والمكان": _AMBIENCE_KEYWORDS,
}


def _normalize_set(words: set[str]) -> set[str]:
    """Apply the project's `clean()` normalizer to every keyword."""
    return {clean(w) for w in words if clean(w)}


# Pre-normalize at import time so per-row matching is fast.
KW: dict[str, set[str]] = {cat: _normalize_set(words) for cat, words in _KW_RAW.items()}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def count_hits(text: str, keywords: set[str]) -> int:
    """Count how many keywords from `keywords` appear in `text`.

    Both sides are normalized via `clean()` so orthography differences
    (أ vs ا, ة vs ه, tashkeel) don't cause misses. Multi-word phrases
    are matched as substrings; single-word phrases are matched as whole
    words to avoid false positives like "طعم" matching inside "المطعم".
    """
    if not text or not keywords:
        return 0
    cleaned = clean(text)
    if not cleaned:
        return 0
    words = set(cleaned.split())
    hits = 0
    for kw in keywords:
        if " " in kw:
            if kw in cleaned:
                hits += 1
        else:
            if kw in words:
                hits += 1
    return hits


def classify(text: str, target_category: str) -> str:
    """Coarse classification of a row's keyword evidence vs. its gold label.

    Kept for backward import compatibility (`audit_category.py` historically
    imported this name even though it never actually called it). Returns one
    of: "clean_target", "multi_aspect", "probably_mislabeled", "ambiguous".

    For richer analysis, use `audit_category.classify_for_target` directly.
    """
    target_hits = count_hits(text, KW.get(target_category, set()))
    other_hits = {
        c: count_hits(text, kws)
        for c, kws in KW.items()
        if c != target_category
    }
    others_with_hits = {c: h for c, h in other_hits.items() if h > 0}

    if target_hits == 0 and not others_with_hits:
        return "ambiguous"
    if target_hits == 0:
        return "probably_mislabeled"
    if target_hits >= 2 and not others_with_hits:
        return "clean_target"
    if target_hits >= 1 and others_with_hits:
        max_other = max(others_with_hits.values())
        if max_other >= target_hits:
            return "multi_aspect" if max_other == target_hits else "probably_mislabeled"
        return "multi_aspect"
    return "clean_target"


__all__ = ["KW", "count_hits", "classify"]
