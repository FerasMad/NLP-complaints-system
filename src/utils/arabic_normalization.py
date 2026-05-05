"""Canonical Arabic normalization for the project.

This is the single source of truth. Any new script in `src/` should import
`clean` from here. Two existing files duplicate this logic:

  * `app/ensemble_inference.py` defines `clean_arabic` (kept for now to
    avoid breaking the training/local-Gradio path; future cleanup can
    switch it to import from here).

  * `hf_space/app.py` defines `clean` inline because the HuggingFace
    Space is a standalone deployment and cannot import from `src/`.
    A test (`tests/test_arabic_normalization_sync.py`) asserts that
    both copies produce identical outputs for a fixed input set.

Why these specific normalizations:

  * Tashkeel (diacritics) are stripped — most Arabic complaints don't
    use them, and the model's tokenizer doesn't either.
  * أ إ آ ٱ → ا — alef variants collapsed; users type these
    interchangeably in informal writing.
  * ى → ي — alef-maksura → ya, again user-input variation.
  * ة → ه — ta-marbuta → ha, especially common in dialectal writing.
  * Persian/Gulf-loaned chars (پ چ گ ک ی) → Arabic equivalents.
    These appear in Saudi/Gulf social-media writing.
  * Non-Arabic characters → space. Removes punctuation, emoji, etc.
  * Whitespace collapsed and trimmed; lowercased.

We intentionally do NOT stem. Aggressive stemming damages dialect words
("بكتب" / "كتاب" / "كتب" all collapse to "كتب", losing tense).
PySarf handles morphology where needed (in the aspect-extraction layer);
this normalizer stays surface-level.
"""
from __future__ import annotations

import re

# Public regex patterns — re-exported so callers can build on them.
TASHKEEL = re.compile(r"[ً-ٟ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")

# Character translation table — built once.
_TRANS_TABLE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي",
    "ة": "ه",
    # Gulf/Persian-influenced chars in Saudi social-media writing
    "پ": "ب", "چ": "ج", "گ": "ك", "ک": "ك", "ی": "ي",
})


def clean(text: str) -> str:
    """Normalize Arabic text for keyword matching and tokenization.

    Identical behavior to `hf_space/app.py:clean` (verified by a sync test).
    """
    if not isinstance(text, str) or not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(_TRANS_TABLE)
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


# Backward-compat aliases — same function, different historical names.
# `clean_arabic` is what `app/ensemble_inference.py` uses.
# `normalize_ar` is what the research sandbox used.
clean_arabic = clean
normalize_ar = clean


def is_arabic_enough(text: str, min_ratio: float = 0.30) -> bool:
    """Return True if at least `min_ratio` of characters are Arabic.

    Used to reject obvious non-Arabic input before classification.
    """
    if not text or len(text) < 3:
        return False
    return len(ARABIC_CHAR.findall(text)) / max(len(text), 1) >= min_ratio
