"""HuggingFace Spaces entry point for the Arabic Restaurant Complaints Classifier.

Visual identity is warm, hospitable, Saudi-rooted. Multi-section page with a
golden-hour SVG hero, overlapping stats strip, classify, performance charts,
inline about, and footer. Light + dark themes (toggle top-right). Almarai font.
"""
import base64
import html
import os
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Optional Arabic morphology engine (Gulf-dialect aware) for stem-based aspect
# matching. Falls back to surface-form matching if not installed.
try:
    from pysarf import PySarf
    _SARF = PySarf(dialect="gulf")
except Exception as _sarf_err:  # noqa: BLE001
    print(f"PySarf unavailable, falling back to surface matching: {_sarf_err}")
    _SARF = None

HF_REPO_ID = os.environ.get("HF_REPO_ID", "FerasMad/arabic-complaints-classifier")
GITHUB_URL = "https://github.com/FerasMad/NLP-complaints-system"
MAX_LENGTH = 192
MIN_ARABIC_RATIO = 0.30

CATEGORIES = [
    "التوصيل",
    "السعر والقيمة",
    "النظافة",
    "جودة الطعام",
    "خدمة الموظفين",
    "دقة الطلب",
    "عامة",
    "وقت الانتظار",
]
CATEGORIES_EN = {
    "التوصيل": "Delivery",
    "السعر والقيمة": "Price / value",
    "النظافة": "Cleanliness",
    "جودة الطعام": "Food quality",
    "خدمة الموظفين": "Staff service",
    "دقة الطلب": "Order accuracy",
    "عامة": "General",
    "وقت الانتظار": "Wait time",
}
ID2LABEL = dict(enumerate(CATEGORIES))

TASHKEEL = re.compile(r"[ً-ٟ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean(text: str) -> str:
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    # Standard MSA normalization + Gulf/Persian-influenced char fold-in.
    # The Gulf chars (پ, چ, گ, ک, ی) appear in Saudi/Gulf social-media writing.
    t = t.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "پ": "ب", "چ": "ج", "گ": "ك", "ک": "ك", "ی": "ي",
    }))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


def is_arabic_enough(text: str) -> bool:
    if not text or len(text) < 3:
        return False
    return len(ARABIC_CHAR.findall(text)) / max(len(text), 1) >= MIN_ARABIC_RATIO


# ---- Load model + static SVG assets ----------------------------------------

print(f"Loading {HF_REPO_ID} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID).to(device).eval()
# Fail loudly if a wrong-shape model is ever fetched — covers the HF_REPO_ID
# env-override case + future model-card mismatch. (Adversarial Q&A I-5.)
assert model.config.num_labels == len(CATEGORIES), (
    f"loaded model has {model.config.num_labels} labels but Space expects "
    f"{len(CATEGORIES)} ({CATEGORIES}). Refusing to start."
)
print(f"Model loaded on {device}.")

HERE = Path(__file__).parent


def _load_svg(path: Path) -> str:
    """Read an SVG file and strip XML/DOCTYPE prologue so it inlines cleanly."""
    text = path.read_text(encoding="utf-8")
    if text.startswith("<?xml"):
        text = text.split("?>", 1)[1].lstrip()
    if text.startswith("<!DOCTYPE"):
        text = text.split(">", 1)[1].lstrip()
    return text


def _svg_data_uri(path: Path) -> str:
    """Encode an SVG file as a base64 data URI for use in <img src="...">.
    Robust against inline-HTML/SVG interaction issues that can hide charts."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


HERO_SVG = _load_svg(HERE / "hero.svg")
CHART_F1_LIGHT_URI = _svg_data_uri(HERE / "charts" / "per_class_f1.svg")
CHART_F1_DARK_URI = _svg_data_uri(HERE / "charts" / "per_class_f1.dark.svg")
CHART_BL_LIGHT_URI = _svg_data_uri(HERE / "charts" / "vs_baselines.svg")
CHART_BL_DARK_URI = _svg_data_uri(HERE / "charts" / "vs_baselines.dark.svg")


# ---- Aspect extraction (interpretability layer) ----------------------------
# For every input, we scan for category-specific Arabic phrases. This shows
# the user the EVIDENCE behind each prediction — the words and phrases the
# model could have used to decide. Independent of the classifier output.

ASPECT_VOCAB: dict[str, list[str]] = {
    "جودة الطعام": [
        # phrases (matched first because longer)
        "اللحم محروق", "بدون طعم", "ما عجبني الاكل", "الطعم سيء",
        # words
        "الاكل", "الطعام", "بايخ", "مالح", "محروق", "نيء", "طعم",
        "البرجر", "الطبخ", "الوجبه", "البيتزا", "الكبسه", "اللحم", "الدجاج",
        "بارد", "حلو", "مذاق", "نكهه", "متقن",
    ],
    "التوصيل": [
        "ضاع الطلب", "المندوب تاخر", "العنوان غلط", "ما رد المندوب",
        "المندوب", "السائق", "ديليفري", "موصل", "العنوان", "التوصيل",
    ],
    "خدمة الموظفين": [
        "غير محترم", "اسلوبه سيء", "موظف وقح", "ما يبتسم", "صاح علي",
        "الموظف", "النادل", "الكاشير", "العمال", "وقح", "محترم", "اسلوب",
    ],
    "وقت الانتظار": [
        "ساعه كامله", "وقت طويل", "قبل ان ياتي", "تاخير الخدمه",
        "انتظر", "انتظرت", "انتظرنا", "ساعه", "ساعتين",
        "دقيقه", "تاخير", "طويل", "جلسنا", "نص ساعه", "نصف ساعه",
    ],
    "النظافة": [
        "ما ينظف", "غير نظيف",
        "الحمام", "تواليت", "متسخ", "وسخ", "اوساخ",
        "ذباب", "صراصير", "نظيف", "نظافه", "قذر",
    ],
    "السعر والقيمة": [
        "ما يستاهل", "مبالغ فيها", "غالي جدا",
        "غالي", "غاليه", "اسعار", "الفاتوره", "السعر", "يستاهل",
    ],
    "دقة الطلب": [
        "غلط في الطلب", "الطلب غلط", "نسوا الطلب",
        "ناقص", "نسوا", "غلط", "بدلوا", "خلطوا", "وضعوا بدلا",
    ],
}

# Category → CSS class suffix (no Arabic in CSS selectors)
ASPECT_CSS_ID = {
    "جودة الطعام": "food",
    "التوصيل": "delivery",
    "خدمة الموظفين": "service",
    "وقت الانتظار": "wait",
    "النظافة": "clean",
    "السعر والقيمة": "price",
    "دقة الطلب": "accuracy",
}


def _build_vocab_indices():
    """Pre-compute two indices for aspect matching:
      - phrases per aspect (multi-word, exact-match w/ prefix tolerance)
      - stem-to-aspect map (single-word, morphology-aware via PySarf)

    The stem index is what makes 'والمندوبين' match the 'المندوب' vocab entry,
    or 'بطعم' match 'طعم', without listing every surface form.
    """
    phrases_by_aspect: dict[str, list[str]] = {}
    stem_to_aspect: dict[str, str] = {}
    for aspect, items in ASPECT_VOCAB.items():
        for item in items:
            words = item.split()
            if len(words) > 1:
                phrases_by_aspect.setdefault(aspect, []).append(item)
                continue
            # Single word: register the surface form AND its stem (if PySarf available)
            phrases_by_aspect.setdefault(aspect, []).append(item)
            if _SARF is not None:
                try:
                    stem = _SARF.analyze(words[0]).stem
                except Exception:
                    stem = None
                if stem and stem not in stem_to_aspect:
                    stem_to_aspect[stem] = aspect
    return phrases_by_aspect, stem_to_aspect


_PHRASES_BY_ASPECT, _STEM_TO_ASPECT = _build_vocab_indices()


# Single-char prefixes that attach to words in Arabic (و, ف, ب, ل, ك, س).
# When checking a word boundary at the START of a phrase, allow these prefixes
# so 'المندوب' matches inside 'والمندوب' (and-the-delivery-person).
ARABIC_PREFIX_CHARS = set("وفبلكس")


def _is_word_boundary_start(text: str, idx: int) -> bool:
    """True if position idx is the start of a word (text start, after space,
    or after a one-char Arabic prefix that itself follows a space)."""
    if idx <= 0:
        return True
    if text[idx - 1].isspace():
        return True
    # Allow common Arabic prefix attached to the preceding char
    if text[idx - 1] in ARABIC_PREFIX_CHARS and (idx - 2 < 0 or text[idx - 2].isspace()):
        return True
    return False


def _is_word_boundary_end(text: str, idx: int) -> bool:
    """True if position idx is the end of a word (text end or before space)."""
    if idx >= len(text):
        return True
    return text[idx].isspace()


def extract_aspects(cleaned_text: str) -> tuple[list[tuple[int, int, str, str]], dict[str, list[str]]]:
    """Find aspect-specific phrases in the cleaned text.

    Two-pass matching:
      1. Multi-word phrases match exactly (with one-char Arabic prefix tolerance)
      2. Single-word matching uses PySarf stems so 'والمندوبين' matches the
         'المندوب' vocab entry without listing every surface form

    Word-boundary aware: 'طعم' in 'المطعم' (restaurant) does NOT match the
    food keyword, because their stems are 'مطعم' vs 'طعم'.

    Returns:
        matches: list of (start, end, aspect, matched_phrase) sorted by start
        by_aspect: {aspect_name: [matched_phrases]}
    """
    # Defensive: clean() is idempotent. Callers SHOULD pass cleaned text,
    # but if raw text arrives (e.g., a future caller forgets) we still want
    # alif/taa-marbuta folding to apply so phrase matching works.
    # (Adversarial Q&A I-8.)
    cleaned_text = clean(cleaned_text)
    raw_matches: list[tuple[int, int, str, str]] = []

    # Pass 1: phrase matching (exact substring with prefix-tolerant boundary)
    for aspect, phrases in _PHRASES_BY_ASPECT.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            start = 0
            while True:
                idx = cleaned_text.find(phrase, start)
                if idx == -1:
                    break
                end = idx + len(phrase)
                if _is_word_boundary_start(cleaned_text, idx) and _is_word_boundary_end(cleaned_text, end):
                    raw_matches.append((idx, end, aspect, phrase))
                start = idx + 1

    # Pass 2: stem matching for any single word in the input (PySarf only)
    if _SARF is not None and _STEM_TO_ASPECT:
        char_pos = 0
        for word in cleaned_text.split():
            word_start = cleaned_text.find(word, char_pos)
            if word_start < 0:
                char_pos += len(word) + 1
                continue
            char_pos = word_start + len(word)
            try:
                stem = _SARF.analyze(word).stem
            except Exception:
                continue
            if stem and stem in _STEM_TO_ASPECT:
                raw_matches.append((word_start, word_start + len(word), _STEM_TO_ASPECT[stem], word))

    # Greedy longest-match-first dedup so multi-word phrase matches beat
    # overlapping single-word stem matches
    raw_matches.sort(key=lambda m: -(m[1] - m[0]))
    final: list[tuple[int, int, str, str]] = []
    covered: set[int] = set()
    for m in raw_matches:
        s, e = m[0], m[1]
        if any(i in covered for i in range(s, e)):
            continue
        final.append(m)
        covered.update(range(s, e))
    final.sort(key=lambda m: m[0])

    by_aspect: dict[str, list[str]] = {}
    for s, e, asp, phrase in final:
        by_aspect.setdefault(asp, []).append(phrase)

    return final, by_aspect


def annotate_text(text: str, matches: list[tuple[int, int, str, str]]) -> str:
    """Wrap matched ranges with <mark> spans tagged by aspect.

    User text is escaped before HTML interpolation — the inner-text path is
    fragile-by-construction and Gradio's outer sanitizer does not cover
    template-string interpolation. (Code audit XSS finding.)
    """
    if not matches:
        return html.escape(text)
    out = []
    last = 0
    for s, e, aspect, _phrase in matches:
        if s > last:
            out.append(html.escape(text[last:s]))
        css = ASPECT_CSS_ID.get(aspect, "other")
        out.append(
            f'<mark class="aspect-mark aspect-{css}" '
            f'title="{html.escape(CATEGORIES_EN.get(aspect, aspect))}">'
            f'{html.escape(text[s:e])}'
            f'</mark>'
        )
        last = e
    if last < len(text):
        out.append(html.escape(text[last:]))
    return "".join(out)


def render_understanding(
    cleaned_text: str,
    matches: list[tuple[int, int, str, str]] | None = None,
    by_aspect: dict[str, list[str]] | None = None,
) -> str:
    """Render the 'how I read this' interpretability panel.

    Optionally accepts pre-computed matches/by_aspect so the caller can avoid
    running extract_aspects twice (once for multi-aspect detection, once here).
    """
    if matches is None or by_aspect is None:
        matches, by_aspect = extract_aspects(cleaned_text)

    if not by_aspect:
        return (
            '<div class="understanding understanding-empty">'
            '  <div class="understanding-head">'
            '    <span class="understanding-eyebrow">how I read this</span>'
            '    <strong>كيف فهمت شكواك</strong>'
            '  </div>'
            f'  <div class="understanding-text">{html.escape(cleaned_text)}</div>'
            '  <p class="understanding-note">'
            '    لم أجد أي كلمة تشير إلى جانب محدد في النص. '
            '    <em>no aspect-specific words detected.</em>'
            '  </p>'
            '</div>'
        )

    annotated = annotate_text(cleaned_text, matches)

    chips = []
    for aspect, phrases in by_aspect.items():
        css = ASPECT_CSS_ID.get(aspect, "other")
        en = CATEGORIES_EN.get(aspect, "")
        # dedupe phrases preserving order
        seen = []
        for p in phrases:
            if p not in seen:
                seen.append(p)
        # Escape user-derived `seen` phrases; aspect/en come from internal
        # constants and don't need escaping but we apply for safety.
        sample = "، ".join(html.escape(s) for s in seen[:4])
        chips.append(
            f'<div class="aspect-chip aspect-{css}">'
            f'  <span class="aspect-chip-label">'
            f'    <span class="aspect-chip-cat">{html.escape(aspect)}</span>'
            f'    <span class="aspect-chip-en">{html.escape(en)}</span>'
            f'  </span>'
            f'  <span class="aspect-chip-evidence">{sample}</span>'
            f'</div>'
        )

    return (
        '<div class="understanding">'
        '  <div class="understanding-head">'
        '    <span class="understanding-eyebrow">how I read this</span>'
        '    <strong>كيف فهمت شكواك</strong>'
        '  </div>'
        f'  <div class="understanding-text">{annotated}</div>'
        f'  <div class="aspect-chips">{"".join(chips)}</div>'
        '</div>'
    )


# ---- Prediction ------------------------------------------------------------

EMPTY_RESULT = """
<div class="result-empty">
  <div class="result-empty-text">
    <strong>اكتب شكوى أو اضغط على مثال أدناه</strong>
    <span>type a complaint above, or pick an example below</span>
  </div>
</div>
"""


def is_multi_aspect(
    top: list[tuple[str, float]],
    by_aspect: dict[str, list[str]] | None = None,
) -> bool:
    """Detect genuinely multi-aspect complaints.

    Source of truth is aspect extraction (what's actually in the text), not
    model probabilities (which conflate "two real aspects" with "model uncertain
    between two related categories"). A 51%/49% split between wait-time and
    food-quality on a pure wait-time complaint reflects model confusion, not
    a multi-aspect input — only the aspect-extraction layer can tell the
    difference. Fires only when 2+ distinct aspect categories are detected.
    """
    return by_aspect is not None and len(by_aspect) >= 2


def render_general_fallback(top: list[tuple[str, float]]) -> str:
    """Special render for when 'general' is the top-1 prediction.

    'general' is a fallback that doesn't help users route or act on the
    complaint. We reframe it as 'no specific aspect detected' and surface
    the next 2 specific categories so the user has SOMETHING actionable.
    """
    # Filter out general; show the next 2 specific categories with their (low) confidence
    specific = [(c, s) for c, s in top if c != "عامة"][:2]
    rows = []
    for cat, score in specific:
        en = CATEGORIES_EN.get(cat, "")
        pct = f"{score * 100:.0f}%"
        rows.append(
            f'<div class="result-row result-rank-fallback">'
            f'  <div class="result-meta">'
            f'    <span class="result-rank">·</span>'
            f'    <span class="result-en">{en}</span>'
            f'  </div>'
            f'  <div class="result-cat">{cat}</div>'
            f'  <div class="result-pct">{pct}</div>'
            f'</div>'
        )

    return (
        '<div class="general-fallback">'
        '  <div class="general-headline">'
        '    <strong>شكوى عامة</strong>'
        '    <span>general complaint — covers the overall experience</span>'
        '  </div>'
        f'  <div class="general-rail">'
        f'    <span class="general-rail-label">قد تتعلّق بأحد هذه الجوانب:</span>'
        f'    <div class="result-stack">{"".join(rows)}</div>'
        f'  </div>'
        '</div>'
    )


def render_result(
    top: list[tuple[str, float]],
    by_aspect: dict[str, list[str]] | None = None,
    full_probs: dict[str, float] | None = None,
    rescued_cat: str | None = None,
) -> str:
    """Render the colored category rail.

    Source of truth: `by_aspect` (the dict from extract_aspects), the same
    source the "كيف فهمت" panel uses — so they're always consistent.

    Algorithm:
      - If aspect-extraction found >=1 categories: show all of them, sorted
        by the model's softmax probability (highest = darker, at top).
        Cap at 4 to prevent very long reviews from cluttering the rail.
      - If aspect-extraction found 0 categories: fall back to model's top-1
        (single badge — the historical behavior for vague inputs).
      - If a rescue fired for a category extract_aspects missed: force-include
        the rescued category at the top so the rail reflects the actual
        prediction.

    Visual:
      - Index 0 (most-confident among displayed): result-rank-top (darker)
      - Indices 1..N: result-rank-other (standard)
      - Multi-aspect badge: fired whenever >=2 badges are shown.
    """
    if not top:
        return EMPTY_RESULT

    # General fallback unchanged — when "عامة" is the prediction, show the
    # "شكوى عامة" reframing UI instead of the rail.
    if top[0][0] == "عامة":
        return render_general_fallback(top)

    # Build the categories list — aspect-extraction drives the rail.
    aspect_cats: list[str] = list((by_aspect or {}).keys())

    # Sort by model softmax (descending) — most-confident first.
    if full_probs is not None and aspect_cats:
        aspect_cats.sort(key=lambda c: full_probs.get(c, 0.0), reverse=True)

    # Force-include the rescued category at the top so the rail matches the
    # actual prediction. (Rare: rescue usually overlaps with aspect vocab,
    # but edge cases exist where keyword rescue fires and extract_aspects
    # doesn't catch the same signal.)
    if rescued_cat and rescued_cat != "عامة":
        if rescued_cat in aspect_cats:
            aspect_cats.remove(rescued_cat)
        aspect_cats.insert(0, rescued_cat)

    # Cap at 4 — beyond this the rail loses visual hierarchy.
    aspect_cats = aspect_cats[:4]

    # Fall back to model's top-1 if no aspects detected (rare — usually
    # means very short input or unusual phrasing).
    if not aspect_cats:
        aspect_cats = [top[0][0]]

    # Render. Index 0 = darker top, the rest = standard.
    rows = []
    for rank, cat in enumerate(aspect_cats):
        en = CATEGORIES_EN.get(cat, "")
        css_class = "result-rank-top" if rank == 0 else "result-rank-other"
        rows.append(
            f'<div class="result-row {css_class}">'
            f'  <div class="result-meta">'
            f'    <span class="result-en">{en}</span>'
            f'  </div>'
            f'  <div class="result-cat">{cat}</div>'
            f'</div>'
        )

    # Multi-aspect badge whenever the rail shows 2+ categories.
    badge = ""
    if len(aspect_cats) >= 2:
        badge = (
            '<div class="multi-aspect-badge">'
            '  <span class="multi-aspect-mark">⊕</span>'
            '  <span class="multi-aspect-text">'
            '    <strong>تشمل الشكوى أكثر من جانب</strong>'
            '    <span>multi-aspect complaint, all detected aspects shown</span>'
            '  </span>'
            '</div>'
        )

    return f'{badge}<div class="result-stack">{"".join(rows)}</div>'


def render_message(headline_ar: str, headline_en: str) -> str:
    return (
        f'<div class="result-message">'
        f'  <strong>{headline_ar}</strong>'
        f'  <span>{headline_en}</span>'
        f'</div>'
    )


CATEGORY_TO_ID = {cat: i for i, cat in ID2LABEL.items()}

# Rescue rules: unambiguous Arabic phrases that must produce the matched
# category as top-1.
#
# ORDER IS LOAD-BEARING — first match wins (see `break` in apply_rescue).
# Most-specific phrases first, with the catch-all `عامة` last.
#
# NOTE: this layer has diverged from app/ensemble_inference.py::apply_keyword_priors.
# The Space adds: (a) the `عامة` rescue category, (b) first-match-only
# semantics, (c) 0.30x dampening on non-rescued classes. The "Audit-validated
# 85% → 100%" claim in `apply_rescue` was measured against the API's older
# rescue, not this one — treat the docstring as historical context, not a
# current measurement. The 500-row Space test in dataset/_audits/test_run_results.md
# is the authoritative measurement for this rescue layer.
RESCUE_RULES: list[tuple[str, list[str]]] = [
    ("النظافة", ["الحمام", "تواليت", "ذباب", "صراصير"]),
    ("وقت الانتظار", [
        "انتظرت ساعه", "انتظرت ساعتين", "ساعه كامله في المطعم",
        "انتظرنا ساعه", "انتظرنا ساعتين",
    ]),
    ("التوصيل", ["ضاع الطلب", "المندوب تاخر", "المندوب ما رد"]),
    ("جودة الطعام", ["الطبخ", "اللحم محروق", "بدون طعم", "الاكل بايخ"]),
    # General-complaint rescue — added after live-Space audit found
    # "تجربه سيئه عموما لن اعود" being misclassified as السعر والقيمة.
    # These phrases are unambiguous "no specific aspect, overall bad"
    # signals that the model under-fires on. Conservative phrase set —
    # full sentences only, not bare words.
    ("عامة", [
        "تجربه سيئه عموما", "تجربة سيئة عموما",
        "لن اعود", "لن أعود",
        "اخر مره اطلب", "آخر مرة اطلب",
        "اول مره واخر مره", "اول مرة واخر مرة",
        "ما عجبني شي", "ما عجبني شيء",
        "بشكل عام سيء", "بشكل عام سيئ",
        "تجربه محبطه", "تجربة محبطة",
    ]),
]


def apply_rescue(
    probs,
    cleaned_text: str,
    rescue_floor: float = 0.55,
    other_class_dampening: float = 0.30,
):
    """If an unambiguous keyword is present, force its category to top
    AND suppress the other categories so the rail reflects the rescue
    decision instead of the model's pre-rescue prior.

    Why dampening exists: the original asymmetric rescue (boost target only)
    produced rails like "51% wait time / 49% food quality" on pure
    wait-time complaints. Root cause: the model has a strong food-quality
    prior (47% of training data is food complaints), so even after rescue
    sets the wait floor, the original 0.95 food probability re-normalizes
    to ~0.49. The rail looked like the model was hedging when it was
    actually rescue-overridden. Dampening other classes by 70% before
    re-normalize gives a cleaner ~78%/22% split that matches user intent.

    Conservative: only rescues when phrase clearly indicates one category.

    Historical context: an early version of this rescue (no dampening, 4
    categories, iterate-all) was audited via src/audit_predictions.py and
    moved the API model from 85% to 100% on a 20-row behavioral set. The
    current Space layer has diverged from that — the canonical measurement
    for THIS rescue is the 500-row Space test in
    dataset/_audits/test_run_results.md, not the older 20-row audit.
    """
    out = probs.copy()
    rescued_idx = None
    for cat, phrases in RESCUE_RULES:
        if cat not in CATEGORY_TO_ID:
            continue
        if any(p in cleaned_text for p in phrases):
            cat_idx = CATEGORY_TO_ID[cat]
            current_max = float(out.max())
            out[cat_idx] = max(out[cat_idx], current_max + 0.05, rescue_floor)
            rescued_idx = cat_idx
            break  # only one rescue fires per row — first match wins

    # When rescue fired, dampen the other classes so the rail reflects
    # the rescue's confidence instead of the model's original prior.
    if rescued_idx is not None:
        for i in range(len(out)):
            if i != rescued_idx:
                out[i] *= other_class_dampening

    s = out.sum()
    if s > 0:
        out = out / s
    # Return (normalized probs, rescued category index or None).
    # render_result uses rescued_idx to force-include the rescued
    # category in the rail even when extract_aspects missed it.
    return out, rescued_idx


# Praise vs complaint screen — protects against the "ممتاز جدا شكرا" failure
# where positive feedback gets dumped into 'عامة' with 100% confidence.
# Word-set matching (not substring) so short particles like "لا" / "ما" don't
# false-match inside common words like "الاكل" (which contains "لا").
PRAISE_WORDS = frozenset({
    "ممتاز", "ممتازه", "ممتازة", "رائع", "رائعه", "رائعة", "احسن", "أحسن",
    "افضل", "أفضل", "جميل", "جميله", "جميلة", "حلو", "حلوه", "حلوة",
    "لذيذ", "لذيذه", "لذيذة", "نظيف", "نظيفه", "نظيفة",
    "مريح", "مريحه", "مريحة", "سريع", "سريعه", "سريعة",
    "شكرا", "شكراً", "احب", "أحب", "نصحت", "انصح", "أنصح",
})
NEGATIVE_WORDS = frozenset({
    "سيء", "سيئ", "سيئه", "سيئة", "سيئا", "بايخ", "بايخه", "بايخة",
    "مالح", "مالحه", "مالحة", "بارد", "بارده", "باردة",
    "تاخر", "تاخرت", "تأخر", "تأخرت", "متأخر", "متاخر",
    "قذر", "قذره", "قذرة", "متسخ", "متسخه", "متسخة", "وسخ", "وسخه", "وسخة",
    # negation particles — only match as whole words.
    # Includes ليس/ليست (formal MSA), ابدا/ابداً (emphatic Saudi negator),
    # and Gulf colloquial مهو/مهي. (Adversarial Q&A I-2.)
    "غير", "ما", "لا", "مو", "مب", "ليس", "ليست", "ابدا", "ابداً",
    "مهو", "مهي", "مهوب", "مهوش",
    "مزعج", "مزعجه", "مزعجة", "غالي", "غاليه", "غالية",
    "ضاع", "ضاعت", "محروق", "محروقه", "محروقة", "نسي", "نسوا",
    "مشكله", "مشكلة", "مشاكل", "خربان", "خربانه", "خربانة",
})

# Contrastive markers — if any appear in the cleaned text, skip the praise
# screen. Inputs like "ممتاز بس الجو حار" or "الموظف رائع لكن الكاشير غلط"
# are legitimate complaints framed politely; the bare praise word should not
# swallow them. (Adversarial Q&A I-3.)
CONTRASTIVE_MARKERS = frozenset({
    "بس", "لكن", "لاكن", "مع", "رغم", "بالرغم", "الا", "إلا",
})


def looks_like_praise(cleaned: str) -> bool:
    """True when the input has positive sentiment words and no negatives.

    Uses whole-word matching (not substring) so short negation particles like
    'لا' / 'ما' don't false-match inside common words such as 'الاكل'.

    Contrastive markers (بس، لكن، رغم، ...) bypass the praise screen so that
    "ممتاز بس الجو حار" reaches the model — a mixed-sentiment complaint
    should not be discarded just because it contains one praise word.
    """
    words = set(cleaned.split())
    if words & CONTRASTIVE_MARKERS:
        return False
    has_praise = bool(words & PRAISE_WORDS)
    has_negative = bool(words & NEGATIVE_WORDS)
    return has_praise and not has_negative


@torch.no_grad()
def predict(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return render_message(
            "اكتب شكوى أطول من ٣ أحرف",
            "type a longer complaint (at least 3 characters)",
        )
    # DoS guard — the model truncates at MAX_LENGTH (192) tokens anyway, but
    # the regex passes in clean() are O(n) over the raw input. A 1MB string
    # would pin the cpu-basic worker; cap at 4000 chars (~500 Arabic words,
    # well above any real complaint). (Adversarial Q&A I-10, Code audit #3.)
    if len(text) > 4000:
        text = text[:4000]
    if not is_arabic_enough(text):
        return render_message(
            "النص ليس بالعربية",
            "the input doesn't appear to be Arabic",
        )

    cleaned = clean(text)

    # Topic-only abstain: a single bare word like "الاكل" has no opinion.
    # Require ≥2 words so the model only fires on actual complaint shapes.
    if len(cleaned.split()) < 2:
        return render_message(
            "اكتب شكوى تحتوي على رأي محدد، وليس كلمة واحدة فقط",
            "type a complaint with an opinion, not just a topic word",
        )

    # Praise screen: route obvious praise away from the classifier so it
    # doesn't get dumped into 'عامة' at 100% confidence.
    if looks_like_praise(cleaned):
        return render_message(
            "هذا يبدو ثناءً وليس شكوى",
            "this looks like praise, not a complaint — try describing a problem",
        )

    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().numpy()
    probs, rescued_idx = apply_rescue(probs, cleaned)
    # Full probability map so render_result can sort aspect-extracted
    # categories by the model's softmax — drives the "darker = top" choice.
    full_probs = {ID2LABEL[int(i)]: float(probs[i]) for i in range(len(probs))}
    top_idx = probs.argsort()[::-1][:3]
    top = [(ID2LABEL[int(i)], float(probs[i])) for i in top_idx]
    rescued_cat = ID2LABEL[int(rescued_idx)] if rescued_idx is not None else None

    # Compute aspects once and reuse — also feeds the multi-aspect detector
    # so 100%/0% predictions still trigger the multi-aspect badge when two
    # distinct aspect categories are present in the text.
    matches, by_aspect = extract_aspects(cleaned)
    return (
        render_result(top, by_aspect, full_probs=full_probs, rescued_cat=rescued_cat)
        + render_understanding(cleaned, matches, by_aspect)
    )


EXAMPLES = [
    ["وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين"],
    ["الاسعار مبالغ فيها لا تناسب الجوده المقدمه ابدا"],
    ["النظافه سيئه الطاولات متسخه والارض غير نظيفه"],
    ["طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي"],
    ["انتظرت ساعه كامله في المطعم قبل ان ياتي طلبي"],
    ["الموظف اسلوبه سيء جدا وغير محترم"],
    ["الاكل بايخ ومالح والطبخ مو متقن"],
    ["تجربه سيئه عموما لن اعود لهذا المكان"],
]


# ---- Theme toggle (head injection: favicon + theme JS) ---------------------

HEAD = """
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='15' fill='%23C75D3D'/><text x='16' y='23' font-size='20' font-weight='800' text-anchor='middle' fill='%23F5EFE6' font-family='sans-serif'>ش</text></svg>">
<script>
(function() {
  const KEY = 'arabic-complaints-theme';
  function apply(theme) {
    if (theme === 'dark') {
      document.body.classList.add('dark');
    } else {
      document.body.classList.remove('dark');
    }
  }
  function wireToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn || btn.dataset.wired === '1') return false;
    btn.dataset.wired = '1';
    btn.addEventListener('click', function() {
      const next = document.body.classList.contains('dark') ? 'light' : 'dark';
      try { localStorage.setItem(KEY, next); } catch (e) {}
      apply(next);
    });
    return true;
  }
  function init() {
    let saved = null;
    try { saved = localStorage.getItem(KEY); } catch (e) {}
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    apply(saved || (prefersDark ? 'dark' : 'light'));

    if (wireToggle()) return;
    const observer = new MutationObserver(function() {
      if (wireToggle()) observer.disconnect();
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
</script>
"""


# ---- Visual identity --------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Almarai:wght@300;400;700;800&display=swap');

:root {
    /* Light theme tokens */
    --surface: #F5EFE6;
    --paper: #EEE6D7;
    --paper-deep: #E5DAC5;
    --ink: #2D211A;
    --ink-muted: #5C4F45;
    --terracotta: #C75D3D;
    --terracotta-deep: #A14828;
    --terracotta-tint: rgba(199, 93, 61, 0.10);
    --olive: #5F6845;
    --olive-tint: rgba(95, 104, 69, 0.10);
    --border: #D9CFC0;
    --border-strong: #C9BDA8;
    --shadow: 0 12px 40px -12px rgba(45, 17, 8, 0.25);
    --hero-ground: #3A1810;
    --footer-bg: #2D211A;
    --footer-text: rgba(245, 239, 230, 0.7);
}

body.dark {
    /* Dark theme tokens — warm-tinted, never pure black */
    --surface: #1A140F;
    --paper: #241B14;
    --paper-deep: #2E2218;
    --ink: #F0E7D8;
    --ink-muted: #A89C8C;
    --terracotta: #D87852;
    --terracotta-deep: #C75D3D;
    --terracotta-tint: rgba(216, 120, 82, 0.14);
    --olive: #8A9468;
    --olive-tint: rgba(138, 148, 104, 0.12);
    --border: #3A2D24;
    --border-strong: #4A3A2D;
    --shadow: 0 12px 40px -12px rgba(0, 0, 0, 0.65);
    --hero-ground: #0F0805;
    --footer-bg: #0F0A07;
    --footer-text: rgba(240, 231, 216, 0.6);
}

* { font-family: 'Almarai', system-ui, -apple-system, sans-serif !important; box-sizing: border-box; }

html, body, .gradio-container {
    background: var(--surface) !important;
    color: var(--ink) !important;
    margin: 0 !important;
    padding: 0 !important;
    transition: background-color 240ms ease, color 240ms ease;
}

.gradio-container {
    max-width: none !important;
    margin: 0 !important;
    padding: 0 !important;
}

.gradio-container > .main,
.gradio-container > .main > .wrap {
    padding: 0 !important;
    max-width: none !important;
    gap: 0 !important;
}

/* ---- Theme toggle ---- */

#theme-toggle-wrap {
    position: fixed;
    top: 18px;
    right: 18px;
    z-index: 50;
}

#theme-toggle {
    width: 42px;
    height: 42px;
    border-radius: 50%;
    border: 1px solid rgba(245, 239, 230, 0.30);
    background: rgba(45, 17, 8, 0.55);
    backdrop-filter: blur(8px);
    color: #F5EFE6;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    transition: background 160ms ease, border-color 160ms ease, transform 160ms ease;
}

#theme-toggle:hover {
    background: rgba(45, 17, 8, 0.75);
    border-color: rgba(245, 239, 230, 0.55);
    transform: scale(1.05);
}

#theme-toggle:active { transform: scale(0.95); }

#theme-toggle svg { width: 18px; height: 18px; }
#theme-toggle .icon-sun { display: none; }
#theme-toggle .icon-moon { display: block; }

body.dark #theme-toggle {
    background: rgba(240, 231, 216, 0.10);
    border-color: rgba(240, 231, 216, 0.25);
    color: #F0E7D8;
}
body.dark #theme-toggle:hover {
    background: rgba(240, 231, 216, 0.18);
}
body.dark #theme-toggle .icon-sun { display: block; }
body.dark #theme-toggle .icon-moon { display: none; }

/* ---- Hero ---- */

#hero {
    position: relative;
    width: 100%;
    height: clamp(440px, 60vh, 560px);
    overflow: hidden;
    color: #F5EFE6;
    background: var(--hero-ground);
}

#hero .hero-bg { position: absolute; inset: 0; width: 100%; height: 100%; }
#hero .hero-bg svg { width: 100%; height: 100%; display: block; }

#hero .hero-content {
    position: relative;
    z-index: 2;
    height: 100%;
    max-width: 1080px;
    margin: 0 auto;
    padding: clamp(48px, 7vw, 88px) clamp(24px, 5vw, 56px);
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    direction: rtl;
}

#hero .eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(245, 239, 230, 0.85);
    margin: 0 0 16px;
    direction: ltr;
    text-align: right;
}

#hero h1 {
    font-size: clamp(2.4rem, 6vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.01em;
    margin: 0 0 14px;
    color: #F5EFE6;
    max-width: 16ch;
    text-shadow: 0 2px 24px rgba(45, 17, 8, 0.35);
}

#hero .lede {
    font-size: clamp(1rem, 1.5vw, 1.18rem);
    line-height: 1.7;
    color: rgba(245, 239, 230, 0.92);
    max-width: 56ch;
    margin: 0;
    text-shadow: 0 1px 12px rgba(45, 17, 8, 0.4);
}

/* ---- Headline figure (replaces 4-cell stats strip) ---- */

#headline-wrap { width: 100%; background: transparent; }

#headline {
    max-width: 1080px;
    margin: clamp(48px, 6vw, 72px) auto 0;
    padding: 0 clamp(24px, 5vw, 56px);
    position: relative;
    z-index: 3;
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: clamp(20px, 4vw, 56px);
    direction: ltr;
}

.headline-stat {
    display: flex;
    align-items: baseline;
    gap: 12px;
    flex-shrink: 0;
}

.headline-figure {
    font-size: clamp(3.6rem, 8vw, 5.4rem);
    font-weight: 800;
    color: var(--terracotta-deep);
    line-height: 1;
    letter-spacing: -0.03em;
    font-variant-numeric: tabular-nums;
}

body.dark .headline-figure { color: var(--terracotta); }

.headline-unit {
    font-size: clamp(0.85rem, 1.1vw, 0.95rem);
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-muted);
    transform: translateY(-6px);
}

.headline-note {
    font-size: clamp(0.92rem, 1.2vw, 1rem);
    color: var(--ink-muted);
    line-height: 1.6;
    margin: 0;
    max-width: 42ch;
    flex: 1;
    min-width: 240px;
    direction: ltr;
}

/* ---- Section scaffolding ---- */

.section {
    width: 100%;
    padding: clamp(56px, 8vw, 96px) clamp(24px, 5vw, 56px);
    transition: background-color 240ms ease;
}

.section-inner { max-width: 1080px; margin: 0 auto; }
.section--paper { background: var(--paper); }
.section--surface { background: var(--surface); }

.section-eyebrow {
    font-size: 0.74rem !important;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--terracotta-deep);
    margin: 0 0 12px;
    direction: ltr;
    text-align: right;
}

.section-title {
    font-size: clamp(1.6rem, 3vw, 2.2rem);
    font-weight: 800;
    color: var(--ink);
    margin: 0 0 14px;
    line-height: 1.2;
    direction: rtl;
    text-align: right;
}

.section-lede {
    font-size: clamp(0.95rem, 1.3vw, 1.05rem);
    color: var(--ink-muted);
    line-height: 1.7;
    margin: 0 0 36px auto;
    max-width: 60ch;
    direction: rtl;
    text-align: right;
}

/* ---- Classify section ---- */

#section-classify { padding-top: clamp(72px, 10vw, 120px); }

#input_panel { background: transparent; padding: 0; }

#input_panel .gr-form,
#input_panel .form,
#input_panel .gr-block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#input_panel textarea {
    width: 100%;
    background: var(--surface) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 22px 24px !important;
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
    transition: border-color 160ms ease, box-shadow 160ms ease, background-color 240ms ease, color 240ms ease;
    resize: vertical;
    min-height: 140px;
    direction: rtl;
    text-align: right;
}

body.dark #input_panel textarea { background: var(--paper) !important; }

#input_panel textarea::placeholder {
    color: var(--ink-muted) !important;
    opacity: 0.55;
    font-weight: 400;
}

#input_panel textarea:focus {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 4px var(--terracotta-tint) !important;
    outline: none;
}

#input_panel label > span { display: none !important; }

#actions { display: flex; gap: 12px; margin-top: 18px; direction: rtl; }

button.primary {
    background: var(--ink) !important;
    color: var(--surface) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    cursor: pointer;
    transition: background 140ms ease, transform 140ms ease, color 240ms ease;
}

button.primary:hover {
    background: var(--terracotta-deep) !important;
    color: #F5EFE6 !important;
    transform: translateY(-1px);
}

button.secondary {
    background: transparent !important;
    color: var(--ink-muted) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 12px !important;
    padding: 14px 22px !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    cursor: pointer;
    transition: all 140ms ease;
}

button.secondary:hover {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border-color: var(--ink-muted) !important;
}

/* ---- Result panel ---- */

#result_panel {
    margin-top: clamp(28px, 4vw, 40px);
    padding-top: clamp(28px, 4vw, 40px);
    border-top: 1px solid var(--border);
}

#result_panel .gr-html,
#result_panel .gr-block {
    background: transparent !important;
    padding: 0 !important;
    border: none !important;
}

.result-empty {
    padding: 28px 24px;
    border: 1px dashed var(--border-strong);
    border-radius: 14px;
    background: var(--surface);
    color: var(--ink-muted);
    direction: rtl;
}
body.dark .result-empty { background: var(--paper); }

.result-empty-text { display: flex; flex-direction: column; gap: 4px; }
.result-empty-text strong { font-size: 1.05rem; font-weight: 700; color: var(--ink); }
.result-empty-text span { font-size: 0.9rem; color: var(--ink-muted); }

.result-message {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 22px 24px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 14px;
    direction: rtl;
}
.result-message strong { font-size: 1.1rem; font-weight: 700; color: var(--ink); }
.result-message span { font-size: 0.92rem; color: var(--ink-muted); }

.result-stack { display: grid; gap: 10px; direction: rtl; }

.result-row {
    display: grid;
    grid-template-columns: auto 1fr auto;
    align-items: baseline;
    gap: 18px;
    padding: 22px 24px;
    border-radius: 14px;
    transition: transform 140ms ease, background-color 240ms ease;
}

.result-row:hover { transform: translateX(-2px); }

.result-meta {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    min-width: 56px;
}

.result-rank {
    font-size: 0.78rem;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
}

.result-en {
    font-size: 0.78rem;
    color: var(--ink-muted);
    direction: ltr;
    opacity: 0.75;
}

.result-cat {
    font-size: clamp(1.15rem, 2.2vw, 1.45rem);
    font-weight: 800;
    color: var(--ink);
    text-align: right;
}

.result-pct {
    font-size: clamp(1.4rem, 3vw, 2rem);
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    color: var(--ink);
    direction: ltr;
}

.result-rank-1 { background: var(--terracotta); color: #F5EFE6; }
.result-rank-1 .result-rank,
.result-rank-1 .result-cat,
.result-rank-1 .result-pct { color: #F5EFE6; }
.result-rank-1 .result-en { color: rgba(245, 239, 230, 0.75); }

.result-rank-2 {
    background: var(--olive-tint);
    border: 1px solid rgba(95, 104, 69, 0.22);
}
body.dark .result-rank-2 { border-color: rgba(138, 148, 104, 0.30); }
.result-rank-2 .result-rank { color: var(--olive); }

.result-rank-3 {
    background: var(--paper);
    border: 1px solid var(--border);
    opacity: 0.92;
}
.result-rank-3 .result-cat { color: var(--ink-muted); font-weight: 700; }
.result-rank-3 .result-pct { color: var(--ink-muted); }
.result-rank-3 .result-rank { color: var(--ink-muted); }

/* General fallback: when model can't identify a specific aspect */
.general-fallback {
    direction: rtl;
    padding: 28px 26px;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    gap: 22px;
}

.general-headline {
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.general-headline strong {
    font-size: 1.15rem;
    font-weight: 800;
    color: var(--ink);
}
.general-headline span {
    font-size: 0.88rem;
    color: var(--ink-muted);
    direction: ltr;
    text-align: right;
}

.general-hint {
    font-size: 0.95rem;
    line-height: 1.7;
    color: var(--ink);
    padding: 14px 18px;
    background: var(--surface);
    border-right: 3px solid var(--terracotta);
    border-radius: 8px;
}
.general-hint em {
    color: var(--terracotta-deep);
    font-style: normal;
    font-weight: 700;
}
body.dark .general-hint em { color: var(--terracotta); }

.general-rail {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-top: 8px;
    border-top: 1px solid var(--border);
}
.general-rail-label {
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-muted);
    direction: ltr;
    text-align: right;
    padding-right: 4px;
}

.result-rank-fallback {
    background: transparent;
    border: 1px solid var(--border);
}
.result-rank-fallback .result-rank {
    color: var(--terracotta-deep);
    font-size: 1rem;
}
body.dark .result-rank-fallback .result-rank { color: var(--terracotta); }
.result-rank-fallback .result-cat {
    color: var(--ink);
    font-weight: 700;
}
.result-rank-fallback .result-pct {
    color: var(--ink-muted);
    font-size: 1.1rem;
}

/* ---- "How I read this" interpretability panel ---- */

.understanding {
    margin-top: 22px;
    padding: 22px 24px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    direction: rtl;
}

body.dark .understanding { background: var(--paper); }

.understanding-head {
    display: flex;
    flex-direction: column-reverse;
    gap: 4px;
    margin-bottom: 16px;
    align-items: flex-start;
}

.understanding-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-muted);
    direction: ltr;
}

.understanding-head strong {
    font-size: 1.05rem;
    font-weight: 800;
    color: var(--ink);
}

.understanding-text {
    font-size: 1.1rem;
    line-height: 1.85;
    color: var(--ink);
    padding: 16px 18px;
    background: var(--paper);
    border-radius: 10px;
    margin-bottom: 16px;
    word-spacing: 0.05em;
}

body.dark .understanding-text { background: var(--paper-deep); }

.understanding-note {
    margin: 0;
    padding: 12px 14px;
    font-size: 0.92rem;
    color: var(--ink-muted);
    line-height: 1.65;
}

.understanding-note em {
    color: var(--ink-muted);
    font-style: italic;
    direction: ltr;
}

/* Aspect highlight marks inline in the text */
.aspect-mark {
    background: transparent;
    color: inherit;
    padding: 1px 3px;
    border-radius: 3px;
    border-bottom: 2px solid;
    margin: 0 1px;
    cursor: help;
}

.aspect-food         { border-color: #C75D3D; background: rgba(199, 93, 61, 0.08); }
.aspect-delivery     { border-color: #5F6845; background: rgba(95, 104, 69, 0.10); }
.aspect-service      { border-color: #5B6E7E; background: rgba(91, 110, 126, 0.10); }
.aspect-wait         { border-color: #C49443; background: rgba(196, 148, 67, 0.10); }
.aspect-clean        { border-color: #4F8378; background: rgba(79, 131, 120, 0.10); }
.aspect-price        { border-color: #7A5C84; background: rgba(122, 92, 132, 0.10); }
.aspect-accuracy     { border-color: #C77556; background: rgba(199, 117, 86, 0.10); }

body.dark .aspect-food         { background: rgba(216, 120, 82, 0.18); border-color: #D87852; }
body.dark .aspect-delivery     { background: rgba(138, 148, 104, 0.18); border-color: #8A9468; }
body.dark .aspect-service      { background: rgba(140, 162, 178, 0.18); border-color: #8CA2B2; }
body.dark .aspect-wait         { background: rgba(220, 175, 92, 0.18); border-color: #DCAF5C; }
body.dark .aspect-clean        { background: rgba(122, 175, 165, 0.18); border-color: #7AAFA5; }
body.dark .aspect-price        { background: rgba(168, 130, 184, 0.18); border-color: #A882B8; }
body.dark .aspect-accuracy     { background: rgba(220, 145, 115, 0.18); border-color: #DC9173; }

/* Detected-aspects chip rail below the highlighted text */
.aspect-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.aspect-chip {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: var(--paper);
    direction: rtl;
    flex: 1 1 220px;
    min-width: 0;
}

body.dark .aspect-chip { background: var(--paper-deep); }

.aspect-chip-label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
}

.aspect-chip-cat {
    font-size: 0.92rem;
    font-weight: 800;
    color: var(--ink);
}

.aspect-chip-en {
    font-size: 0.72rem;
    color: var(--ink-muted);
    direction: ltr;
    text-align: right;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.aspect-chip-evidence {
    font-size: 0.88rem;
    color: var(--ink-muted);
    line-height: 1.5;
    direction: rtl;
    text-align: right;
}

/* Color the chip's left border to match its aspect */
.aspect-chip.aspect-food     .aspect-chip-cat { color: #C75D3D; }
.aspect-chip.aspect-delivery .aspect-chip-cat { color: #5F6845; }
.aspect-chip.aspect-service  .aspect-chip-cat { color: #5B6E7E; }
.aspect-chip.aspect-wait     .aspect-chip-cat { color: #C49443; }
.aspect-chip.aspect-clean    .aspect-chip-cat { color: #4F8378; }
.aspect-chip.aspect-price    .aspect-chip-cat { color: #7A5C84; }
.aspect-chip.aspect-accuracy .aspect-chip-cat { color: #C77556; }

body.dark .aspect-chip.aspect-food     .aspect-chip-cat { color: #D87852; }
body.dark .aspect-chip.aspect-delivery .aspect-chip-cat { color: #8A9468; }
body.dark .aspect-chip.aspect-service  .aspect-chip-cat { color: #8CA2B2; }
body.dark .aspect-chip.aspect-wait     .aspect-chip-cat { color: #DCAF5C; }
body.dark .aspect-chip.aspect-clean    .aspect-chip-cat { color: #7AAFA5; }
body.dark .aspect-chip.aspect-price    .aspect-chip-cat { color: #A882B8; }
body.dark .aspect-chip.aspect-accuracy .aspect-chip-cat { color: #DC9173; }

.understanding-empty .understanding-text {
    color: var(--ink-muted);
    opacity: 0.85;
}

/* Multi-aspect: rank 1 and 2 share equal visual weight */
.result-rank-co1 { background: var(--terracotta); color: #F5EFE6; }
.result-rank-co1 .result-rank,
.result-rank-co1 .result-cat,
.result-rank-co1 .result-pct { color: #F5EFE6; }
.result-rank-co1 .result-en { color: rgba(245, 239, 230, 0.75); }
.result-rank-co1:nth-of-type(2) { background: var(--terracotta-deep); }

/* New aspect-driven rail (aspect-extraction is the source of truth) — */
/* the "top" badge is darker so the user can see which the model considers */
/* most prominent. All other badges share the standard terracotta. */
.result-rank-top { background: var(--terracotta-deep); color: #F5EFE6; }
.result-rank-top .result-rank,
.result-rank-top .result-cat,
.result-rank-top .result-pct { color: #F5EFE6; }
.result-rank-top .result-en { color: rgba(245, 239, 230, 0.78); }

.result-rank-other { background: var(--terracotta); color: #F5EFE6; }
.result-rank-other .result-rank,
.result-rank-other .result-cat,
.result-rank-other .result-pct { color: #F5EFE6; }
.result-rank-other .result-en { color: rgba(245, 239, 230, 0.72); }

.multi-aspect-badge {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 16px 20px;
    margin-bottom: 14px;
    background: var(--olive-tint);
    border: 1px solid rgba(95, 104, 69, 0.30);
    border-radius: 12px;
    direction: rtl;
}

body.dark .multi-aspect-badge { border-color: rgba(138, 148, 104, 0.40); }

.multi-aspect-mark {
    font-size: 1.4rem;
    color: var(--olive);
    line-height: 1;
    font-weight: 800;
    flex-shrink: 0;
}

.multi-aspect-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.multi-aspect-text strong {
    font-size: 0.98rem;
    font-weight: 700;
    color: var(--ink);
}

.multi-aspect-text span {
    font-size: 0.82rem;
    color: var(--ink-muted);
    direction: ltr;
    text-align: right;
}

/* Loading pulse during prediction */
.gradio-container .pending #result_panel { animation: pulse 1.6s ease-in-out infinite; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.55; }
}

/* ---- Examples ---- */

#examples_panel {
    margin-top: clamp(36px, 5vw, 48px);
    padding-top: clamp(28px, 4vw, 36px);
    border-top: 1px solid var(--border);
}

#examples_panel .gr-examples,
#examples_panel .gr-form,
#examples_panel .gr-block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#examples_panel .gr-examples > .label,
#examples_panel .label { display: none !important; }

#examples_panel button {
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 999px !important;
    padding: 10px 18px !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    direction: rtl;
    text-align: right;
    transition: all 120ms ease;
    cursor: pointer;
}

#examples_panel button:hover {
    background: var(--ink) !important;
    color: var(--surface) !important;
    border-color: var(--ink) !important;
}

/* ---- Performance section (charts) ---- */

.perf-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: clamp(40px, 5vw, 56px);
}

.perf-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: clamp(24px, 3vw, 32px);
    box-shadow: 0 1px 2px rgba(45, 33, 26, 0.04);
    transition: background-color 240ms ease, border-color 240ms ease;
}

.perf-card-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    margin-bottom: 20px;
    direction: rtl;
}

.perf-card-title {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--ink);
    direction: rtl;
    text-align: right;
}

.perf-card-sub {
    font-size: 0.88rem;
    font-weight: 600;
    color: var(--ink);
    opacity: 0.88;
    direction: ltr;
}
body.dark .perf-card-sub { opacity: 0.92; }

.perf-card-chart {
    width: 100%;
    height: auto;
    display: block;
}

img.perf-card-chart {
    max-width: 100%;
}

/* Light/dark chart swap */
.chart-dark { display: none; }
body.dark .chart-light { display: none; }
body.dark .chart-dark { display: block; }

.perf-card-caption {
    font-size: 0.95rem;
    color: var(--ink);
    opacity: 0.88;
    line-height: 1.75;
    margin: 16px 0 0;
    direction: rtl;
    text-align: right;
}
body.dark .perf-card-caption { opacity: 0.94; }

/* ---- Inline About section ---- */

.about-prose {
    direction: rtl;
    margin: 0 0 56px auto;
    max-width: 60ch;
}

.about-prose p {
    font-size: clamp(1.05rem, 1.5vw, 1.2rem);
    font-weight: 400;
    color: var(--ink);
    line-height: 1.85;
    margin: 0;
    direction: rtl;
    text-align: right;
}

.about-prose p strong {
    color: var(--ink);
    font-weight: 700;
}

.categories-rail {
    direction: rtl;
    border-top: 1px solid var(--border);
    padding-top: 28px;
}

.categories-rail .rail-label {
    display: block;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin: 0 0 18px;
    direction: ltr;
    text-align: right;
}

.categories-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    direction: rtl;
}

.category-cell {
    display: inline-flex;
    align-items: baseline;
    gap: 8px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 999px;
    padding: 8px 16px;
    direction: rtl;
    transition: border-color 240ms ease, background-color 240ms ease;
}

.category-cell:hover {
    border-color: var(--ink-muted);
    background: var(--surface);
}

.category-cell .cat-ar {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--ink);
}

.category-cell .cat-en {
    font-size: 0.75rem;
    color: var(--ink-muted);
    direction: ltr;
}

/* ---- Footer ---- */

#footer {
    width: 100%;
    background: var(--footer-bg);
    color: var(--footer-text);
    padding: clamp(36px, 5vw, 48px) clamp(24px, 5vw, 56px);
    transition: background-color 240ms ease;
}

#footer .footer-inner {
    max-width: 1080px;
    margin: 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    font-size: 0.92rem;
}

#footer .links { display: flex; gap: 24px; flex-wrap: wrap; }

#footer a {
    color: #F5EFE6 !important;
    text-decoration: none;
    font-weight: 700;
    border-bottom: 1px solid transparent;
    transition: border-color 140ms ease, color 140ms ease;
}

#footer a:hover {
    color: #F5C16C !important;
    border-bottom-color: var(--terracotta);
}

footer { display: none !important; }

/* ---- Reduced motion ---- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ---- Mobile ---- */
@media (max-width: 720px) {
    #stats .stats-grid { grid-template-columns: repeat(2, 1fr); }
    #stats .stat:nth-child(2) { border-right: none; }
    #stats .stat:nth-child(1),
    #stats .stat:nth-child(2) { border-bottom: 1px solid var(--border); }

    .result-row {
        grid-template-columns: auto 1fr;
        gap: 12px 16px;
    }
    .result-pct { grid-column: 2; text-align: left; margin-top: 4px; }
    #actions button { flex: 1; }
}

@media (max-width: 480px) {
    #stats .stats-grid { grid-template-columns: 1fr; }
    #stats .stat { border-right: none; border-bottom: 1px solid var(--border); }
    #stats .stat:last-child { border-bottom: none; }
    #theme-toggle { width: 38px; height: 38px; }
}
"""


# ---- Build the about section as raw HTML (no accordion) --------------------

ABOUT_HTML = """
<div class="about-prose">
  <p>
    نموذج <strong>CAMeLBERT-mix</strong> تدرّب على ٩٥ ألف شكوى عربية حقيقية،
    معظمها من تطبيقات التوصيل السعودية. مخصّص للّهجة السعودية والخليجية.
    على مجموعة اختبار مستقلّة من ١٣٬٩٨٦ مراجعة، دقّة الـ<em>ensemble</em> الكامل
    (٤ نماذج) <strong>٩٥٫٠٥٪</strong> بفاصل ثقة ٩٥٪ بين ٩٤٫٧٠٪ و ٩٥٫٤١٪.
    النموذج المنشور هنا هو أفضل نموذج فردي من هذا الـ<em>ensemble</em>،
    وكل الفئات الثماني فوق ٨٠٪ F1.
  </p>
</div>
"""


CATEGORIES_GRID_HTML = '<div class="categories-grid">' + "".join(
    f'<div class="category-cell">'
    f'  <span class="cat-ar">{ar}</span>'
    f'  <span class="cat-en">{CATEGORIES_EN[ar]}</span>'
    f'</div>'
    for ar in CATEGORIES
) + "</div>"


# ---- Build UI ---------------------------------------------------------------

THEME_TOGGLE_HTML = """
<div id="theme-toggle-wrap">
  <button id="theme-toggle" type="button" aria-label="Toggle dark mode">
    <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
    <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>
    </svg>
  </button>
</div>
"""


with gr.Blocks(
    title="تصنيف شكاوى المطاعم العربية",
    analytics_enabled=False,
    theme=gr.themes.Base(),
    css=CSS,
    head=HEAD,
) as demo:
    # Theme toggle (fixed top-right)
    gr.HTML(THEME_TOGGLE_HTML)

    # Hero with embedded SVG dusk scene + stats strip
    gr.HTML(
        f"""
        <section id="hero">
          <div class="hero-bg">{HERO_SVG}</div>
          <div class="hero-content">
            <div class="eyebrow">Arabic Restaurant Complaints Classifier</div>
            <h1>تصنيف شكاوى المطاعم العربية</h1>
            <p class="lede">
              نموذج عربي تدرّب على ٩٥ ألف شكوى حقيقية من تطبيقات التوصيل السعودية،
              يصنّف أي شكوى إلى واحدة من ٨ فئات.
            </p>
          </div>
        </section>

        """
    )

    # Classify section
    with gr.Column(elem_id="section-classify", elem_classes=["section", "section--surface"]):
        gr.HTML(
            """
            <div class="section-inner">
              <div class="section-eyebrow">Classify</div>
              <h2 class="section-title">جرّب النموذج بشكواك</h2>
              <p class="section-lede">
                اكتب شكواك بالعربية، وستظهر أعلى ثلاث فئات مع نسبة الثقة لكل واحدة.
              </p>
            </div>
            """
        )

        with gr.Column(elem_id="input_panel"):
            input_box = gr.Textbox(
                lines=4,
                show_label=False,
                placeholder="مثال: الاكل بايخ ومالح والطبخ مو متقن",
                rtl=True,
                container=False,
            )
            with gr.Row(elem_id="actions"):
                submit_btn = gr.Button("صنّف الشكوى", variant="primary", scale=2)
                clear_btn = gr.Button("مسح", variant="secondary", scale=1)

        with gr.Column(elem_id="result_panel"):
            output_html = gr.HTML(value=EMPTY_RESULT)

        with gr.Column(elem_id="examples_panel"):
            gr.Examples(
                examples=EXAMPLES,
                inputs=input_box,
                outputs=output_html,
                fn=predict,
                cache_examples=False,
                label=None,
            )

    # Performance section with light + dark chart variants
    gr.HTML(
        f"""
        <section class="section section--paper">
          <div class="section-inner">
            <div class="section-eyebrow">Performance</div>
            <h2 class="section-title">كل الفئات فوق ٨٠٪</h2>
            <p class="section-lede">
              النموذج يتعامل مع الفئات الكبيرة والصغيرة بنفس الجودة.
              هذه القياسات على مجموعة اختبار مستقلّة، لم تُستخدم أثناء التدريب.
            </p>

            <div class="perf-grid">
              <div class="perf-card">
                <div class="perf-card-head">
                  <div class="perf-card-title">F1 لكل فئة</div>
                  <div class="perf-card-sub">test set, 13,986 reviews</div>
                </div>
                <img class="perf-card-chart chart-light" src="{CHART_F1_LIGHT_URI}" alt="Per-class F1 bar chart" loading="lazy">
                <img class="perf-card-chart chart-dark" src="{CHART_F1_DARK_URI}" alt="Per-class F1 bar chart" loading="lazy">
                <p class="perf-card-caption">
                  أعلى فئة (جودة الطعام) ٩٦٫٢٪، أدنى فئة (عامة) ٨٤٫٩٪. الفارق ١١٫٣ نقطة فقط.
                </p>
              </div>

              <div class="perf-card">
                <div class="perf-card-head">
                  <div class="perf-card-title">رحلة النموذج</div>
                  <div class="perf-card-sub">accuracy across iterations</div>
                </div>
                <img class="perf-card-chart chart-light" src="{CHART_BL_LIGHT_URI}" alt="Accuracy progression across iterations" loading="lazy">
                <img class="perf-card-chart chart-dark" src="{CHART_BL_DARK_URI}" alt="Accuracy progression across iterations" loading="lazy">
                <p class="perf-card-caption">
                  رحلتنا من TF-IDF إلى BERT، ثم إلى <em>ensemble</em> من ٤ نماذج.
                  وحذفنا فئة "الجو والمكان" بعد تدقيق كشف أن ٩٩٪ من تصنيفاتها كانت خاطئة.
                </p>
              </div>
            </div>
          </div>
        </section>
        """
    )

    # About + Categories — combined into one quiet section, no template rhythm
    gr.HTML(
        f"""
        <section class="section section--surface" id="about-section">
          <div class="section-inner">
            <h2 class="section-title">عن النموذج</h2>
            {ABOUT_HTML}
          </div>
        </section>
        """
    )

    # Footer
    gr.HTML(
        f"""
        <footer id="footer">
          <div class="footer-inner">
            <span class="made-by">Made by the NLP team at the AI Club of KSU</span>
            <span class="links">
              <a href="{GITHUB_URL}" target="_blank" rel="noopener">Source on GitHub</a>
              <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" rel="noopener">Model on HF Hub</a>
            </span>
          </div>
        </footer>
        """
    )

    submit_btn.click(predict, inputs=input_box, outputs=output_html)
    input_box.submit(predict, inputs=input_box, outputs=output_html)
    clear_btn.click(lambda: ("", EMPTY_RESULT), outputs=[input_box, output_html])


demo.launch(server_name="0.0.0.0", server_port=7860)
