"""Saudi/Gulf ambience keyword vocabulary, organized by subtype.

All strings are stored in their natural surface form. The filter script
normalizes them through `clean()` (same normalization as the production
inference pipeline) before matching, so:

  * أ إ آ ٱ → ا
  * ى → ي
  * ة → ه
  * Persian/Gulf chars (پ چ گ ک ی) → Arabic equivalents
  * tashkeel removed
  * lowercased

This means you can write keywords with whatever orthography is most
natural and matching will still work against differently-orthographed text.

The 12 subtypes mirror the schema in BOUNDARIES.md:

  temperature_ac, seating_comfort, space_crowding, noise_music, lighting,
  smell, decor_furniture, parking, bathroom_facilities, outdoor_view,
  privacy, general_place_vibe
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Direct ambience keywords — by subtype
# ---------------------------------------------------------------------------
# Single-word and multi-word phrases that, when present in context, are
# strong evidence for the ambience class. Multi-word phrases give higher
# precision and are weighted more in the filter script.

AMBIENCE_BY_SUBTYPE: dict[str, list[str]] = {
    "general_place_vibe": [
        "المحل", "المطعم", "الفرع", "المكان", "اللوكيشن",
        "الجو", "الأجواء", "اجواء", "الفايب",
        "رايق", "كئيب", "يكتم", "يفتح النفس", "ما يفتح النفس",
        "شرح", "منفس", "مقفل", "قديم", "متهالك", "مكروف", "مهلوك",
        # Place + condition multi-word phrases — high-precision ambience cues
        # that combine an otherwise-risky place token with a condition.
        # Without these, sentences like "المكان بارد والمكيف قوي" demote to
        # review because both individual words are risky-ambiguous.
        "المكان بارد", "المكان حار", "المكان قديم", "المكان كئيب",
        "المطعم بارد", "المطعم حار", "المطعم قديم",
        "الفرع بارد", "الفرع حار", "الفرع قديم",
        "المحل بارد", "المحل حار", "المحل قديم",
    ],
    "seating_comfort": [
        "الجلسه", "الجلسات", "قعدة", "القعده", "القعدات",
        "الكراسي", "الكرسي", "الكنب",
        "الطاولات", "الطاوله", "طاوله",
        "جلسات ارضيه", "جلسات عوائل", "جلسات شباب",
        "البارتشن", "بارتشن", "ساتر", "فاصل", "قزاز", "حواجز",
        "مزحومين", "متلاصقين", "لاصقين", "مصفوطين",
        "مافي مجال", "ما فيه مجال",
        "الكنب وصخ", "الكراسي مهلوكه", "طاولة تهتز", "طاوله تهتز",
    ],
    "space_crowding": [
        "زحمه", "زحمة", "زحام", "زحمة موت", "زحمة مره",
        "مزدحم", "كتمه", "كتمة", "ضيق", "ضيقه", "مخنوق", "مكتوم",
        "مافي مساحه", "ما فيه مساحة",
        "واسع", "وسع", "رحب", "فاضي", "هادي", "هادي مره",
        # NB: 'رايق' (calm) was here originally — moved to general_place_vibe
        # exclusively, since duplicates make subtype tagging non-deterministic.
    ],
    "temperature_ac": [
        "المكيف", "المكيفات", "التكييف", "التهويه", "التهوية",
        "حر", "حار", "حراره", "حرارة", "برد", "بارد",
        "ثلج", "يصقع",
        "مكيف خربان", "المكيف خربان", "المكيف ما يبرد",
        "التكييف سيء", "التكييف ما يبرد",
        "رطوبه", "رطوبة", "خانق", "مافي هوا", "ما فيه هوا",
        # AC condition multi-word phrases
        "المكيف قوي", "المكيف بارد", "المكيف ضعيف",
    ],
    "noise_music": [
        "ازعاج", "إزعاج", "مزعج", "صجه", "صجة", "لجه", "لجة",
        "دوشتنا", "صوت عالي", "اصوات", "أصوات",
        "موسيقى", "الاغاني", "الأغاني",
        "الصوت عالي", "الدي جي", "دي جي",
        "بزارين", "اطفال", "أطفال", "صراخ", "زعاق",
        "ما نقدر نسولف", "ما نسمع بعض",
    ],
    "lighting": [
        "الاضاءه", "الإضاءة", "الإنارة", "الاناره",
        "النور", "النور قوي", "النور ضعيف",
        "ظلام", "مظلم", "كشافات", "اللمبات",
        "اضاءة مزعجة", "اضاءة خافته", "اضاءة قوية",
    ],
    "smell": [
        "ريحه", "ريحة", "رائحة",
        "ريحة زيت", "ريحة قلي", "ريحة شاورما",
        "ريحة دخان", "دخان", "شيشه", "شيشة", "معسل",
        "ريحة حمام", "ريحة مجاري", "مجاري",
        "ريحة رطوبه", "ريحة رطوبة",
        "ريحة المكان", "الريحة تلصق",
        "مسكت في الملابس", "طلعت ريحتي",
    ],
    "decor_furniture": [
        "الديكور", "الاثاث", "الأثاث", "الفرش",
        "الجدران", "البويه", "البوية",
        "الأرضية", "الارضيه", "السيراميك", "الرخام",
        "الكوشن", "المخدات",
        "الاثاث قديم", "الديكور قديم", "الشكل قديم",
    ],
    "parking": [
        "المواقف", "مواقف", "موقف",
        "باركنق", "باركنج", "parking",
        "مافي مواقف", "ما فيه مواقف",
        "المواقف بعيده", "المواقف بعيدة", "وقفت بعيد",
        "المدخل", "الدخول", "الطلعه", "الطلعة",
        "الموقع صعب",
    ],
    "bathroom_facilities": [
        # Note: per BOUNDARIES.md, bathroom CLEANLINESS goes to النظافة.
        # These are FACILITIES words (broken sink, no soap) only.
        "السيفون", "المغاسل", "المغسلة", "مغسله",
        "الصابون", "المناديل",
        "حمام خربان", "السيفون ما يشتغل",
        "مافي صابون", "ما فيه صابون",
        "باب الحمام ما يقفل",
        "دورات المياه سيئة",
    ],
    "outdoor_view": [
        "جلسات خارجيه", "جلسات خارجية", "الخارجي",
        "التراس", "تراس", "البلكونه", "البلكونة", "السطح",
        "الاطلاله", "الإطلالة", "اطلالة",
        "المنظر", "الواجهة", "الواجهة زجاج",
        "جلسة برا", "برا", "بره",
    ],
    "privacy": [
        "خصوصيه", "خصوصية", "مافي خصوصيه", "ما فيه خصوصية",
        "عوائل", "عوايل", "افراد", "أفراد",
        "قسم العوائل", "جلسات العوائل",
        "مكشوف", "كل الناس تشوفك",
        "لاصقين فينا", "جنبنا ناس",
    ],
}


# Flat list of all multi-word ambience phrases (≥2 words). High-precision —
# matching one of these is strong evidence for ambience.
AMBIENCE_HIGH_PRECISION_PHRASES: list[str] = sorted(
    {p for phrases in AMBIENCE_BY_SUBTYPE.values() for p in phrases if " " in p},
    key=len,
    reverse=True,
)


# Flat list of all single-word ambience tokens. Lower precision because
# many of these can appear in non-ambience contexts (المكان, المطعم, الفرع
# all appear in food/service complaints too).
AMBIENCE_SINGLE_TOKENS: list[str] = sorted(
    {p for phrases in AMBIENCE_BY_SUBTYPE.values() for p in phrases if " " not in p}
)


# ---------------------------------------------------------------------------
# Risky keywords — ambiguous between ambience and other classes
# ---------------------------------------------------------------------------
# These words alone are NOT enough evidence for ambience. They need
# co-occurring context (a place-word from AMBIENCE_SINGLE_TOKENS, or a
# subtype-specific ambience word). The filter script demotes rows that
# only match these.

RISKY_AMBIGUOUS: set[str] = {
    "حار", "بارد", "حاره", "بارده",  # food temperature vs room temperature
    "قديم", "قديمه",                   # food freshness vs decor age
    "ريحه", "ريحة", "رائحة",          # food smell vs room smell
    "زيت",                              # food grease vs floor/air grease
    "وسخ", "وسخه", "متسخ", "متسخه",   # cleanliness vs furniture condition
    "عالي", "عاليه",                   # voice/music volume vs price
    "برا", "بره",                       # outdoor seating vs literal "outside"
    # Generic place tokens — appear in every class, never strong anchors alone:
    "المطعم", "الفرع", "المحل", "المكان", "اللوكيشن",
    # Furniture/surface tokens — ambience subtype words but co-occur with
    # cleanliness contamination ("الطاولة وسخه فيها بقع طعام") and with
    # ambience-as-fragment lines ("طلبت ان غير الطاولة"). Need context.
    "الطاوله", "الطاولة", "الارضيه", "الأرضية",
}


# ---------------------------------------------------------------------------
# Hard-negative cues — strong evidence the row is NOT ambience
# ---------------------------------------------------------------------------
# When any of these appear, the row goes to hard_negative_candidates.csv
# even if it also has ambience cues. We're being conservative on purpose.

NEGATIVE_FOOD: set[str] = {
    "الاكل", "الطعام", "الوجبه", "الوجبة", "البرجر", "الشاورما",
    "البيتزا", "السندويش", "الساندويتش", "الدجاج", "اللحم", "اللحمه",
    "الرز", "الارز", "السلطه", "السلطة", "البطاطس", "العيش", "الخبز",
    "الطعم", "طعمه", "طعمها", "النكهه", "النكهة",
    "بايخ", "مالح", "حلو زياده",
    "محروق", "ني", "نيء", "مطبوخ زياده",
    "بارد الاكل", "الاكل بارد",
    "الاكل وسخ",  # food contamination signal — overlaps with cleanliness
}

NEGATIVE_SERVICE: set[str] = {
    "الموظف", "الموظفين", "الموظفه", "الكاشير", "النادل",
    "الويتر", "المضيف", "المضيفه",
    "اسلوبه", "اسلوبها", "اسلوب",
    "غير محترم", "ما رحب", "ما رحبت", "متجاهل", "متجاهله",
    "ما رد", "ما ردت", "ما اهتم", "ما اهتمت",
}

NEGATIVE_DELIVERY: set[str] = {
    "المندوب", "المندوبين", "السائق", "الديلفري", "التوصيل",
    "الطلب وصل", "ما وصل الطلب", "ضاع الطلب",
    "تطبيق", "هنقرستيشن", "جاهز", "مرسول", "طلبات",
}

NEGATIVE_PRICE: set[str] = {
    "السعر", "الاسعار", "الفاتوره", "الفاتورة",
    "غالي", "غاليه", "غالية", "مكلف", "مكلفه",
    "ما يستاهل", "مبالغ فيه", "مبالغ فيها",
}

NEGATIVE_ORDER: set[str] = {
    "نسوا", "نسي", "ناقص", "ناقصه", "ناقصة",
    "غلط", "خطا", "خطأ", "غير اللي طلبته",
    "بدون بصل", "بدون مايونيز", "بدون صوص",
    "ما طلبت", "ما طلبنا",
}

NEGATIVE_HYGIENE: set[str] = {
    # Food/dish/cutlery hygiene — goes to النظافة, not ambience
    "صحن وسخ", "صحون وسخه", "ملعقه وسخه",
    "ذباب", "صراصير", "حشره", "حشرة",
    "الحمام وصخ", "الحمام قذر", "الحمام مقرف",
    # Surface contamination cues — when paired with table/floor tokens,
    # signal cleanliness rather than furniture-condition. Filter routes
    # these to hard_negative; the labeler decides via BOUNDARIES.md.
    "بقع طعام", "بقع",
    "ما تنظفت", "ما تنظف", "ما يتم تنظيف",
    "اثر شفايف",
    "دبقه", "دبق",
}

NEGATIVE_WAIT: set[str] = {
    "انتظرت", "انتظرنا", "ساعه كامله", "ساعتين",
    "تاخر الطلب", "وقت طويل", "بطء", "بطيء",
}

ALL_HARD_NEGATIVES: set[str] = (
    NEGATIVE_FOOD | NEGATIVE_SERVICE | NEGATIVE_DELIVERY |
    NEGATIVE_PRICE | NEGATIVE_ORDER | NEGATIVE_HYGIENE | NEGATIVE_WAIT
)


# ---------------------------------------------------------------------------
# Restaurant domain anchors — used by the v5 inference wrapper for OOD detection.
# ---------------------------------------------------------------------------
# If an input contains NONE of these anchors AND no ambience keyword either,
# it's almost certainly out-of-domain (not a restaurant complaint).
# The wrapper abstains in that case to avoid forcing a category prediction.
# Combines: generic place words + the 8 production hard-negative sets +
# the ambience single tokens. Effectively "this text mentions something
# restaurant-related in any of the 9 schema categories".

RESTAURANT_DOMAIN_ANCHORS: set[str] = (
    NEGATIVE_FOOD | NEGATIVE_SERVICE | NEGATIVE_DELIVERY |
    NEGATIVE_PRICE | NEGATIVE_ORDER | NEGATIVE_HYGIENE | NEGATIVE_WAIT |
    set(AMBIENCE_SINGLE_TOKENS) |
    # Multi-word ambience phrases that don't fit the single-tokens set
    {phrase for phrase in AMBIENCE_HIGH_PRECISION_PHRASES} |
    # Add common restaurant-context words not in any of the above
    {
        "مطعم", "كافيه", "كوفي", "مقهى", "بوفيه",
        "وجبه", "وجبة", "افطار", "غداء", "عشاء",
        "حجز", "حجزت", "حجزنا",
        "زرت", "زرنا", "ذهبت", "رحت", "روحت",
        "تجربه", "تجربة",
    }
)


# ---------------------------------------------------------------------------
# Subtype keyword index — for tagging ambience candidates with their subtype
# ---------------------------------------------------------------------------

def subtype_for_phrase(phrase: str) -> str | None:
    """Return which ambience subtype this phrase belongs to, or None."""
    for subtype, phrases in AMBIENCE_BY_SUBTYPE.items():
        if phrase in phrases:
            return subtype
    return None


# Pre-normalize the anchor set at module load so matching is fast and
# orthography-tolerant. The vocab includes mixed orthography (with/without
# tashkeel, ة vs ه, etc.); cleaned input always has the normalized form.
def _normalize_anchor(s: str) -> str:
    """Same normalization as src/utils/arabic_normalization.py:clean(),
    inlined here to avoid a circular import (utils → config OK; config → utils
    creates a cycle if config is later imported earlier in the chain)."""
    import re
    if not s:
        return ""
    s = re.sub(r"[ً-ٟ]", "", s)
    s = s.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "پ": "ب", "چ": "ج", "گ": "ك", "ک": "ك", "ی": "ي",
    }))
    s = re.sub(r"[^؀-ۿa-zA-Z0-9٠-٩\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


_NORMALIZED_ANCHORS_SINGLE: set[str] = set()
_NORMALIZED_ANCHORS_MULTI: list[str] = []
for _anchor in RESTAURANT_DOMAIN_ANCHORS:
    _norm = _normalize_anchor(_anchor)
    if not _norm:
        continue
    if " " in _norm:
        _NORMALIZED_ANCHORS_MULTI.append(_norm)
    else:
        _NORMALIZED_ANCHORS_SINGLE.add(_norm)
        # Symmetric ال-prefix handling — anchors are stored mixed
        # ('المكيف' with ال, 'موسيقي' without). Add both forms so
        # cleaned input matches regardless of which form the anchor uses.
        if _norm.startswith("ال") and len(_norm) > 3:
            _NORMALIZED_ANCHORS_SINGLE.add(_norm[2:])
# Sort multi by length desc — longer phrases first (better precision)
_NORMALIZED_ANCHORS_MULTI.sort(key=len, reverse=True)


def has_restaurant_domain_anchor(cleaned_text: str) -> bool:
    """True if the cleaned text contains at least one restaurant-domain word.

    Used by the v5 inference wrapper to detect out-of-domain inputs.
    Conservative: a single anchor word is enough to PASS this gate.

    Matching:
    - Multi-word anchors → substring check (already cleaned both sides)
    - Single-word anchors → exact word OR ال-prefix-stripped word match
      (Arabic words frequently appear with the definite article ال
      attached; e.g. cleaned input has 'الموسيقي' but anchor is 'موسيقي').
    """
    if not cleaned_text:
        return False
    # Multi-word substring check
    for anchor in _NORMALIZED_ANCHORS_MULTI:
        if anchor in cleaned_text:
            return True
    # Single-word check with ال-prefix tolerance
    words = cleaned_text.split()
    for word in words:
        if word in _NORMALIZED_ANCHORS_SINGLE:
            return True
        # Strip ال prefix and try again (handles ambience vocab like 'موسيقي'
        # matching cleaned input 'الموسيقي')
        if word.startswith("ال") and len(word) > 3 and word[2:] in _NORMALIZED_ANCHORS_SINGLE:
            return True
    return False
