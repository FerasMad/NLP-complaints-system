"""HuggingFace Spaces entry point for the Arabic Restaurant Complaints Classifier.

Visual identity is warm, hospitable, Saudi-rooted. Multi-section page with a
golden-hour SVG hero, overlapping stats strip, classify, performance charts,
inline about, and footer. Light + dark themes (toggle top-right). Almarai font.
"""
import base64
import os
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
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


# ---- Prediction ------------------------------------------------------------

EMPTY_RESULT = """
<div class="result-empty">
  <div class="result-empty-text">
    <strong>اكتب شكوى أو اضغط على مثال أدناه</strong>
    <span>type a complaint above, or pick an example below</span>
  </div>
</div>
"""


def is_multi_aspect(top: list[tuple[str, float]]) -> bool:
    """Heuristic: top-1 < 70% AND top-2 > 30% means competing signals."""
    if len(top) < 2:
        return False
    return top[0][1] < 0.70 and top[1][1] > 0.30


def render_result(top: list[tuple[str, float]]) -> str:
    if not top:
        return EMPTY_RESULT

    multi = is_multi_aspect(top)

    rows = []
    for rank, (cat, score) in enumerate(top):
        en = CATEGORIES_EN.get(cat, "")
        pct = f"{score * 100:.0f}%"
        # Multi-aspect: rank 1 + 2 both get the primary treatment
        if multi and rank < 2:
            row_class = "result-row result-rank-co1"
        else:
            row_class = f"result-row result-rank-{rank + 1}"
        rows.append(
            f'<div class="{row_class}">'
            f'  <div class="result-meta">'
            f'    <span class="result-rank">#{rank + 1}</span>'
            f'    <span class="result-en">{en}</span>'
            f'  </div>'
            f'  <div class="result-cat">{cat}</div>'
            f'  <div class="result-pct">{pct}</div>'
            f'</div>'
        )

    badge = ""
    if multi:
        badge = (
            '<div class="multi-aspect-badge">'
            '  <span class="multi-aspect-mark">⊕</span>'
            '  <span class="multi-aspect-text">'
            '    <strong>تشمل الشكوى أكثر من جانب</strong>'
            '    <span>multi-aspect complaint, both top categories shown together</span>'
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


@torch.no_grad()
def predict(text: str) -> str:
    if not text or len(text.strip()) < 3:
        return render_message(
            "اكتب شكوى أطول من ٣ أحرف",
            "type a longer complaint (at least 3 characters)",
        )
    if not is_arabic_enough(text):
        return render_message(
            "النص ليس بالعربية",
            "the input doesn't appear to be Arabic",
        )

    enc = tokenizer(clean(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().numpy()
    top_idx = probs.argsort()[::-1][:3]
    top = [(ID2LABEL[int(i)], float(probs[i])) for i in top_idx]
    return render_result(top)


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

/* Multi-aspect: rank 1 and 2 share equal visual weight */
.result-rank-co1 { background: var(--terracotta); color: #F5EFE6; }
.result-rank-co1 .result-rank,
.result-rank-co1 .result-cat,
.result-rank-co1 .result-pct { color: #F5EFE6; }
.result-rank-co1 .result-en { color: rgba(245, 239, 230, 0.75); }
.result-rank-co1:nth-of-type(2) { background: var(--terracotta-deep); }

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
    font-size: 0.82rem;
    color: var(--ink-muted);
    direction: ltr;
}

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
    font-size: 0.92rem;
    color: var(--ink-muted);
    line-height: 1.7;
    margin: 16px 0 0;
    direction: rtl;
    text-align: right;
}

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
    نموذج <strong>CAMeLBERT-mix</strong> دقّق على ٩٨ ألف شكوى عربية حقيقية، معظمها من تطبيقات
    التوصيل السعودية. مخصّص للهجة السعودية والخليجية. الدقّة على مجموعة اختبار محتجزة
    من ١٣٬٩٨٦ مراجعة: <strong>٩٥٫٠٥٪</strong>، بفاصل ثقة ٩٥٪ بين ٩٤٫٧٠٪ و ٩٥٫٤١٪.
    جميع الفئات الثمان فوق ٨٠٪ F1.
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
              نموذج عربي مدرّب على ٩٨٬٠٠٠ شكوى حقيقية من تطبيقات التوصيل السعودية،
              يصنّف أي شكوى إلى واحدة من ٨ فئات بدقّة ٩٥٪.
            </p>
          </div>
        </section>

        <div id="headline-wrap">
          <div id="headline">
            <div class="headline-stat">
              <span class="headline-figure">95.05%</span>
              <span class="headline-unit">test accuracy</span>
            </div>
            <p class="headline-note">
              13,986 held-out real reviews. 8 categories, all above 80% F1.
              Single best CAMeLBERT-mix from a 4-model ensemble.
            </p>
          </div>
        </div>
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
              النموذج يتعامل مع الفئات الكبيرة والصغيرة بنفس الجودة. هذه القياسات على
              مجموعة اختبار محتجزة تماماً، لم تُستخدم أثناء التدريب.
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
                  بدأنا بنموذج TF-IDF كأساس، ثم انتقلنا إلى BERT، ثم إلى المجموعة، ثم حذفنا فئة "الجو والمكان"
                  بعد تدقيق كشف أن ٩٩٪ من بياناتها كانت غير دقيقة.
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
            <div class="categories-rail">
              <span class="rail-label">الفئات</span>
              {CATEGORIES_GRID_HTML}
            </div>
          </div>
        </section>
        """
    )

    # Footer
    gr.HTML(
        f"""
        <footer id="footer">
          <div class="footer-inner">
            <span class="made-by">Made by the NLP team at AI Club</span>
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
