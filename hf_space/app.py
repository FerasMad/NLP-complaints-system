"""HuggingFace Spaces entry point for the Arabic Restaurant Complaints Classifier.

Visual identity is warm, hospitable, Saudi-rooted. Multi-section page:
hero scene at golden hour, stats strip, classify, performance, about, footer.
Cream working surface, terracotta accent, deep ink type. Almarai font.
"""
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
HERO_SVG = (HERE / "hero.svg").read_text(encoding="utf-8")
CHART_F1_SVG = (HERE / "charts" / "per_class_f1.svg").read_text(encoding="utf-8")
CHART_BASELINES_SVG = (HERE / "charts" / "vs_baselines.svg").read_text(encoding="utf-8")


# ---- Prediction ------------------------------------------------------------

EMPTY_RESULT = """
<div class="result-empty">
  <div class="result-empty-text">
    <strong>اكتب شكوى وستظهر النتيجة هنا</strong>
    <span>type a complaint to see the prediction</span>
  </div>
</div>
"""


def render_result(top: list[tuple[str, float]]) -> str:
    if not top:
        return EMPTY_RESULT
    rows = []
    for rank, (cat, score) in enumerate(top):
        en = CATEGORIES_EN.get(cat, "")
        pct = f"{score * 100:.0f}%"
        rows.append(
            f'<div class="result-row result-rank-{rank + 1}">'
            f'  <div class="result-meta">'
            f'    <span class="result-rank">#{rank + 1}</span>'
            f'    <span class="result-en">{en}</span>'
            f'  </div>'
            f'  <div class="result-cat">{cat}</div>'
            f'  <div class="result-pct">{pct}</div>'
            f'</div>'
        )
    return f'<div class="result-stack">{"".join(rows)}</div>'


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


# ---- Visual identity --------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Almarai:wght@300;400;700;800&display=swap');

:root {
    --cream: #F5EFE6;
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
}

* { font-family: 'Almarai', system-ui, -apple-system, sans-serif !important; box-sizing: border-box; }

html, body, .gradio-container {
    background: var(--cream) !important;
    color: var(--ink) !important;
    margin: 0 !important;
    padding: 0 !important;
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

/* ---- Hero ---- */

#hero {
    position: relative;
    width: 100%;
    height: clamp(440px, 60vh, 560px);
    overflow: hidden;
    color: var(--cream);
    background: #3A1810;
}

#hero .hero-bg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
}

#hero .hero-bg svg {
    width: 100%;
    height: 100%;
    display: block;
}

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
    color: var(--cream);
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

/* ---- Stats strip overlapping hero ---- */

#stats-wrap {
    width: 100%;
    background: transparent;
}

#stats {
    max-width: 1080px;
    margin: -56px auto 0;
    padding: 0 clamp(24px, 5vw, 56px);
    position: relative;
    z-index: 3;
}

#stats .stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 12px 40px -12px rgba(45, 17, 8, 0.25);
    overflow: hidden;
}

#stats .stat {
    padding: 22px 24px;
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 4px;
    direction: ltr;
}

#stats .stat:last-child { border-right: none; }

#stats .stat-key {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-muted);
}

#stats .stat-val {
    font-size: clamp(1.4rem, 2.8vw, 1.8rem);
    font-weight: 800;
    color: var(--ink);
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}

#stats .stat-sub {
    font-size: 0.78rem;
    color: var(--ink-muted);
    margin-top: 2px;
}

/* ---- Section scaffolding ---- */

.section {
    width: 100%;
    padding: clamp(56px, 8vw, 96px) clamp(24px, 5vw, 56px);
}

.section-inner {
    max-width: 1080px;
    margin: 0 auto;
}

.section--paper { background: var(--paper); }
.section--cream { background: var(--cream); }

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
    margin: 0 0 36px;
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
    background: var(--cream) !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 22px 24px !important;
    font-size: 1.15rem !important;
    line-height: 1.7 !important;
    font-weight: 400 !important;
    transition: border-color 160ms ease, box-shadow 160ms ease;
    resize: vertical;
    min-height: 140px;
    direction: rtl;
    text-align: right;
}

#input_panel textarea::placeholder {
    color: rgba(45, 33, 26, 0.35) !important;
    font-weight: 400;
}

#input_panel textarea:focus {
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 4px var(--terracotta-tint) !important;
    outline: none;
}

#input_panel label > span { display: none !important; }

#actions {
    display: flex;
    gap: 12px;
    margin-top: 18px;
    direction: rtl;
}

button.primary {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    cursor: pointer;
    transition: background 140ms ease, transform 140ms ease;
}

button.primary:hover {
    background: var(--terracotta-deep) !important;
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
    background: var(--cream) !important;
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
    background: var(--cream);
    color: var(--ink-muted);
    direction: rtl;
}
.result-empty-text { display: flex; flex-direction: column; gap: 4px; }
.result-empty-text strong { font-size: 1.05rem; font-weight: 700; color: var(--ink); }
.result-empty-text span { font-size: 0.9rem; color: var(--ink-muted); }

.result-message {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 22px 24px;
    background: var(--cream);
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
    transition: transform 140ms ease;
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
    letter-spacing: 0.04em;
    font-variant-numeric: tabular-nums;
}

.result-en {
    font-size: 0.78rem;
    color: rgba(45, 33, 26, 0.55);
    direction: ltr;
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

.result-rank-1 { background: var(--terracotta); color: var(--cream); }
.result-rank-1 .result-rank,
.result-rank-1 .result-cat,
.result-rank-1 .result-pct { color: var(--cream); }
.result-rank-1 .result-en { color: rgba(245, 239, 230, 0.7); }

.result-rank-2 {
    background: var(--olive-tint);
    border: 1px solid rgba(95, 104, 69, 0.22);
}
.result-rank-2 .result-rank { color: var(--olive); }

.result-rank-3 {
    background: var(--cream);
    border: 1px solid var(--border);
    opacity: 0.92;
}
.result-rank-3 .result-cat { color: var(--ink-muted); font-weight: 700; }
.result-rank-3 .result-pct { color: var(--ink-muted); }
.result-rank-3 .result-rank { color: var(--ink-muted); }

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
    color: var(--cream) !important;
    border-color: var(--ink) !important;
}

/* ---- Performance section (charts) ---- */

.perf-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: clamp(40px, 5vw, 56px);
}

.perf-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: clamp(24px, 3vw, 32px);
    box-shadow: 0 1px 2px rgba(45, 33, 26, 0.04);
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

.perf-card-chart svg {
    width: 100%;
    height: auto;
    display: block;
}

.perf-card-caption {
    font-size: 0.92rem;
    color: var(--ink-muted);
    line-height: 1.7;
    margin: 16px 0 0;
    direction: rtl;
    text-align: right;
}

/* ---- About accordion ---- */

#workspace .gradio-accordion,
.gradio-container .accordion {
    background: transparent !important;
    border: none !important;
    border-top: 1px solid var(--border) !important;
    border-radius: 0 !important;
    margin-top: clamp(36px, 5vw, 48px) !important;
    padding-top: clamp(20px, 3vw, 24px) !important;
}

.gradio-container .accordion .label-wrap {
    color: var(--ink) !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    direction: rtl;
}

.gradio-container .accordion .prose {
    color: var(--ink) !important;
    line-height: 1.75;
    direction: rtl;
    text-align: right;
}

.gradio-container .accordion .prose h3 {
    color: var(--ink) !important;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 28px 0 12px;
}

.gradio-container .accordion .prose p { color: var(--ink-muted) !important; }

.gradio-container .accordion .prose strong { color: var(--ink) !important; font-weight: 700; }

.gradio-container .accordion .prose a {
    color: var(--terracotta-deep) !important;
    text-decoration: none;
    border-bottom: 1px solid var(--terracotta-tint);
}
.gradio-container .accordion .prose a:hover {
    border-bottom-color: var(--terracotta) !important;
}

.gradio-container .accordion .prose table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 24px;
    direction: rtl;
}

.gradio-container .accordion .prose th,
.gradio-container .accordion .prose td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    text-align: right;
    color: var(--ink) !important;
}

.gradio-container .accordion .prose th {
    color: var(--ink-muted) !important;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ---- Footer ---- */

#footer {
    width: 100%;
    background: var(--ink);
    color: rgba(245, 239, 230, 0.7);
    padding: clamp(36px, 5vw, 48px) clamp(24px, 5vw, 56px);
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
    color: var(--cream) !important;
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
}
"""


CATEGORIES_TABLE = "\n".join(
    f"| {ar} | {CATEGORIES_EN[ar]} |" for ar in CATEGORIES
)

ABOUT_MD = f"""
### What this is

A fine-tuned Arabic BERT classifier (CAMeLBERT-mix) that sorts restaurant complaints into 8 actionable categories. Built end-to-end by the NLP team at AI Club.

### Performance

95.05% test accuracy on 13,986 held-out real reviews. Bootstrap 95% CI: [94.70%, 95.41%]. Macro F1 92.03%, every class above 80% F1. Calibration ECE 0.014 after temperature scaling.

### Training data

About 98K labeled Arabic complaints. Sources: a production complaints corpus and scraped reviews from Saudi food delivery apps (HungerStation, Jahez, Mrsool, Talabat). Synthetic and augmented rows are train-only, never in val or test.

### Specialization

The model is Saudi-Gulf dialect by design. Cross-dialect canary numbers: Saudi 67%, Levantine 60%, Egyptian and pure-MSA 50%. For Egyptian or Levantine production use, retrain on data from those dialects.

### Limits

Single-label classification. For multi-aspect complaints (cold food plus rude server), the second and third predictions show the alternative aspects. Not for safety-critical decisions.

### Categories

| Arabic | English |
|---|---|
{CATEGORIES_TABLE}
"""


with gr.Blocks(
    title="تصنيف شكاوى المطاعم العربية",
    analytics_enabled=False,
    theme=gr.themes.Base(),
    css=CSS,
) as demo:
    # Hero with embedded SVG dusk scene
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

        <div id="stats-wrap">
          <div id="stats">
            <div class="stats-grid">
              <div class="stat">
                <span class="stat-key">Test accuracy</span>
                <span class="stat-val">95.05%</span>
                <span class="stat-sub">on 13,986 real reviews</span>
              </div>
              <div class="stat">
                <span class="stat-key">Categories</span>
                <span class="stat-val">8</span>
                <span class="stat-sub">all ≥ 80% F1</span>
              </div>
              <div class="stat">
                <span class="stat-key">Architecture</span>
                <span class="stat-val">CAMeLBERT-mix</span>
                <span class="stat-sub">single best of a 4-model ensemble</span>
              </div>
              <div class="stat">
                <span class="stat-key">Dialect</span>
                <span class="stat-val">Saudi / Gulf</span>
                <span class="stat-sub">specialized by design</span>
              </div>
            </div>
          </div>
        </div>
        """
    )

    # Classify section
    with gr.Column(elem_id="section-classify", elem_classes=["section", "section--cream"]):
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
                submit_btn = gr.Button("صنّف الشكوى  ·  Classify", variant="primary", scale=2)
                clear_btn = gr.Button("مسح  ·  Clear", variant="secondary", scale=1)

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

    # Performance section
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
                <div class="perf-card-chart">{CHART_F1_SVG}</div>
                <p class="perf-card-caption">
                  أعلى فئة (جودة الطعام) ٩٦٫٢٪، أدنى فئة (عامة) ٨٤٫٩٪. الفارق ١١٫٣ نقطة فقط.
                </p>
              </div>

              <div class="perf-card">
                <div class="perf-card-head">
                  <div class="perf-card-title">رحلة النموذج</div>
                  <div class="perf-card-sub">accuracy across iterations</div>
                </div>
                <div class="perf-card-chart">{CHART_BASELINES_SVG}</div>
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

    # About accordion (still useful for category list + methodology details)
    with gr.Column(elem_classes=["section", "section--cream"]):
        gr.HTML(
            """
            <div class="section-inner">
              <div class="section-eyebrow">About</div>
              <h2 class="section-title">عن النموذج</h2>
            </div>
            """
        )
        with gr.Column(elem_id="workspace"):
            with gr.Accordion("التفاصيل التقنية والمنهجية · technical details", open=False):
                gr.Markdown(ABOUT_MD)

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
