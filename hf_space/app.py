"""HuggingFace Spaces entry point for the Arabic Restaurant Complaints Classifier.

Visual identity is warm, hospitable, Saudi-rooted. Cream surface, terracotta
accent, deep-ink type. Almarai for Arabic and Latin in one family.
"""
import os
import re

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


print(f"Loading {HF_REPO_ID} ...")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID).to(device).eval()
print(f"Model loaded on {device}.")


# Helpful empty-state HTML used when there is no prediction yet.
EMPTY_RESULT = """
<div class="result-empty">
  <div class="result-empty-mark">·</div>
  <div class="result-empty-text">
    <strong>اكتب شكوى وستظهر النتيجة هنا</strong>
    <span>type a complaint to see the prediction</span>
  </div>
</div>
"""


def render_result(top: list[tuple[str, float]]) -> str:
    """Custom HTML for the top-3 predictions; rank-colored by background tint."""
    if not top:
        return EMPTY_RESULT
    rows = []
    for rank, (cat, score) in enumerate(top):
        en = CATEGORIES_EN.get(cat, "")
        pct = f"{score * 100:.0f}%"
        rows.append(
            f'<div class="result-row result-rank-{rank + 1}">'
            f'  <div class="result-meta">'
            f'    <span class="result-rank">{rank + 1}</span>'
            f'    <span class="result-en">{en}</span>'
            f'  </div>'
            f'  <div class="result-cat">{cat}</div>'
            f'  <div class="result-pct">{pct}</div>'
            f'</div>'
        )
    return f'<div class="result-stack">{"".join(rows)}</div>'


def render_message(headline_ar: str, headline_en: str) -> str:
    """Inline message used for empty / non-Arabic / too-short cases."""
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


# ---- Visual identity -------------------------------------------------------
# Warm, hospitable, Saudi-rooted. Cream surface, terracotta accent, deep ink.

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Almarai:wght@300;400;700;800&display=swap');

:root {
    --cream: #F5EFE6;
    --paper: #EEE6D7;
    --ink: #2D211A;
    --ink-muted: #5C4F45;
    --terracotta: #C75D3D;
    --terracotta-deep: #A14828;
    --terracotta-tint: rgba(199, 93, 61, 0.10);
    --olive: #5F6845;
    --olive-tint: rgba(95, 104, 69, 0.10);
    --border: #D9CFC0;
    --border-strong: #C9BDA8;
    --shadow: 0 1px 2px rgba(45, 33, 26, 0.04), 0 8px 24px rgba(45, 33, 26, 0.06);
}

* { font-family: 'Almarai', system-ui, -apple-system, sans-serif !important; }

html, body, .gradio-container {
    background: var(--cream) !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
    padding: 0 !important;
}

/* ---- Hero band ---- */

#hero {
    background: linear-gradient(180deg, var(--terracotta) 0%, var(--terracotta-deep) 100%);
    color: var(--cream);
    padding: clamp(40px, 7vw, 72px) clamp(24px, 5vw, 56px) clamp(48px, 8vw, 88px);
    text-align: right;
    direction: rtl;
    position: relative;
    overflow: hidden;
}

#hero::after {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 90% 0%, rgba(255, 240, 220, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 0% 100%, rgba(0, 0, 0, 0.10) 0%, transparent 70%);
    pointer-events: none;
}

#hero .eyebrow {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(245, 239, 230, 0.78);
    margin: 0 0 16px;
    direction: ltr;
    text-align: right;
    font-family: 'Almarai', monospace !important;
}

#hero h1 {
    font-size: clamp(2.2rem, 5.4vw, 3.6rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.01em;
    margin: 0 0 12px;
    color: var(--cream);
    max-width: 18ch;
    margin-right: 0;
}

#hero .lede {
    font-size: clamp(1rem, 1.5vw, 1.15rem);
    line-height: 1.7;
    color: rgba(245, 239, 230, 0.88);
    max-width: 56ch;
    margin: 0 0 24px;
    font-weight: 400;
}

#hero .meta {
    display: flex;
    flex-wrap: wrap;
    gap: 18px 32px;
    direction: ltr;
    justify-content: flex-end;
    border-top: 1px solid rgba(245, 239, 230, 0.18);
    padding-top: 20px;
}

#hero .meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    text-align: left;
    font-family: 'Almarai', system-ui !important;
}

#hero .meta-key {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: rgba(245, 239, 230, 0.6);
}

#hero .meta-val {
    font-size: 1rem;
    font-weight: 700;
    color: var(--cream);
}

/* ---- Working surface ---- */

#workspace {
    padding: clamp(32px, 5vw, 56px) clamp(20px, 4vw, 48px);
    background: var(--cream);
}

.section-label {
    font-size: 0.74rem !important;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--ink-muted);
    margin: 0 0 14px;
}

/* ---- Input ---- */

#input_panel {
    background: transparent;
    padding: 0;
}

#input_panel .gr-form,
#input_panel .form,
#input_panel .gr-block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

#input_panel textarea {
    width: 100%;
    background: var(--paper) !important;
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

#input_panel label > span {
    display: none !important;
}

/* ---- Buttons ---- */

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
    letter-spacing: 0.01em;
    cursor: pointer;
    transition: background 140ms ease, transform 140ms ease;
}

button.primary:hover {
    background: var(--terracotta-deep) !important;
    transform: translateY(-1px);
}

button.primary:active {
    transform: translateY(0);
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
    transition: background 140ms ease, color 140ms ease, border-color 140ms ease;
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
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 24px;
    border: 1px dashed var(--border-strong);
    border-radius: 14px;
    background: var(--paper);
    color: var(--ink-muted);
    direction: rtl;
}

.result-empty-mark {
    font-size: 2.4rem;
    font-weight: 800;
    color: var(--terracotta);
    line-height: 1;
}

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

.result-stack {
    display: grid;
    gap: 10px;
    direction: rtl;
}

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
    font-family: 'Almarai', system-ui !important;
}

.result-rank {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
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

/* Rank-1: terracotta drench */
.result-rank-1 {
    background: var(--terracotta);
    color: var(--cream);
}
.result-rank-1 .result-rank,
.result-rank-1 .result-cat,
.result-rank-1 .result-pct { color: var(--cream); }
.result-rank-1 .result-en { color: rgba(245, 239, 230, 0.7); }

/* Rank-2: olive tint */
.result-rank-2 {
    background: var(--olive-tint);
    border: 1px solid rgba(95, 104, 69, 0.22);
}
.result-rank-2 .result-rank { color: var(--olive); }

/* Rank-3: paper, muted */
.result-rank-3 {
    background: var(--paper);
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
#examples_panel .label {
    display: none !important;
}

#examples_panel .gr-sample-btn,
#examples_panel button {
    background: transparent !important;
    color: var(--ink) !important;
    border: 1px solid var(--border) !important;
    border-radius: 999px !important;
    padding: 10px 18px !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    direction: rtl;
    text-align: right;
    transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    cursor: pointer;
}

#examples_panel .gr-sample-btn:hover,
#examples_panel button:hover {
    background: var(--ink) !important;
    color: var(--cream) !important;
    border-color: var(--ink) !important;
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
    background: transparent;
}

/* ---- Footer ---- */

#footer {
    background: var(--paper);
    border-top: 1px solid var(--border);
    padding: 28px clamp(20px, 4vw, 48px);
    color: var(--ink-muted);
    font-size: 0.9rem;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
}

#footer .links {
    display: flex;
    gap: 22px;
    flex-wrap: wrap;
}

#footer a {
    color: var(--ink) !important;
    text-decoration: none;
    font-weight: 700;
    border-bottom: 1px solid transparent;
    transition: border-color 140ms ease, color 140ms ease;
}

#footer a:hover {
    color: var(--terracotta-deep) !important;
    border-bottom-color: var(--terracotta);
}

#footer .made-by {
    font-weight: 400;
    color: var(--ink-muted);
}

/* ---- Hide HF Spaces footer ---- */
footer { display: none !important; }

/* ---- Reduced motion ---- */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}

/* ---- Mobile ---- */
@media (max-width: 640px) {
    #hero .meta { gap: 14px 24px; }
    #hero .meta-val { font-size: 0.92rem; }
    .result-row {
        grid-template-columns: auto 1fr;
        gap: 12px 16px;
    }
    .result-pct { grid-column: 2; text-align: left; margin-top: 4px; }
    #actions button { flex: 1; }
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
    gr.HTML(
        """
        <section id="hero">
          <div class="eyebrow">Arabic Restaurant Complaints Classifier</div>
          <h1>تصنيف شكاوى المطاعم العربية</h1>
          <p class="lede">
            اكتب أي شكوى بالعربية وسيصنّفها النموذج إلى واحدة من ٨ فئات.
            مدرّب على ٩٨٬٠٠٠ مراجعة من تطبيقات التوصيل السعودية. لهجة سعودية / خليجية.
          </p>
          <div class="meta">
            <div class="meta-item">
              <span class="meta-key">Test accuracy</span>
              <span class="meta-val">95.05%</span>
            </div>
            <div class="meta-item">
              <span class="meta-key">Categories</span>
              <span class="meta-val">8</span>
            </div>
            <div class="meta-item">
              <span class="meta-key">Architecture</span>
              <span class="meta-val">CAMeLBERT-mix</span>
            </div>
            <div class="meta-item">
              <span class="meta-key">Dialect</span>
              <span class="meta-val">Saudi / Gulf</span>
            </div>
          </div>
        </section>
        """
    )

    with gr.Column(elem_id="workspace"):
        gr.HTML('<div class="section-label">شكواك · your complaint</div>')

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
            gr.HTML('<div class="section-label">التصنيف · prediction</div>')
            output_html = gr.HTML(value=EMPTY_RESULT)

        with gr.Column(elem_id="examples_panel"):
            gr.HTML('<div class="section-label">جرّب أمثلة · try an example</div>')
            gr.Examples(
                examples=EXAMPLES,
                inputs=input_box,
                outputs=output_html,
                fn=predict,
                cache_examples=False,
                label=None,
            )

        with gr.Accordion("عن النموذج · About the model", open=False):
            gr.Markdown(ABOUT_MD)

    gr.HTML(
        f"""
        <footer id="footer">
          <span class="made-by">Made by the NLP team at AI Club</span>
          <span class="links">
            <a href="{GITHUB_URL}" target="_blank" rel="noopener">Source on GitHub</a>
            <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank" rel="noopener">Model on HF Hub</a>
          </span>
        </footer>
        """
    )

    submit_btn.click(predict, inputs=input_box, outputs=output_html)
    input_box.submit(predict, inputs=input_box, outputs=output_html)
    clear_btn.click(lambda: ("", EMPTY_RESULT), outputs=[input_box, output_html])


demo.launch(server_name="0.0.0.0", server_port=7860)
