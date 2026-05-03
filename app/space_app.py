"""Standalone Gradio app for HuggingFace Spaces or local demo.

Uses the Thmanyah typeface family (assets/fonts/) for the visual identity.

Two ways to launch:

    # Local server only (http://localhost:7860)
    python app/space_app.py

    # Local server + public *.gradio.live tunnel (72-hour link, no account needed)
    SHARE=true python app/space_app.py

    # Use the lighter single-model variant
    ENSEMBLE_CONFIG=models/single_final/config.json python app/space_app.py
"""
import base64
import json
import os
from pathlib import Path

import gradio as gr

from app.ensemble_inference import EnsembleClassifier

# ---- Config ----
ENSEMBLE_CONFIG = os.environ.get("ENSEMBLE_CONFIG", "models/ensemble_final/config.json")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.30"))
MIN_ARABIC_RATIO = float(os.environ.get("MIN_ARABIC_RATIO", "0.30"))
SHARE = os.environ.get("SHARE", "").lower() in {"1", "true", "yes", "y"}

ROOT = Path(__file__).resolve().parent.parent
# Optional Thmanyah typeface — drop the .woff2 files into FONTS_DIR locally to use them.
# Not shipped in the source repo (commercial license).
FONTS_DIR = Path(os.environ.get("FONTS_DIR", str(ROOT / "assets" / "fonts")))


# ---- Load model ----
print(f"Loading model from {ENSEMBLE_CONFIG} ...")
clf = EnsembleClassifier(Path(ENSEMBLE_CONFIG), min_arabic_ratio=MIN_ARABIC_RATIO)
ENSEMBLE_SIZE = len(clf.models)
ARCH_LABEL = (
    "4-model ensemble — CAMeLBERT-mix×2 + MARBERT + AraBERTv02"
    if ENSEMBLE_SIZE >= 4
    else (f"single CAMeLBERT-mix" if ENSEMBLE_SIZE == 1 else f"{ENSEMBLE_SIZE}-model ensemble")
)
print(f"Ready. {ENSEMBLE_SIZE} models loaded.")


# ---- Embed Thmanyah fonts as data URIs (so HF Spaces / local both work) ----
def font_data_uri(font_filename: str) -> str:
    path = FONTS_DIR / font_filename
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:font/woff2;base64,{b64}"


FONT_FACES = ""
font_specs = [
    ("Thmanyah Sans", "thmanyahsans-Regular.woff2", 400, "normal"),
    ("Thmanyah Sans", "thmanyahsans-Medium.woff2", 500, "normal"),
    ("Thmanyah Sans", "thmanyahsans-Bold.woff2", 700, "normal"),
    ("Thmanyah Sans", "thmanyahsans-Black.woff2", 900, "normal"),
    ("Thmanyah Display", "thmanyahserifdisplay-Bold.woff2", 700, "normal"),
    ("Thmanyah Display", "thmanyahserifdisplay-Black.woff2", 900, "normal"),
    ("Thmanyah Text", "thmanyahseriftext-Regular.woff2", 400, "normal"),
    ("Thmanyah Text", "thmanyahseriftext-Medium.woff2", 500, "normal"),
]
for fam, filename, weight, style in font_specs:
    uri = font_data_uri(filename)
    if uri:
        FONT_FACES += f"""@font-face {{
    font-family: "{fam}";
    src: url("{uri}") format("woff2");
    font-weight: {weight};
    font-style: {style};
    font-display: swap;
}}
"""


# ---- Prediction ----
def predict(text):
    """Gradio interface entry point. Returns dict {category: confidence}."""
    if not text or len(text.strip()) < 3:
        return {"النص قصير جدا — please type a longer Arabic complaint": 1.0}

    result = clf.predict(text, top_k=3)

    if result.abstain_reason == "out_of_domain":
        return {"النص ليس باللغه العربيه — please use Arabic input": 1.0}
    if result.abstain_reason in ("low_confidence", "high_entropy"):
        return {
            "عامة (uncertain — model is not confident)": 1.0,
            **{c: float(s) for c, s in result.top_3},
        }
    return {cat: float(score) for cat, score in result.top_3}


EXAMPLES = [
    "وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين",
    "الاسعار مبالغ فيها لا تناسب الجوده المقدمه ابدا",
    "النظافه سيئه الطاولات متسخه والارض غير نظيفه",
    "طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي",
    "انتظرت ساعه كامله في المطعم قبل ان ياتي طلبي",
    "الموظف اسلوبه سيء جدا وغير محترم",
    "الاكل بايخ ومالح والطبخ مو متقن",
    "تجربه سيئه عموما لن اعود لهذا المكان",
]


# ---- Visual identity ----
PRIMARY = "#0A2540"      # deep navy
ACCENT = "#16C172"        # emerald green
BG_GRADIENT_FROM = "#0F1A2E"
BG_GRADIENT_TO = "#1B2D4D"
SURFACE = "#FFFFFF"
TEXT_DARK = "#0A2540"
TEXT_MUTED = "#64748B"

CSS = f"""
{FONT_FACES}

* {{
    font-family: "Thmanyah Sans", -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
}}

.gradio-container {{
    background: linear-gradient(135deg, {BG_GRADIENT_FROM} 0%, {BG_GRADIENT_TO} 100%) !important;
    min-height: 100vh;
    padding: 32px 16px !important;
}}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {{
    font-family: "Thmanyah Display", serif !important;
    font-weight: 700 !important;
    color: {SURFACE} !important;
}}

.gradio-container .prose h1 {{
    font-size: 2.4rem !important;
    margin-bottom: 4px !important;
    letter-spacing: -0.02em;
}}

.gradio-container .prose h2 {{
    font-size: 1.3rem !important;
    color: {ACCENT} !important;
    font-weight: 500 !important;
    margin-top: 0 !important;
}}

.gradio-container .prose p,
.gradio-container .prose strong {{
    color: rgba(255, 255, 255, 0.85) !important;
    font-family: "Thmanyah Text", "Thmanyah Sans", serif !important;
    font-size: 1.05rem;
    line-height: 1.6;
}}

.gradio-container .prose a {{
    color: {ACCENT} !important;
    text-decoration: none;
    border-bottom: 1px dotted {ACCENT};
}}

/* Card surfaces */
#card_input, #card_output {{
    background: {SURFACE} !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    padding: 24px !important;
}}

/* Inputs */
.gradio-container textarea {{
    font-family: "Thmanyah Sans", sans-serif !important;
    font-size: 1.1rem !important;
    line-height: 1.6;
    border: 2px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    color: {TEXT_DARK} !important;
}}
.gradio-container textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 4px rgba(22, 193, 114, 0.12) !important;
}}

/* Labels above inputs */
.gradio-container label > span {{
    font-family: "Thmanyah Sans", sans-serif !important;
    font-weight: 700 !important;
    color: {TEXT_DARK} !important;
    font-size: 1rem !important;
}}

/* Primary button */
.gradio-container button.primary {{
    background: {ACCENT} !important;
    border: none !important;
    color: white !important;
    font-family: "Thmanyah Sans", sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 12px rgba(22, 193, 114, 0.3);
    transition: transform 0.15s ease;
}}
.gradio-container button.primary:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(22, 193, 114, 0.4);
}}

.gradio-container button.secondary {{
    background: white !important;
    color: {TEXT_DARK} !important;
    border: 2px solid #E2E8F0 !important;
    font-family: "Thmanyah Sans", sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
}}

/* Output label */
#main_label {{
    direction: rtl;
    text-align: right;
}}
#main_label > div > div > .label-wrap {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #1B2D4D 100%) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    font-family: "Thmanyah Display", serif !important;
    font-weight: 700 !important;
    margin-bottom: 8px !important;
}}
#main_label .confidence-set {{
    background: white !important;
}}

/* Examples */
.examples-holder {{
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.examples-holder .label {{
    color: white !important;
    font-family: "Thmanyah Sans", sans-serif !important;
    font-weight: 700 !important;
}}
.examples-holder button {{
    background: rgba(255, 255, 255, 0.95) !important;
    color: {TEXT_DARK} !important;
    border-radius: 8px !important;
    direction: rtl;
    text-align: right;
    font-family: "Thmanyah Sans", sans-serif !important;
}}

/* Accordion (About) */
.gradio-container .accordion {{
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}}
.gradio-container .accordion .label-wrap {{
    color: white !important;
    font-family: "Thmanyah Sans", sans-serif !important;
}}
.gradio-container .accordion .prose {{
    color: rgba(255, 255, 255, 0.9) !important;
}}
.gradio-container .accordion table {{
    color: white !important;
}}

/* Hide gradio footer */
footer {{ display: none !important; }}
"""

DESCRIPTION_MD = f"""
نموذج {ARCH_LABEL} يصنّف شكاوى المطاعم العربية (لهجة سعودية / خليجية) إلى **8 فئات**.

**95٪ دقة** على مجموعة اختبار من ١٤,٠٠٠ مراجعة حقيقية. كل فئة ≥ ٨٤٪ F1.

اكتب الشكوى بالعربي بالأسفل ↓ وستحصل على أعلى ٣ تصنيفات مع نسبة الثقة.
"""


with gr.Blocks(title="تصنيف شكاوى المطاعم العربية", analytics_enabled=False) as demo:
    gr.Markdown(
        """# تصنيف شكاوى المطاعم العربية
## Arabic Restaurant Complaints Classifier"""
    )
    gr.Markdown(DESCRIPTION_MD)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_id="card_input"):
            input_box = gr.Textbox(
                lines=5,
                label="اكتب شكواك هنا",
                placeholder="مثال: الاكل بايخ ومالح والطبخ مو متقن",
                rtl=True,
                show_label=True,
                elem_classes=["complaint_input"],
            )
            with gr.Row():
                submit_btn = gr.Button("صنّف الشكوى", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("مسح", variant="secondary", scale=1)

        with gr.Column(scale=2, elem_id="card_output"):
            output_label = gr.Label(num_top_classes=3, label="التصنيف", elem_id="main_label")

    submit_btn.click(predict, inputs=input_box, outputs=output_label)
    input_box.submit(predict, inputs=input_box, outputs=output_label)
    clear_btn.click(lambda: ("", {}), outputs=[input_box, output_label])

    gr.Examples(
        examples=EXAMPLES,
        inputs=input_box,
        outputs=output_label,
        fn=predict,
        cache_examples=False,
        label="أمثلة (اضغط لتجربتها)",
    )

    with gr.Accordion("📖 عن النموذج — About this model", open=False):
        gr.Markdown(
            f"""
**Architecture:** ensemble of fine-tuned Arabic BERT variants
- CAMeLBERT-mix (CAMeL Lab, NYU Abu Dhabi) × 2 seeds
- MARBERT (UBC-NLP, Twitter-trained for dialect)
- AraBERTv02 (aubmindlab)

**Training data:** ~98K labeled Arabic restaurant complaints from production data + scraped Saudi food delivery app reviews (HungerStation, Jahez, Mrsool, Talabat).

**Categories (8):**
| Arabic | English |
|---|---|
| التوصيل | Delivery — late, missing, driver issues |
| السعر والقيمة | Price and value |
| النظافة | Cleanliness, hygiene |
| جودة الطعام | Food quality — taste, freshness |
| خدمة الموظفين | Staff service and behavior |
| دقة الطلب | Order accuracy — wrong/missing items |
| عامة | General fallback |
| وقت الانتظار | In-restaurant wait time |

**Limitations:**
- Specialized for Saudi/Gulf dialect (intentional). Egyptian / Levantine / pure MSA performance is lower.
- Single-label only — for multi-aspect complaints, the top-3 list shows alternatives.
- Not for safety-critical decisions.

**Source code:** [github.com/FerasMad/NLP-complaints-system](https://github.com/FerasMad/NLP-complaints-system)
"""
        )

    gr.Markdown(
        """
---
*AI Club portfolio project • Designed with the [Thmanyah](https://thmanyah.com/) typeface*
"""
    )


if __name__ == "__main__":
    demo.launch(
        share=SHARE,
        server_name="0.0.0.0" if SHARE else "127.0.0.1",
        server_port=int(os.environ.get("PORT", 7860)),
        allowed_paths=[str(FONTS_DIR)] if FONTS_DIR.exists() else [],
        theme=gr.themes.Base(),
        css=CSS,
    )
