"""HuggingFace Spaces entry point — Arabic Restaurant Complaints Classifier.

Loads the single best CAMeLBERT-mix model from HuggingFace Hub at startup
and serves a polished Gradio UI styled with the Thmanyah typeface.

Set HF_REPO_ID env var in your Space settings to your HF model repo ID,
e.g. "FerasMad/arabic-complaints-classifier".
"""
import base64
import os
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---- Config ----
HF_REPO_ID = os.environ.get(
    "HF_REPO_ID",
    "FerasMad/arabic-complaints-classifier",  # replace with your HF model repo
)
HF_REVISION = os.environ.get("HF_REVISION", None)
MAX_LENGTH = 192
MIN_ARABIC_RATIO = 0.30
NUM_LABELS = 8
# Optional Thmanyah typeface — not bundled in source (commercial license).
# Drop .woff2 files into FONTS_DIR locally or in the Space repo to enable them.
FONTS_DIR = Path(os.environ.get("FONTS_DIR", str(Path(__file__).parent / "fonts")))

CATEGORIES = [
    "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
    "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
]
ID2LABEL = dict(enumerate(CATEGORIES))

# ---- Text cleaning (matches training pipeline) ----
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean_arabic(text: str) -> str:
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


def is_arabic_enough(text: str, min_ratio: float = MIN_ARABIC_RATIO) -> bool:
    if not text or len(text) < 3:
        return False
    return len(ARABIC_CHAR.findall(text)) / max(len(text), 1) >= min_ratio


# ---- Embed Thmanyah fonts as data URIs ----
def font_data_uri(filename: str) -> str:
    p = FONTS_DIR / filename
    if not p.exists():
        return ""
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:font/woff2;base64,{b64}"


FONT_FACES = ""
for fam, filename, weight in [
    ("Thmanyah Sans", "thmanyahsans-Regular.woff2", 400),
    ("Thmanyah Sans", "thmanyahsans-Medium.woff2", 500),
    ("Thmanyah Sans", "thmanyahsans-Bold.woff2", 700),
    ("Thmanyah Sans", "thmanyahsans-Black.woff2", 900),
    ("Thmanyah Display", "thmanyahserifdisplay-Bold.woff2", 700),
    ("Thmanyah Display", "thmanyahserifdisplay-Black.woff2", 900),
    ("Thmanyah Text", "thmanyahseriftext-Regular.woff2", 400),
]:
    uri = font_data_uri(filename)
    if uri:
        FONT_FACES += f"""@font-face {{ font-family: "{fam}"; src: url("{uri}") format("woff2"); font-weight: {weight}; font-display: swap; }}
"""


# ---- Model loading ----
print(f"Loading {HF_REPO_ID}{' @ ' + HF_REVISION if HF_REVISION else ''} ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_kwargs = {"revision": HF_REVISION} if HF_REVISION else {}
tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID, **load_kwargs)
model = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID, **load_kwargs).to(device)
model.eval()
print(f"Model loaded on {device}.")


@torch.no_grad()
def predict(text: str):
    if not text or len(text.strip()) < 3:
        return {"النص قصير جدا — please type a longer Arabic complaint": 1.0}
    if not is_arabic_enough(text):
        return {"النص ليس باللغه العربيه — please use Arabic input": 1.0}

    cleaned = clean_arabic(text)
    enc = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    logits = model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    top_idx = probs.argsort()[::-1][:3]
    return {ID2LABEL[int(i)]: float(probs[i]) for i in top_idx}


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
PRIMARY = "#0A2540"
ACCENT = "#16C172"
BG_FROM = "#0F1A2E"
BG_TO = "#1B2D4D"
SURFACE = "#FFFFFF"
TEXT_DARK = "#0A2540"

CSS = f"""
{FONT_FACES}

* {{ font-family: "Thmanyah Sans", -apple-system, "Segoe UI", sans-serif; }}

.gradio-container {{
    background: linear-gradient(135deg, {BG_FROM} 0%, {BG_TO} 100%) !important;
    min-height: 100vh;
    padding: 32px 16px !important;
}}

.gradio-container .prose h1, .gradio-container .prose h2, .gradio-container .prose h3 {{
    font-family: "Thmanyah Display", serif !important;
    font-weight: 700 !important;
    color: {SURFACE} !important;
}}
.gradio-container .prose h1 {{ font-size: 2.4rem !important; letter-spacing: -0.02em; }}
.gradio-container .prose h2 {{ font-size: 1.3rem !important; color: {ACCENT} !important; font-weight: 500 !important; }}
.gradio-container .prose p, .gradio-container .prose strong {{
    color: rgba(255, 255, 255, 0.85) !important;
    font-family: "Thmanyah Text", "Thmanyah Sans", serif !important;
    font-size: 1.05rem; line-height: 1.6;
}}
.gradio-container .prose a {{ color: {ACCENT} !important; text-decoration: none; border-bottom: 1px dotted {ACCENT}; }}

#card_input, #card_output {{
    background: {SURFACE} !important;
    border-radius: 16px !important;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    padding: 24px !important;
}}

.gradio-container textarea {{
    font-family: "Thmanyah Sans", sans-serif !important;
    font-size: 1.1rem !important; line-height: 1.6;
    border: 2px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    color: {TEXT_DARK} !important;
}}
.gradio-container textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 4px rgba(22, 193, 114, 0.12) !important;
}}

.gradio-container label > span {{
    font-family: "Thmanyah Sans", sans-serif !important;
    font-weight: 700 !important; color: {TEXT_DARK} !important; font-size: 1rem !important;
}}

.gradio-container button.primary {{
    background: {ACCENT} !important; border: none !important; color: white !important;
    font-family: "Thmanyah Sans", sans-serif !important; font-weight: 700 !important;
    font-size: 1.05rem !important; border-radius: 10px !important; padding: 12px 28px !important;
    box-shadow: 0 4px 12px rgba(22, 193, 114, 0.3);
    transition: transform 0.15s ease;
}}
.gradio-container button.primary:hover {{ transform: translateY(-1px); box-shadow: 0 6px 18px rgba(22, 193, 114, 0.4); }}

.gradio-container button.secondary {{
    background: white !important; color: {TEXT_DARK} !important;
    border: 2px solid #E2E8F0 !important;
    font-family: "Thmanyah Sans", sans-serif !important; font-weight: 500 !important;
    border-radius: 10px !important;
}}

#main_label {{ direction: rtl; text-align: right; }}
#main_label > div > div > .label-wrap {{
    background: linear-gradient(135deg, {PRIMARY} 0%, #1B2D4D 100%) !important;
    color: white !important; border-radius: 12px !important;
    padding: 16px 20px !important;
    font-family: "Thmanyah Display", serif !important;
    font-weight: 700 !important;
}}

.examples-holder {{
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important; padding: 16px !important;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.examples-holder .label {{ color: white !important; font-family: "Thmanyah Sans", sans-serif !important; font-weight: 700 !important; }}
.examples-holder button {{
    background: rgba(255, 255, 255, 0.95) !important;
    color: {TEXT_DARK} !important; border-radius: 8px !important;
    direction: rtl; text-align: right;
    font-family: "Thmanyah Sans", sans-serif !important;
}}

.gradio-container .accordion {{
    background: rgba(255, 255, 255, 0.06) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}}
.gradio-container .accordion .label-wrap {{ color: white !important; }}
.gradio-container .accordion .prose {{ color: rgba(255, 255, 255, 0.9) !important; }}
.gradio-container .accordion table {{ color: white !important; }}

footer {{ display: none !important; }}
"""

DESCRIPTION_MD = """
نموذج CAMeLBERT-mix يصنّف شكاوى المطاعم العربية (لهجة سعودية / خليجية) إلى **8 فئات**.

**95٪ دقة** على مجموعة اختبار من ١٤,٠٠٠ مراجعة حقيقية. كل فئة ≥ ٨٤٪ F1.

اكتب الشكوى بالعربي بالأسفل ↓ وستحصل على أعلى ٣ تصنيفات مع نسبة الثقة.
"""


with gr.Blocks(title="تصنيف شكاوى المطاعم العربية", analytics_enabled=False, theme=gr.themes.Base(), css=CSS) as demo:
    gr.Markdown(
        """# تصنيف شكاوى المطاعم العربية
## Arabic Restaurant Complaints Classifier"""
    )
    gr.Markdown(DESCRIPTION_MD)

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_id="card_input"):
            input_box = gr.Textbox(
                lines=5, label="اكتب شكواك هنا",
                placeholder="مثال: الاكل بايخ ومالح والطبخ مو متقن",
                rtl=True,
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
        examples=EXAMPLES, inputs=input_box, outputs=output_label, fn=predict,
        cache_examples=False, label="أمثلة (اضغط لتجربتها)",
    )

    with gr.Accordion("عن النموذج — About this model", open=False):
        gr.Markdown(
            f"""
**Architecture:** CAMeLBERT-mix (CAMeL Lab, NYU Abu Dhabi) fine-tuned for 8-class Arabic restaurant complaint classification.

**Training data:** ~98K labeled Arabic complaints from Saudi food delivery apps + production complaint corpus.

**Source:** https://github.com/FerasMad/NLP-complaints-system

**Model:** `{HF_REPO_ID}`{' @ ' + HF_REVISION if HF_REVISION else ''}

**Typography:** [Thmanyah Typeface](https://thmanyah.com/)
"""
        )


if __name__ == "__main__":
    demo.launch()

# HF Spaces auto-launches the `demo` variable; no manual launch needed there.
demo.queue()
