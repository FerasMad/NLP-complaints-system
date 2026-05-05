"""HuggingFace Spaces entry point — Arabic Restaurant Complaints Classifier.

Loads the single best CAMeLBERT-mix model from HuggingFace Hub and serves
a Gradio UI with a custom dark theme + Tajawal Arabic webfont.
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


@torch.no_grad()
def predict(text: str):
    if not text or len(text.strip()) < 3:
        return {"اكتب شكوى أطول من ٣ أحرف · type a longer complaint": 1.0}
    if not is_arabic_enough(text):
        return {"النص ليس بالعربية · please use Arabic input": 1.0}

    enc = tokenizer(clean(text), return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    probs = torch.softmax(model(**enc).logits[0], dim=-1).cpu().numpy()
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
PRIMARY = "#0a1628"     # deep navy
SURFACE = "#14253f"     # card surface
ACCENT = "#16C172"      # emerald
ACCENT_HOVER = "#1ad888"
TEXT = "#ffffff"
MUTED = "rgba(255,255,255,0.7)"
BORDER = "rgba(255,255,255,0.1)"

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');

* {{ font-family: 'Tajawal', -apple-system, 'Segoe UI', sans-serif !important; }}

.gradio-container {{
    background: linear-gradient(180deg, {PRIMARY} 0%, {SURFACE} 100%) !important;
    min-height: 100vh;
    padding: 24px 16px !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}}

#hero {{
    text-align: center;
    padding: 32px 16px 24px;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 32px;
}}

#hero h1 {{
    color: {TEXT} !important;
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    margin: 0 0 8px 0 !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
}}

#hero h2 {{
    color: {ACCENT} !important;
    font-size: 1.1rem !important;
    font-weight: 500 !important;
    margin: 0 0 16px 0 !important;
    letter-spacing: 0.02em;
}}

#hero p {{
    color: {MUTED} !important;
    font-size: 0.95rem;
    margin: 0;
    max-width: 560px;
    margin-left: auto;
    margin-right: auto;
    line-height: 1.6;
}}

#hero .badges {{
    display: flex;
    gap: 8px;
    justify-content: center;
    margin-top: 16px;
    flex-wrap: wrap;
}}

#hero .badge {{
    background: rgba(22,193,114,0.12);
    color: {ACCENT};
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    border: 1px solid rgba(22,193,114,0.25);
}}

#card_input, #card_output {{
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid {BORDER};
    border-radius: 16px !important;
    padding: 24px !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}}

textarea {{
    font-family: 'Tajawal', sans-serif !important;
    font-size: 1.05rem !important;
    background: rgba(0,0,0,0.25) !important;
    color: {TEXT} !important;
    border: 2px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 14px !important;
    line-height: 1.6 !important;
}}

textarea:focus {{
    border-color: {ACCENT} !important;
    outline: none;
    box-shadow: 0 0 0 4px rgba(22,193,114,0.12) !important;
}}

textarea::placeholder {{
    color: rgba(255,255,255,0.35) !important;
}}

button.primary {{
    background: {ACCENT} !important;
    color: {PRIMARY} !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(22,193,114,0.25);
}}

button.primary:hover {{
    background: {ACCENT_HOVER} !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(22,193,114,0.35);
}}

button.secondary {{
    background: transparent !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    padding: 12px 20px !important;
}}

button.secondary:hover {{
    background: rgba(255,255,255,0.05) !important;
    border-color: {ACCENT} !important;
}}

label > span {{
    color: rgba(255,255,255,0.9) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    margin-bottom: 8px !important;
}}

/* Prediction label component */
.output-class, .output-label {{
    background: linear-gradient(135deg, rgba(22,193,114,0.15) 0%, rgba(22,193,114,0.05) 100%) !important;
    border: 1px solid rgba(22,193,114,0.3) !important;
    border-radius: 12px !important;
}}

.output-label .confidence-set .confidence-bar {{
    background: linear-gradient(90deg, {ACCENT} 0%, #1ad888 100%) !important;
}}

/* Examples row */
.examples-holder, .gradio-container .examples {{
    background: rgba(255,255,255,0.03) !important;
    border-radius: 12px !important;
    border: 1px solid {BORDER} !important;
    padding: 16px !important;
    margin-top: 16px;
}}

.examples-holder button, .gradio-container .examples button {{
    background: rgba(255,255,255,0.06) !important;
    color: {TEXT} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', sans-serif !important;
    padding: 8px 14px !important;
    transition: all 0.15s;
}}

.examples-holder button:hover, .gradio-container .examples button:hover {{
    background: rgba(22,193,114,0.15) !important;
    border-color: {ACCENT} !important;
    transform: translateY(-1px);
}}

.examples-holder .label, .gradio-container .examples .label {{
    color: rgba(255,255,255,0.6) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

/* About accordion */
.gradio-container .accordion {{
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    margin-top: 24px;
}}

.gradio-container .accordion .label-wrap {{
    color: {TEXT} !important;
    font-weight: 700 !important;
}}

.gradio-container .accordion .prose {{
    color: rgba(255,255,255,0.85) !important;
    line-height: 1.7;
}}

.gradio-container .accordion table {{
    color: rgba(255,255,255,0.85) !important;
    border-collapse: collapse;
    width: 100%;
}}

.gradio-container .accordion table th,
.gradio-container .accordion table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {BORDER};
    text-align: right;
}}

.gradio-container .accordion table th {{
    color: {ACCENT} !important;
    font-weight: 700;
    background: rgba(22,193,114,0.05);
}}

#footer {{
    text-align: center;
    padding: 32px 16px 16px;
    color: rgba(255,255,255,0.5);
    font-size: 0.85rem;
    border-top: 1px solid {BORDER};
    margin-top: 32px;
}}

#footer a {{
    color: {ACCENT} !important;
    text-decoration: none;
    font-weight: 500;
}}

#footer a:hover {{
    color: {ACCENT_HOVER} !important;
    text-decoration: underline;
}}

footer {{ display: none !important; }}
"""

CATEGORIES_TABLE = "\n".join(
    f"| {ar} | {CATEGORIES_EN[ar]} |" for ar in CATEGORIES
)

ABOUT_MD = f"""
**Architecture.** Fine-tuned [CAMeLBERT-mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix), single-best member of a 4-model ensemble.

**Test accuracy.** 95.05% on a held-out set of 13,986 real reviews. Every category ≥ 80% F1. Bootstrap 95% CI: [94.70%, 95.41%].

**Training data.** ~98K labeled Arabic complaints from production data + scraped Saudi food delivery apps (HungerStation, Jahez, Mrsool, Talabat).

**Specialization.** Saudi-Gulf dialect by design. Egyptian / Levantine / pure MSA performance is lower (Saudi 67%, Levantine 60%, Egyptian 50% on a small canary).

**Limitations.** Single-label only — for multi-aspect complaints, the top-3 list shows alternative categories. Not for safety-critical decisions.

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
        <div id="hero">
          <h1>تصنيف شكاوى المطاعم العربية</h1>
          <h2>Arabic Restaurant Complaints Classifier</h2>
          <p>اكتب شكواك بالعربية وسيصنّفها النموذج إلى واحدة من ٨ فئات. لهجة سعودية / خليجية.</p>
          <div class="badges">
            <span class="badge">95% test accuracy</span>
            <span class="badge">8 categories</span>
            <span class="badge">CAMeLBERT-mix</span>
            <span class="badge">Saudi-Gulf dialect</span>
          </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, elem_id="card_input"):
            input_box = gr.Textbox(
                lines=6,
                label="اكتب شكواك",
                placeholder="مثال: الاكل بايخ ومالح والطبخ مو متقن",
                rtl=True,
                show_label=True,
            )
            with gr.Row():
                submit_btn = gr.Button("صنّف الشكوى", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("مسح", variant="secondary", scale=1)

        with gr.Column(scale=1, elem_id="card_output"):
            output_label = gr.Label(
                num_top_classes=3,
                label="أعلى ٣ تصنيفات",
                show_label=True,
            )

    submit_btn.click(predict, inputs=input_box, outputs=output_label)
    input_box.submit(predict, inputs=input_box, outputs=output_label)
    clear_btn.click(lambda: ("", {}), outputs=[input_box, output_label])

    gr.Examples(
        examples=EXAMPLES,
        inputs=input_box,
        outputs=output_label,
        fn=predict,
        cache_examples=False,
        label="جرّب أمثلة",
    )

    with gr.Accordion("عن النموذج · About this model", open=False):
        gr.Markdown(ABOUT_MD)

    gr.HTML(
        f"""
        <div id="footer">
          Made by the NLP team at AI Club &middot;
          <a href="{GITHUB_URL}" target="_blank">Source on GitHub</a> &middot;
          <a href="https://huggingface.co/{HF_REPO_ID}" target="_blank">Model on HF Hub</a>
        </div>
        """
    )


demo.launch(server_name="0.0.0.0", server_port=7860)
