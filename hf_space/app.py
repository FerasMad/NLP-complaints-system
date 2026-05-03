"""HuggingFace Spaces entry point — Arabic Restaurant Complaints Classifier.

Loads the single best CAMeLBERT-mix model from HuggingFace Hub and serves
a Gradio UI. Set HF_REPO_ID env var in the Space settings to point at
your model repo.
"""
import os
import re

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HF_REPO_ID = os.environ.get(
    "HF_REPO_ID", "FerasMad/arabic-complaints-classifier"
)
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
        return {"النص قصير جدا — please type a longer Arabic complaint": 1.0}
    if not is_arabic_enough(text):
        return {"النص ليس باللغه العربيه — please use Arabic input": 1.0}

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


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        label="اكتب الشكوى",
        placeholder="مثال: الاكل بايخ ومالح والطبخ مو متقن",
        rtl=True,
    ),
    outputs=gr.Label(num_top_classes=3, label="التصنيف"),
    examples=EXAMPLES,
    title="تصنيف شكاوى المطاعم العربية — Arabic Restaurant Complaints Classifier",
    description=(
        "نموذج CAMeLBERT-mix مدرب على ٨ فئات من الشكاوى. لهجة سعودية / خليجية. "
        "Fine-tuned CAMeLBERT-mix · 8 categories · 95% test accuracy · Saudi-Gulf dialect."
    ),
    article=(
        "**Source:** [github.com/FerasMad/NLP-complaints-system]"
        "(https://github.com/FerasMad/NLP-complaints-system)"
    ),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
