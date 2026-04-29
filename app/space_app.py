"""
Standalone Gradio app for HuggingFace Spaces.
Loads the model directly — no external API call, no cold-start delay.

Deploy by copying this file (renamed to app.py) into a HuggingFace Space
along with classifier.pkl and requirements.txt.
"""
import os
import re
import joblib
import gradio as gr

MODEL_PATH = os.environ.get("MODEL_PATH", "classifier.pkl")
CONFIDENCE_THRESHOLD = 0.30

CATEGORIES = [
    "التوصيل", "الجو والمكان", "السعر والقيمة", "النظافة",
    "جودة الطعام", "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
]

TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")


def clean(text):
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()


print(f"Loading model from {MODEL_PATH}...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")


def predict(text):
    if not text or len(text.strip()) < 3:
        return {}
    cleaned = clean(text)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([cleaned])[0]
    else:
        import numpy as np
        scores = model.decision_function([cleaned])[0]
        e = np.exp(scores - scores.max())
        probs = e / e.sum()
    pairs = sorted(zip(CATEGORIES, probs), key=lambda x: -x[1])[:3]
    return {cat: float(score) for cat, score in pairs}


EXAMPLES = [
    "وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين",
    "الاسعار مبالغ فيها لا تناسب الجوده المقدمه ابدا",
    "النظافة سيئه الطاولات متسخه والارض غير نظيفه",
    "طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي",
    "الديكور قديم والكراسي غير مريحه والمطعم مزدحم",
]


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="اكتب شكواك هنا", rtl=True),
    outputs=gr.Label(num_top_classes=3, label="التصنيف"),
    title="تصنيف الشكاوى العربية",
    description="نموذج ذكاء اصطناعي يصنف شكاوى المطاعم العربية الى 9 فئات",
    examples=EXAMPLES,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
