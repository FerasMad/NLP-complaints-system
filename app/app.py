"""
Gradio frontend for the Arabic complaints classifier.
Calls the FastAPI service deployed at API_URL.
"""
import os
import gradio as gr
import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000/predict")


def predict(text):
    if not text or len(text.strip()) < 3:
        return {}
    try:
        r = requests.post(API_URL, json={"text": text}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {item["category"]: item["confidence"] for item in data["top_3"]}
    except requests.RequestException as e:
        return {f"خطأ في الاتصال: {e}": 1.0}


EXAMPLES = [
    "وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين",
    "الاسعار مبالغ فيها لا تناسب الجوده المقدمه ابدا",
    "النظافة سيئه الطاولات متسخه والارض غير نظيفه",
    "طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي",
    "الاكل لذيذ جدا والخدمه ممتازه",  # positive — model will force-classify
    "الديكور قديم والكراسي غير مريحه والمطعم مزدحم",
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=4,
        label="اكتب شكواك هنا",
        placeholder="مثال: الطعام بارد والتوصيل تاخر",
        rtl=True,
    ),
    outputs=gr.Label(num_top_classes=3, label="التصنيف"),
    title="تصنيف الشكاوى العربية",
    description="نموذج ذكاء اصطناعي يصنف شكاوى المطاعم العربية الى 9 فئات",
    examples=EXAMPLES,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
