"""
FastAPI service for the Arabic complaints classifier.
Loads classifier.pkl at startup and exposes POST /predict.
"""
import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/classifier.pkl")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.30"))

CATEGORIES = [
    "التوصيل", "الجو والمكان", "السعر والقيمة", "النظافة",
    "جودة الطعام", "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
]

# matches src/build_dataset.py cleaning
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


app = FastAPI(title="Arabic Complaint Classifier", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")


class PredictRequest(BaseModel):
    text: str


class CategoryScore(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    category: str
    confidence: float
    top_3: list[CategoryScore]


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = clean(req.text)
    if len(text) < 3:
        raise HTTPException(status_code=400, detail="Text too short")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
    else:
        # LinearSVC fallback: decision_function then softmax
        import numpy as np
        scores = model.decision_function([text])[0]
        e = np.exp(scores - scores.max())
        probs = e / e.sum()

    pairs = sorted(zip(CATEGORIES, probs), key=lambda x: -x[1])
    top_cat, top_conf = pairs[0]
    top_3 = [CategoryScore(category=c, confidence=float(s)) for c, s in pairs[:3]]

    # low confidence fallback
    if top_conf < CONFIDENCE_THRESHOLD:
        top_cat = "عامة"

    return PredictResponse(
        category=top_cat,
        confidence=float(top_conf),
        top_3=top_3,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
