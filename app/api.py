"""FastAPI service for the Arabic complaints classifier.

Endpoints:
    GET  /            human-readable info
    GET  /healthz     liveness probe (always 200 if process alive)
    GET  /readyz      readiness probe (200 only if model loaded)
    GET  /metrics     Prometheus metrics (if instrumentator installed)
    POST /predict     single text classification
    POST /predict_batch   batch classification (up to 64 texts)

Configuration via environment variables:
    ENSEMBLE_CONFIG          path to ensemble manifest (default: models/ensemble_final/config.json)
    CONFIDENCE_THRESHOLD     min top-1 prob to return a category (default 0.30)
    MIN_ARABIC_RATIO         OOD gate (default 0.30)
    ALLOWED_ORIGINS          comma-separated CORS origins (default: localhost only)
    RATE_LIMIT               per-IP req/min for /predict (default 60)
"""
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.ensemble_inference import EnsembleClassifier

# ---------- Config ----------
ENSEMBLE_CONFIG = os.environ.get("ENSEMBLE_CONFIG", "models/ensemble_final/config.json")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.30"))
MIN_ARABIC_RATIO = float(os.environ.get("MIN_ARABIC_RATIO", "0.30"))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost,http://localhost:7860,http://localhost:8000").split(",")
    if o.strip()
]
RATE_LIMIT = os.environ.get("RATE_LIMIT", "60/minute")
BATCH_SIZE_LIMIT = int(os.environ.get("BATCH_SIZE_LIMIT", "64"))

# ---------- PII scrubbing for safe logging ----------
PHONE_PATTERN = re.compile(r"\+?\d[\d\s-]{8,}\d")
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_PATTERN = re.compile(r"https?://\S+")


def scrub_pii(text: str) -> str:
    """Mask phone-number / email / URL in text before logging."""
    if not text:
        return text
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = URL_PATTERN.sub("[REDACTED_URL]", text)
    return text


# ---------- Logging ----------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":%(message)s}',
)
logger = logging.getLogger("app.api")


# ---------- App lifespan (replaces deprecated @app.on_event) ----------
state: dict = {"clf": None, "ready": False, "load_error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model. Shutdown: cleanup (none needed for now)."""
    cfg = Path(ENSEMBLE_CONFIG)
    if not cfg.exists():
        state["load_error"] = f"config not found: {ENSEMBLE_CONFIG}"
        logger.error('"ensemble config missing"')
        # Still yield so the app starts; readyz will return 503.
        yield
        return

    try:
        state["clf"] = EnsembleClassifier(cfg, min_arabic_ratio=MIN_ARABIC_RATIO)
        state["ready"] = True
        logger.info(f'"loaded {len(state[\"clf\"].models)} models from {cfg}"')
    except Exception as exc:
        state["load_error"] = str(exc)
        logger.error(f'"model load failed: {exc}"')

    yield

    # Shutdown — nothing to clean up beyond Python GC


app = FastAPI(
    title="Arabic Restaurant Complaints Classifier",
    description="4-model 8-class ensemble. 95% test accuracy on Saudi-Gulf dialect Arabic.",
    version="0.5.0",
    lifespan=lifespan,
)

# CORS — restrict to known origins (override via ALLOWED_ORIGINS env)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Optional: Prometheus metrics if dep available
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app)
    logger.info('"prometheus metrics enabled at /metrics"')
except ImportError:
    logger.info('"prometheus-fastapi-instrumentator not installed; /metrics disabled"')

# Optional: per-IP rate limiting if dep available
try:
    from slowapi import Limiter
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    from slowapi.util import get_remote_address

    limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(_request: Request, exc: RateLimitExceeded):
        return JSONResponse(
            status_code=429,
            content={"error": "rate limit exceeded", "code": "rate_limit", "detail": str(exc)},
        )

    HAS_RATE_LIMIT = True
    logger.info(f'"rate limiting enabled: {RATE_LIMIT}"')
except ImportError:
    HAS_RATE_LIMIT = False
    limiter = None
    logger.info('"slowapi not installed; rate limiting disabled"')


# ---------- Request ID middleware ----------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Generate or propagate a request ID for log correlation."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    logger.info(
        f'"request_id":"{rid}","method":"{request.method}",'
        f'"path":"{request.url.path}","status":{response.status_code},"latency_ms":{elapsed_ms:.1f}'
    )
    return response


# ---------- Pydantic models ----------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Arabic complaint text")
    top_k: int = Field(3, ge=1, le=8, description="Number of top-k categories to return")


class PredictBatchRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=64)
    top_k: int = Field(3, ge=1, le=8)


class CategoryScore(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    category: Optional[str] = Field(None, description="Top category, or null if abstaining.")
    confidence: float
    top_k: list[CategoryScore]
    abstain_reason: Optional[str] = Field(
        None, description="None on normal predict; 'out_of_domain' / 'low_confidence' / 'too_short' / 'high_entropy' on abstain."
    )


class ErrorResponse(BaseModel):
    error: str
    code: str
    request_id: Optional[str] = None


# ---------- Endpoints ----------
@app.get("/", tags=["health"])
def root():
    """Human-readable info."""
    return {
        "name": "Arabic Restaurant Complaints Classifier",
        "version": "0.5.0",
        "model": "4-model 8-class ensemble (CAMeLBERT-mix x2 + MARBERT + AraBERTv02)",
        "ready": state["ready"],
        "ensemble_size": len(state["clf"].models) if state["clf"] else 0,
        "device": str(state["clf"].device) if state["clf"] else None,
        "endpoints": ["/healthz", "/readyz", "/metrics", "POST /predict", "POST /predict_batch", "/docs"],
    }


@app.get("/healthz", tags=["health"])
def healthz():
    """Liveness probe — 200 iff the process is alive."""
    return {"status": "alive"}


@app.get("/readyz", tags=["health"])
def readyz():
    """Readiness probe — 200 only if the model is loaded."""
    if not state["ready"]:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "reason": state.get("load_error")})
    return {"status": "ready", "ensemble_size": len(state["clf"].models)}


def _predict_one(text: str, top_k: int) -> PredictResponse:
    """Shared logic for single + batch endpoints."""
    if state["clf"] is None:
        raise HTTPException(status_code=503, detail={"error": "model not loaded", "code": "not_ready"})

    result = state["clf"].predict(text, top_k=top_k)

    cat = result.category
    # Apply server-level confidence threshold (separate from model abstain)
    if cat is not None and result.confidence < CONFIDENCE_THRESHOLD:
        cat = "عامة"

    return PredictResponse(
        category=cat,
        confidence=float(result.confidence),
        top_k=[CategoryScore(category=c, confidence=s) for c, s in result.top_3],
        abstain_reason=result.abstain_reason,
    )


@app.post("/predict", response_model=PredictResponse, responses={503: {"model": ErrorResponse}})
def predict(req: PredictRequest, request: Request):
    """Classify a single Arabic restaurant complaint into one of 8 categories."""
    rid = getattr(request.state, "request_id", "unknown")
    logger.info(f'"request_id":"{rid}","action":"predict","input":"{scrub_pii(req.text)[:80]}"')
    return _predict_one(req.text, req.top_k)


@app.post("/predict_batch", response_model=list[PredictResponse], responses={503: {"model": ErrorResponse}})
def predict_batch(req: PredictBatchRequest, request: Request):
    """Classify N Arabic restaurant complaints in one round-trip."""
    if len(req.texts) > BATCH_SIZE_LIMIT:
        raise HTTPException(
            status_code=413,
            detail={"error": f"batch size {len(req.texts)} exceeds limit {BATCH_SIZE_LIMIT}", "code": "batch_too_large"},
        )
    rid = getattr(request.state, "request_id", "unknown")
    logger.info(f'"request_id":"{rid}","action":"predict_batch","n":{len(req.texts)}')
    return [_predict_one(t, req.top_k) for t in req.texts]


# ---------- Manual run ----------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
