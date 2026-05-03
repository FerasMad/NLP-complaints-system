# Multi-stage build: small final image with model artifacts pulled at runtime.
# Build: docker build -t arabic-complaints .
# Run: docker run -p 8000:8000 -e HF_TOKEN=$HF_TOKEN arabic-complaints
#
# For GPU: add `--gpus all` and use the cuda base image variant.

FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build deps + CPU torch (slim deploy default; for GPU use the cuda image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip install prometheus-fastapi-instrumentator slowapi

# ---- runtime ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Arabic Restaurant Complaints Classifier"
LABEL org.opencontainers.image.description="4-model 8-class ensemble — 95% test accuracy"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/FerasMad/NLP-complaints-system"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app code (NOT models — those are pulled at boot from HF Hub)
COPY app/ ./app/
COPY src/ ./src/
COPY data/processed/label_map.json ./data/processed/label_map.json
COPY models/ensemble_final/config.json ./models/ensemble_final/config.json

ENV PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    ENSEMBLE_CONFIG=models/ensemble_final/config.json

EXPOSE 8000

# Healthcheck hits /healthz (always 200 if process alive)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
