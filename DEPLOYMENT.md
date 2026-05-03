# Deployment

The ensemble manifest is [`models/ensemble_final/config.json`](models/ensemble_final/config.json) (4 models, ~1.8 GB). For a lighter deploy, use [`models/single_final/config.json`](models/single_final/config.json) (~440 MB).

## HuggingFace Spaces (recommended)

Everything needed is in [`hf_space/`](hf_space/). Step-by-step in [`hf_space/HOW_TO_DEPLOY.md`](hf_space/HOW_TO_DEPLOY.md).

Short version:

1. Push the 4 model dirs under `models/bakeoff/` to a HF Hub model repo.
2. Create a Gradio Space, copy `hf_space/` contents into it, set `HF_REPO` env var to your model repo.
3. Push. Build takes ~3 min. First inference ~30 s (cold start), then ~80 ms (CPU) / ~5 ms (GPU).

## Local

```bash
.venv/Scripts/python -m uvicorn app.api:app --port 8000
# in another terminal
curl -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"text": "وصل الطلب بارد"}'
```

Gradio with public *.gradio.live URL (72h, no account):

```bash
SHARE=true .venv/Scripts/python -m app.space_app
```

## Docker

```bash
docker build -t arabic-complaints .
docker run -p 8000:8000 -v $(pwd)/models:/app/models arabic-complaints
```

## Troubleshooting

**Cold start ~30 s on Spaces free tier.** Keep-warm with UptimeRobot or upgrade hardware tier.

**`OSError: model.safetensors not found`** — HF Hub repo not fully synced. Re-upload from local model dir.

**Wrong category for multi-aspect input** — single-label classifier picks the dominant aspect. Inspect `top_3` for the alternate. Multi-label is on the roadmap.
