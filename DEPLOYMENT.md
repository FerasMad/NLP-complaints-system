# Deployment Guide

This guide covers two deployment paths. Pick one. **Path A (HuggingFace Spaces)** is recommended for the demo — it's free, no cold starts, and one service to manage. Path B (Render) is the original architecture with a separate API + UI for portfolio breadth.

## Prerequisites

Before deploying, you need:

1. **Trained model:** `models/classifier.pkl` from Phase 3
2. **GitHub account** (Path B) or **HuggingFace account** (Path A)
3. **Optional:** GitHub release with classifier.pkl uploaded as an asset (Path B)

---

## Path A — HuggingFace Spaces (RECOMMENDED for demo)

**Why this is better for the demo:**
- Free tier doesn't sleep (Render free tier sleeps after 15 min)
- Native Gradio support
- Model lives in the same repo — no separate model hosting
- Public URL like `huggingface.co/spaces/AIClub/arabic-complaints`
- Setup time: ~15 minutes

### Step-by-step

1. **Create a Space:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Owner: your username or org
   - Name: `arabic-complaints-classifier`
   - License: `mit`
   - SDK: **Gradio**
   - Hardware: CPU basic (free)
   - Visibility: Public

2. **Clone the empty space locally:**
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/arabic-complaints-classifier
   cd arabic-complaints-classifier
   ```

3. **Copy these files in:**
   ```
   app.py              (rename of app/space_app.py from this repo)
   requirements.txt    (copy from this repo)
   classifier.pkl      (the trained model from Phase 3)
   README.md           (write a short description for the Space)
   ```

4. **Minimal `requirements.txt` for the Space** (only the runtime dependencies):
   ```
   scikit-learn>=1.3
   joblib>=1.3
   gradio>=4.0
   numpy>=1.24
   ```

5. **Push:**
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git add .gitattributes app.py requirements.txt classifier.pkl README.md
   git commit -m "Initial deployment"
   git push
   ```

6. **Wait ~5 minutes** for the Space to build, then visit:
   `https://huggingface.co/spaces/<your-user>/arabic-complaints-classifier`

### Result
A public Arabic complaint classifier demo, no cold starts, free forever.

---

## Path B — Render.com (FastAPI + Gradio, original plan)

**Why this path:** matches the original architecture (separate API microservice + UI), and gives you a public REST API endpoint anyone can integrate with.

**Tradeoffs:** Free tier sleeps after 15 min idle (cold start ~30s). Two services to manage.

### Step 1 — Host the model file

The model is too large for git. Pick one:

**Option 1: GitHub Release (simplest)**
```bash
# in your repo
gh release create v1.0 models/classifier.pkl --title "Trained Model v1" --notes "Phase 3 classifier"
# get the asset URL — looks like:
# https://github.com/<user>/<repo>/releases/download/v1.0/classifier.pkl
```

**Option 2: HuggingFace Hub** — upload as a model repo, fetch with `huggingface_hub.hf_hub_download`

### Step 2 — Push code to GitHub

Make sure these are in main:
- `app/api.py`
- `app/app.py`
- `render.yaml`
- `requirements.txt`
- `models/.gitkeep` (model itself NOT committed)

### Step 3 — Connect Render to the repo

1. Sign in at https://dashboard.render.com
2. New → Blueprint
3. Pick your GitHub repo, branch `main`
4. Render reads `render.yaml` and creates two services: `complaints-api` and `complaints-ui`

### Step 4 — Set environment variables

In the **complaints-api** service settings:
- `MODEL_URL` = your GitHub release asset URL (or HF hub URL)

Wait for first deploy to finish, then note the API URL (e.g. `https://complaints-api.onrender.com`).

In the **complaints-ui** service settings:
- `API_URL` = `https://complaints-api.onrender.com/predict`

Trigger a manual redeploy of the UI.

### Step 5 — Verify end-to-end

```bash
# health check
curl https://complaints-api.onrender.com/

# test prediction
curl -X POST https://complaints-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "وصل الطلب بارد والمندوب تاخر ساعتين"}'
```

Expected response:
```json
{
  "category": "التوصيل",
  "confidence": 0.81,
  "top_3": [
    {"category": "التوصيل", "confidence": 0.81},
    {"category": "وقت الانتظار", "confidence": 0.10},
    {"category": "جودة الطعام", "confidence": 0.05}
  ]
}
```

Then open the UI URL (e.g. `https://complaints-ui.onrender.com`) — type an Arabic complaint, see the top-3 categories.

---

## Path C — Local development

Run both services locally for testing:

```bash
# terminal 1: API
pip install -r requirements.txt
python -m uvicorn app.api:app --reload --port 8000

# terminal 2: UI
API_URL=http://localhost:8000/predict python app/app.py
# Open http://localhost:7860
```

Or just the standalone Space app:
```bash
MODEL_PATH=models/classifier.pkl python app/space_app.py
```

---

## Post-deployment checklist

- [ ] Public URL works for an unauthenticated user
- [ ] Top-3 confidence scores display in the UI
- [ ] Arabic input renders right-to-left
- [ ] Examples in the demo all classify reasonably
- [ ] README.md updated with the live URL and a screenshot/GIF
- [ ] LinkedIn / portfolio post drafted with the live URL

---

## Troubleshooting

**Cold start on Render free tier**
The first request after 15 min of idle time takes ~30s while the service wakes up. Subsequent requests are fast. To eliminate this, either:
- Use HuggingFace Spaces instead (Path A)
- Upgrade to Render Starter ($7/mo)
- Use a free uptime ping service to keep it warm (UptimeRobot, every 14 min)

**Model download fails on build**
Make sure `MODEL_URL` is set and points to a publicly accessible asset. Test with `curl -L $MODEL_URL -o /tmp/model.pkl`.

**Model file too large for git**
Don't commit it. Use a GitHub Release asset, HuggingFace Hub, or LFS.

**Gradio shows "Cannot connect to API"**
The API service is sleeping (cold start). Wait 30s and retry.

**CORS error in browser**
The FastAPI service already has CORS configured for `*`. If you tighten it later, add the UI domain explicitly.
