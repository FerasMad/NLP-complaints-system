# Contributing Guide

## Branch Naming

Each member works on their own branch:

```
feras/data-collection
rima/model-training
lana/data-processing
khowla/data-processing
meshal/api-integration
mohammed/model-training
```

Create your branch from `main`:
```bash
git checkout main
git pull origin main
git checkout -b lana/data-processing
```

Open a Pull Request to `main` when your phase is complete.

---

## Current Status — Phase 0

The project is in Phase 0 (setup and onboarding). Data collection and all downstream tasks will be coordinated by the team lead. No code beyond environment setup and the synthetic reference script is expected at this stage.

---

## Phase 6 — Deployment (Meshal + Feras)

**Goal:** Put the trained system on a public website.

**Architecture:**
- `app/api.py` — FastAPI service (`POST /predict`), deployed as a Render Web Service
- `app/app.py` — Gradio interface that calls the live FastAPI endpoint, deployed as a second Render Web Service

**Meshal's Steps:**
1. Complete `app/api.py`:
   - Load `models/classifier.pkl` at startup
   - Expose `POST /predict` — accepts `{"text": "..."}`, returns `{"category": "..."}`
   - Add CORS middleware so the Gradio frontend can call it
2. Complete `app/app.py`:
   - Gradio `Interface` that posts to the live API URL (not local)
   - RTL layout hint for Arabic input text
3. Create `render.yaml` (Render blueprint) defining both services
4. Upload `models/classifier.pkl` as a Render persistent disk or GitHub release asset
5. Deploy both services on Render.com free tier
6. Verify end-to-end: enter an Arabic complaint → API returns correct category label

**What "done" means:**
- A public URL exists where anyone can type an Arabic complaint and receive a category
- Both services stay running after the team's Colab session ends

---

## What NOT to Commit

- Raw or processed data files (`data/`)
- Trained model files (`models/*.pkl`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

See `.gitignore` for the full list.
