# Project Handoff — Arabic Complaints Classifier

Single document with everything needed to continue the project on another machine. Drop this file in the project root and any new Claude session (or human) can read it and pick up where we left off.

---

## 1. What this project is

**Goal:** Build an Arabic NLP system that classifies restaurant complaints into **8 actionable categories** (was 9 — ambiance dropped after audit found unrecoverable label noise). Deploy as a public demo.

**Context:** AI Club portfolio project (NOT a course assignment). Quality bar = production-grade. Owner: Feras Madkhali (team lead).

**Approach:** 4-model ensemble of fine-tuned Arabic BERT variants (CAMeLBERT-mix × 2 seeds + MARBERT + AraBERTv02). Single-model deploy variant also supported.

---

## 2. The 8 categories

| ID | Category (Arabic) | Description |
|----|-------------------|-------------|
| 0 | التوصيل | Delivery — late, missing, wrong address, driver issues |
| 1 | السعر والقيمة | Price and value |
| 2 | النظافة | Cleanliness, hygiene |
| 3 | جودة الطعام | Food quality — taste, freshness, portion |
| 4 | خدمة الموظفين | Staff service and behavior |
| 5 | دقة الطلب | Order accuracy — wrong/missing items |
| 6 | عامة | General fallback |
| 7 | وقت الانتظار | In-restaurant wait time |

**Removed:** الجو والمكان (ambiance). Audit revealed only 2 of 171 val+test ambiance samples were truly clean ambiance (1.2%); 47% were probably mislabeled (other category dominates), 40% were multi-aspect. Category was unrecoverable without rebuilding the eval set from scratch. See REPORT.md §"Why we dropped" for the data.

---

## 3. Phase status

| Phase | Description | Owner | Status |
|-------|-------------|-------|--------|
| 0 | Setup & onboarding | All | done |
| 1 | Text cleaning | Lana | done |
| 2 | Labeling & splitting | Khowla | done |
| 3 | Model training | Rima + Mohammed (Feras covering) | done — 4-model ensemble at 95.05% test acc |
| 4 | Evaluation | Feras | done — bootstrap CI, calibration, robustness, per-source, per-dialect — see `REPORT.md` |
| 5 | Gradio demo | All | done (`app/space_app.py`) — loads ensemble from `models/ensemble_final/` |
| 6 | Deployment (HuggingFace Spaces) | Meshal + Feras | pending HF account creation — turnkey files in `hf_space/`, see `hf_space/HOW_TO_DEPLOY.md` |

---

## 4. Dataset

**~98,000 labeled rows** across 8 categories — assembled from 6 sources, then split 70/15/15 stratified on real-only data.

Per-category training pool (real + augmented):

| Category | Real | Synthetic + augmented | Total train |
|----------|-----:|----------------------:|------------:|
| جودة الطعام | ~45,000 | 0 | ~45,000 (capped to 31K used) |
| السعر والقيمة | ~17,400 | 0 | ~17,400 |
| خدمة الموظفين | ~15,200 | 0 | ~15,200 |
| التوصيل | ~5,700 | 0 | ~5,700 |
| وقت الانتظار | ~4,200 | 0 | ~4,200 |
| النظافة | ~3,400 | 0 | ~3,400 |
| دقة الطلب | ~700 | ~3,500 (synthetic + EDA + pseudo) | ~4,200 |
| عامة | ~1,400 | ~3,200 (synthetic + EDA + pseudo) | ~4,600 |

**Splits (in `data/processed/`):**
- `train.csv` — ~71K rows (real + synthetic + augmented)
- `val.csv` — ~14K rows (real only)
- `test.csv` — 13,986 rows (real only — held out, never used in selection)
- `label_map.json` — category → integer mapping

**Critical design decision:** synthetic + augmented rows (`source` in `{synthetic, augmented_bt, chatgpt_synthetic, pseudo_labeled, eda_augmented}`) are excluded from val and test to prevent evaluation inflation from template memorization.

---

## 5. Data sources

| Source | Path | Rows | Why |
|--------|------|-----:|-----|
| `production_dataset.csv` | `C:\Users\Feras\Complaint-system\data\` | ~88,000 | Pre-existing labeled Arabic restaurant complaints (6 cats, renamed) |
| Google Play scrape | `data/raw/play_store_reviews.csv` | 14,789 unique | Saudi food delivery apps (HungerStation, Jahez, Mrsool, Talabat) — keyword-filtered into delivery + order accuracy |
| Template synthetic | `src/generate_synthetic.py` | ~5,000 | Boosts small categories. Train-only. |
| Multi-model pseudo-labels | `src/pseudo_label.py` | 1,771 | Top-2 ensemble agreement at ≥0.90 confidence on unused scrape. Train-only. |
| EDA augmentation | `src/eda_augment_classes.py` | ~3,000 | Targeted boost on دقة الطلب + عامة. Train-only. |
| ChatGPT synthetic (was for ambiance) | `data/raw/chatgpt_ambiance.txt` | 1,952 (now removed) | Used during 9-class era; dropped with ambiance category. |

See [DATA_CARD.md](DATA_CARD.md) for full provenance and the train-only enforcement rules.

---

## 6. Preprocessing applied (Lana's pipeline)

Applied to every row before saving:
1. Remove tashkeel (Arabic diacritics, U+064B–U+065F)
2. Normalize alef forms: أ إ آ ٱ → ا
3. Normalize ya: ى → ي
4. Normalize ta-marbuta: ة → ه
5. Strip punctuation/symbols (keeps Arabic letters, ASCII letters, digits both Arabic and Latin)
6. Lowercase ASCII
7. Collapse whitespace

---

## 7. Repo layout

```
complaint classifier/
├── PROJECT_HANDOFF.md       ← this file
├── README.md                ← portfolio summary + headline numbers
├── REPORT.md                ← full results, methodology, ablations
├── CHANGELOG.md             ← version history
├── CONTRIBUTING.md          ← team guide, current pipeline, CI
├── DEPLOYMENT.md            ← HF Spaces (primary) + alternatives
├── MODEL_CARD.md            ← intended use, limitations, ethics
├── DATA_CARD.md             ← provenance, biases, schema, train-only sources
├── SECURITY.md              ← vuln reporting + PII handling
├── NOTICES                  ← upstream model licenses
├── LICENSE                  ← MIT
├── pyproject.toml           ← packaging (pip install -e .)
├── requirements.txt         ← runtime deps (mirror of pyproject)
├── pytest.ini               ← test config (also in pyproject)
├── Dockerfile               ← multi-stage container build
├── .pre-commit-config.yaml + .editorconfig
├── data/
│   ├── text/                ← labeled CSVs (tracked)
│   ├── raw/                 ← scraped + ChatGPT-generated (gitignored)
│   ├── processed/           ← train/val/test + label_map + audit CSVs (gitignored)
│   └── canary/              ← cross-dialect canary set
├── src/                     ← all training + eval scripts (32 files)
│   ├── _archive/            ← superseded scripts (kept for git history)
│   ├── config.py            ← central constants
│   ├── inference_utils.py   ← shared predict_probs
│   └── ...                  ← see source for full list
├── app/
│   ├── __init__.py          ← package init
│   ├── ensemble_inference.py ← reusable EnsembleClassifier
│   ├── api.py               ← FastAPI: /, /healthz, /readyz, /metrics, /predict, /predict_batch
│   └── space_app.py         ← Gradio with Thmanyah typeface
├── tests/                   ← pytest suite (data, inference, properties, determinism, regression)
├── scripts/                 ← launch_demo, upload_to_hf, codex_baseline
├── hf_space/                ← turnkey HF Spaces deploy directory
├── assets/fonts/            ← Thmanyah typeface (3 families × 5 weights)
├── notebooks/               ← Colab notebooks per phase (kept for reference)
└── models/                  ← gitignored except ensemble_final/config.json
    ├── ensemble_final/      ← 4-model manifest + temperature.json
    ├── single_final/        ← single-model manifest (lighter deploy)
    ├── bakeoff/             ← trained model dirs (~2 GB total)
    └── error_analysis/      ← confusion matrix, errors.csv, report
```

---

## 8. Current results

### Headline (test set, 13,986 real reviews)

Deploy target: 4-model 8-class ensemble at `models/ensemble_final/`.

| Metric | Point | 95% CI |
|---|---:|---|
| Accuracy | 95.05% | [94.70%, 95.41%] |
| Weighted F1 | 95.08% | [94.72%, 95.43%] |
| Macro F1 | 92.03% | [91.20%, 92.87%] |
| Min class F1 (عامة) | 84.84% | [81.0%, 88.3%] |
| Calibration ECE (T=1.523) | 0.014 | — |
| Inference (ensemble GPU p50) | 18 ms | — |

All 8 categories ≥80% F1.

### Winning ensemble (`models/ensemble_final/config.json`)
1. `camelbert-mix_8c_capALL_s42_final` (seed=42)
2. `camelbert-mix_8c_capALL_s2024_v2_final` (seed=2024, with EDA)
3. `marbert_8c_capALL_v2_final` (with EDA)
4. `arabertv02_8c_v3_final`

Inference: uniform average of softmax probabilities (untuned biases performed better than tuned on this run — tuning overfit val). See [app/ensemble_inference.py](app/ensemble_inference.py).

### Per-class F1 on test (every class ≥84%)
| Category | F1 |
|---|---:|
| جودة الطعام | 96.2% |
| خدمة الموظفين | 95.8% |
| النظافة | 95.3% |
| السعر والقيمة | 94.7% |
| وقت الانتظار | 91.9% |
| التوصيل | 90.1% |
| دقة الطلب | 86.7% |
| عامة | 84.9% |

### Historical comparison

| Model | Test acc | Test wF1 | Test mF1 |
|---|---:|---:|---:|
| TF-IDF + LinearSVC (9-class baseline) | 89.8% | 89.8% | 68.0% |
| CAMeLBERT-mix (9-class single) | 93.2% | 93.2% | 76.7% |
| 4-model ensemble (9-class with ambiance) | 94.07% | 94.00% | 82.80% |
| 4-model ensemble (8-class, no EDA) | 94.86% | 94.89% | 90.50% |
| 4-model ensemble (8-class + EDA + temperature) | 95.05% | 95.08% | 92.03% |

### Biggest wins (by impact)

1. Dropping the ambiance category (+7.7 macro F1, +43 min class F1)
2. Real-data scaling — التوصيل 1k → 5.6k, دقة الطلب 137 → 714
3. 4-model ensemble with multi-arch + multi-seed diversity
4. Targeted EDA augmentation for the bottom 2 classes
5. No dominant-class subsampling (use all 31K food-quality rows)

### What didn't work
- Multi-label classification (keyword-derived labels too noisy)
- camelbert-da (worse than camelbert-mix — cleaning strips dialect markers)
- xlm-roberta-base (worse than Arabic-specific models)
- 5+ model ensembles (dilution beats diversity)
- Threshold tuning on the 8-class ensemble (overfit val)

Pipeline scripts (all deterministic, seeded). See REPORT.md §Reproducibility for the exact command sequence. Deploy steps in DEPLOYMENT.md and hf_space/HOW_TO_DEPLOY.md.

---

## 9. Hardware (Feras's desktop)

- CPU: Intel i5-13500
- GPU: NVIDIA RTX 4070 (12 GB VRAM)
- RAM: 32 GB
- Recommended for CAMeLBERT and any future BERT experiments

---

## 10. Decisions already made (don't relitigate)

- 8 categories (originally 9; ambiance dropped after audit found ~99% label noise)
- No commercial LLM at inference (fully local model)
- CAMeLBERT-mix + MARBERT + AraBERTv02 ensemble is the production target
- Synthetic / augmented / pseudo-labeled rows are train-only (never val/test)
- No dominant-class subsampling (uses all 31K food-quality rows)
- HuggingFace Spaces is the recommended demo deployment
- Clean commit messages, no co-author trailers
- Branch policy: work on main directly; push only when results are validated

---

## 11. What to do next

Manual deploy (needs HF token):
1. Upload the 4 models to HF Hub — see hf_space/HOW_TO_DEPLOY.md Step 1
2. Create HF Space, push hf_space/ contents — Steps 2-3
3. Add live URL to README

Ongoing:
- Retrain every 3-6 months on fresh scraped reviews (distribution drift)
- Run `pytest` before any push — the regression test (`tests/test_regression.py`) blocks PRs that drop test acc below 92%
- Pseudo-label fresh scrapes via `src/pseudo_label.py` to keep adding training data without manual labeling

---

## 12. Files NOT to modify by hand

- `data/text/complaints_labeled.csv` — generated by the pipeline scripts
- `data/processed/*` — generated by `src/split_dataset.py`
- `notebooks/02_text_cleaning.ipynb`, `03_labeling_splitting.ipynb` — Lana and Khowla's work, merged

## 13. Files safe to edit when iterating

- `app/ensemble_inference.py`, `app/api.py`, `app/space_app.py` — inference surface
- `src/generate_synthetic.py` — synthetic templates
- `src/run_bakeoff.py` — training entrypoint
- `models/ensemble_final/config.json` — ensemble manifest (which model dirs, what biases)
