# Arabic Restaurant Complaints Classifier

Classify Arabic restaurant complaints into 8 actionable categories. Saudi-Gulf dialect specialization, 95% test accuracy, every category ≥80% F1.

## Categories

| ID | Arabic | English |
|----|--------|---------|
| 0 | التوصيل | Delivery |
| 1 | السعر والقيمة | Price / value |
| 2 | النظافة | Cleanliness |
| 3 | جودة الطعام | Food quality |
| 4 | خدمة الموظفين | Staff service |
| 5 | دقة الطلب | Order accuracy |
| 6 | عامة | General |
| 7 | وقت الانتظار | Wait time |

## Performance

Held-out test set, 13,986 real reviews:

| Metric | Value | 95% CI |
|---|---:|---|
| Accuracy | **95.05%** | [94.70%, 95.41%] |
| Weighted F1 | 95.08% | [94.72%, 95.43%] |
| Macro F1 | 92.03% | [91.20%, 92.87%] |
| Min class F1 | 84.84% | [81.0%, 88.3%] |

Per-class F1 (test):

| Category | F1 | | Category | F1 |
|---|---:|---|---|---:|
| جودة الطعام | 96.2% | | التوصيل | 90.1% |
| خدمة الموظفين | 95.8% | | دقة الطلب | 86.7% |
| النظافة | 95.3% | | عامة | 84.9% |
| السعر والقيمة | 94.7% | | | |
| وقت الانتظار | 91.9% | | | |

Full results, methodology, ablations: [REPORT.md](REPORT.md).

## Approach

TF-IDF baseline → fine-tuned BERT → multi-source data scaling → 4-model ensemble → drop a broken category → boost the floor → harden eval → deploy.

```
              ┌─────────────────────────────────────────┐
              │            DATA  (~98K rows)            │
              │  production · Play Store scrape         │
              │  synthetic templates · pseudo-labels    │
              │  EDA on دقة الطلب + عامة                │
              └────────────────┬────────────────────────┘
                               │
                        clean() pipeline
                  tashkeel · alef · ya · ta-marbuta
                               │
                  stratified 70/15/15 on REAL ONLY
                  synthetic + augmented → train only
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │                BAKE-OFF                 │
              │  CAMeLBERT-mix×2  CAMeLBERT-da  ✗       │
              │  MARBERT  AraBERTv02  XLM-R  ✗          │
              │             ↓ top 4 by val mF1          │
              └────────────────┬────────────────────────┘
                               │
                  ┌────────────┴────────────┐
                  │  drop ambiance          │
                  │  (audit: 2/171 clean)   │
                  └────────────┬────────────┘
                               │
                  ┌────────────┴────────────┐
                  │  EDA boost              │
                  │  دقة الطلب · عامة ≥ 85% │
                  └────────────┬────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │             4-MODEL ENSEMBLE            │
              │   uniform softmax average               │
              │   + temperature scaling (T = 1.523)     │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │                  EVAL                   │
              │  bootstrap CI · calibration · robustness│
              │  cross-dialect · per-source · per-class │
              ├─────────────────────────────────────────┤
              │  95.05% acc · 92.03% macro F1           │
              │  every class ≥ 80% F1 · ECE 0.014       │
              └────────────────┬────────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────────┐
              │                 DEPLOY                  │
              │  FastAPI · Gradio · HF Hub · HF Space   │
              └─────────────────────────────────────────┘
```

Step-by-step in [REPORT.md §Methodology](REPORT.md#methodology).

## Quick start

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -e .
# GPU users:
# .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

```python
from app.ensemble_inference import EnsembleClassifier

clf = EnsembleClassifier("models/ensemble_final/config.json")
result = clf.predict("الاكل بايخ ومالح")
print(result.category, result.confidence)
# جودة الطعام 0.99
```

REST API:

```bash
.venv/Scripts/python -m uvicorn app.api:app --port 8000
curl -X POST http://localhost:8000/predict \
    -H 'Content-Type: application/json' \
    -d '{"text": "الاكل بايخ"}'
```

Gradio (local + public *.gradio.live URL, 72h, no account):

```bash
SHARE=true .venv/Scripts/python -m app.space_app
```

UI is styled with the [Thmanyah typeface](https://thmanyah.com/).

## Model

4 fine-tuned Arabic BERTs averaged at inference:

- CAMeLBERT-mix (CAMeL-Lab) — 2 seeds for diversity
- MARBERT (UBC-NLP) — Twitter-trained, dialect-friendly
- AraBERTv02 (aubmindlab)

Ensemble manifest: [`models/ensemble_final/config.json`](models/ensemble_final/config.json).
Lighter single-model variant: [`models/single_final/config.json`](models/single_final/config.json) — ~440 MB, ~94.86% test acc.

## Layout

```
.
├── app/                  # API + Gradio + EnsembleClassifier
├── src/                  # data + training + eval scripts
├── tests/                # pytest suite
├── models/ensemble_final # ensemble manifest
├── data/                 # labeled CSVs + canary sets
├── hf_space/             # turnkey HuggingFace Spaces deploy directory
├── scripts/              # launch + upload helpers
└── docs (README, REPORT, MODEL_CARD, DATA_CARD, DEPLOYMENT, CHANGELOG)
```

## Team

Feras (lead, evaluation), Lana (text cleaning), Khowla (labeling/splitting), Rima + Mohammed (model training), Meshal (deployment).

## License

[MIT](LICENSE). Upstream model licenses in [NOTICES](NOTICES).

## Live demo

https://huggingface.co/spaces/<your-username>/arabic-complaints-classifier
