# Arabic Restaurant Complaints Classifier

Classify Arabic restaurant complaints into 8 actionable categories. Saudi-Gulf dialect specialization, 95% test accuracy.

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

| Metric | Value |
|---|---:|
| Accuracy | **95.05%** |
| Macro F1 | 92.03% |
| Min class F1 | 84.84% |

Held-out test set, 13,986 real reviews. Every class ≥80% F1.

## Architecture

```
   ~98K Arabic complaints (production + Saudi delivery apps)
                        │
                        ▼
            ┌──────────────────────────┐
            │  Data pipeline   src/    │
            │  scrape · clean ·        │
            │  augment · split         │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  Bake-off       src/     │
            │  CAMeLBERT-mix · MARBERT │
            │  AraBERTv02 · XLM-R      │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  4-model ensemble        │
            │  uniform softmax avg     │
            │  + temperature scaling   │
            │  models/ensemble_final/  │
            └────────────┬─────────────┘
                         │
                         ▼
            ┌──────────────────────────┐
            │  Inference     app/      │
            │  EnsembleClassifier      │
            └────────────┬─────────────┘
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
        FastAPI      Gradio        Colab
        app/api.py   hf_space/     notebooks/
```

`src/` is the offline pipeline (data + training + eval). `app/` is the runtime (loads the ensemble, exposes `predict`). `hf_space/` is the live demo. `notebooks/colab_demo.ipynb` runs the model on free-tier Colab.

## Quick start

No GPU? Open the [Colab notebook](notebooks/colab_demo.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FerasMad/NLP-complaints-system/blob/main/notebooks/colab_demo.ipynb) — runs on the free tier.

Local install:

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -e .
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
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "الاكل بايخ"}'
```

Gradio (local + public *.gradio.live URL):

```bash
SHARE=true .venv/Scripts/python -m app.space_app
```

## Model

4-model ensemble of fine-tuned Arabic BERTs (CAMeLBERT-mix × 2 seeds, MARBERT, AraBERTv02). Uniform softmax average. Manifest: [`models/ensemble_final/config.json`](models/ensemble_final/config.json). Lighter single-model variant (~440 MB, ~94.86% acc): [`models/single_final/config.json`](models/single_final/config.json).

## Layout

```
app/        API + Gradio + EnsembleClassifier
src/        data + training + eval scripts
tests/      pytest suite
models/     ensemble manifest + label map
hf_space/   HuggingFace Spaces deploy bundle
scripts/    launch + upload helpers
notebooks/  Colab demo + team notebooks
```

## Team

Feras (lead, evaluation), Lana (text cleaning), Khowla (labeling/splitting), Rima + Mohammed (model training), Meshal (deployment).

## License

[MIT](LICENSE). Upstream model licenses in [NOTICES](NOTICES).

## Live demo

https://huggingface.co/spaces/FerasMad/arabic-complaints-classifier
