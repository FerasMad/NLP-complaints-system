# Contributing Guide

## Getting Started

All work is done in **Google Colab** — no local Python installation needed.

### 1. Clone the repository

```bash
git clone https://github.com/FerasMad/NLP-complaints-system.git
cd NLP-complaints-system
```

If you have already cloned it before, just pull the latest changes:
```bash
git pull origin main
```

### 2. Open the setup notebook in Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub**, paste the repo URL
3. Open `notebooks/00_setup.ipynb` and run all cells
4. Run `00_setup.ipynb` **once at the start of every Colab session**

### 3. Create your branch

```bash
git checkout main
git pull origin main
git checkout -b your-name/your-task
git push -u origin your-name/your-task
```

### 4. Submit your work

When your phase is complete, open a Pull Request on GitHub from your branch into `main` and tag the team lead.

---

## Branch Naming

```
feras/evaluation
lana/data-processing
khowla/data-processing
rima/model-training
mohammed/model-training
meshal/api-integration
```

---

## Categories (9)

The model classifies Arabic complaints into 9 categories:

| ID | Category (Arabic) | Description |
|----|-------------------|-------------|
| 0 | التوصيل | Delivery: late, missing, wrong address, driver issues |
| 1 | الجو والمكان | Ambiance: noise, seating, decor, parking, atmosphere |
| 2 | السعر والقيمة | Price/value complaints |
| 3 | النظافة | Cleanliness, hygiene, dirty environment |
| 4 | جودة الطعام | Food quality: taste, freshness, portion |
| 5 | خدمة الموظفين | Staff behavior, attitude, professionalism |
| 6 | دقة الطلب | Order accuracy: wrong/missing items |
| 7 | عامة | General fallback |
| 8 | وقت الانتظار | In-restaurant wait time, slow kitchen |

---

## Phase Status

| Phase | Owner | Status |
|-------|-------|--------|
| 0 — Setup | All | Done |
| 1 — Text cleaning | Lana | Done (PR #1 merged) |
| 2 — Labeling & splitting | Khowla | Done (data ready) |
| 3 — Model training | Rima + Mohammed | **Current** |
| 4 — Evaluation | Feras | Blocked on Phase 3 |
| 5 — Gradio demo | All | Blocked on Phase 4 |
| 6 — Deployment | Meshal + Feras | Blocked on Phase 5 |

---

## Phase 1 — Text Cleaning (Lana) — DONE

**Notebook:** `notebooks/02_text_cleaning.ipynb`
**Branch:** `lana/data-processing` (merged)

The cleaning pipeline applies:
1. Remove tashkeel (Arabic diacritics)
2. Normalize alef forms (أ إ آ ٱ → ا)
3. Normalize ya (ى → ي) and ta-marbuta (ة → ه)
4. Remove punctuation and non-Arabic characters
5. Collapse whitespace

Output: `complaints_labeled_clean.csv` on shared Drive.

---

## Phase 2 — Labeling & Splitting (Khowla) — DONE

**Notebook:** `notebooks/03_labeling_splitting.ipynb`
**Branch:** `khowla/data-processing`

The dataset has been built and split. Inputs are in `data/processed/`:
- `train.csv` — 66,735 rows (70%)
- `val.csv` — 14,298 rows (15%)
- `test.csv` — 14,307 rows (15%)
- `label_map.json` — category → integer mapping

Each split row has columns: `text, category, label, priority`

**Per-category counts in the full dataset (95,340 rows):**

| Category | Count |
|---|---|
| جودة الطعام | 45,058 |
| السعر والقيمة | 17,378 |
| خدمة الموظفين | 15,234 |
| وقت الانتظار | 4,231 |
| النظافة | 3,439 |
| الجو والمكان | 2,500 |
| عامة | 2,500 |
| التوصيل | 2,500 |
| دقة الطلب | 2,500 |

---

## Phase 3 — Model Training (Rima + Mohammed)

**Notebooks:** `notebooks/04_model_rima.ipynb`, `notebooks/04_model_mohammed.ipynb`
**Branches:** `rima/model-training`, `mohammed/model-training`
**Inputs:** `data/processed/train.csv`, `val.csv`

**Pipeline (TF-IDF only — no pre-trained LLMs):**

Use a `FeatureUnion` of word-level AND character-level n-grams. Character n-grams capture Arabic morphology (الأكل / الأكلة / أكل share roots).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion

features = FeatureUnion([
    ('word', TfidfVectorizer(max_features=20000, ngram_range=(1, 2), analyzer='word')),
    ('char', TfidfVectorizer(max_features=20000, ngram_range=(3, 5), analyzer='char_wb')),
])

pipeline_lr = Pipeline([
    ('feat', features),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])

pipeline_svm = Pipeline([
    ('feat', features),
    ('clf', LinearSVC(class_weight='balanced')),
])
```

**Steps:**
1. Load `train.csv` (col: `text`, `label`) and `val.csv`
2. Compute baseline: majority class accuracy = ~47% — your model must beat this
3. Train both pipelines on `train['text']`, `train['label']`
4. Evaluate both on val using **weighted F1**
5. Document which model won and by how much
6. Save the best one: `joblib.dump(best_pipeline, 'models/classifier.pkl')`

**Done when:** `classifier.pkl` exists on shared Drive, val weighted F1 ≥ 0.70, val baseline comparison documented.

**Notes for Rima + Mohammed:**
- `train.csv` includes synthetic rows (~7%). `val.csv` and `test.csv` are 100% real data — synthetic is excluded from evaluation by design (avoids inflated scores from template memorization)
- Use `LogisticRegression` (not LinearSVC) if you need probabilities for confidence scores — Phase 5 demo uses them

---

## Phase 4 — Evaluation (Feras)

**Notebook:** `notebooks/05_evaluation.ipynb`
**Branch:** `feras/evaluation`
**Inputs:** `data/processed/test.csv`, `models/classifier.pkl`

**Steps:**
1. Load test set and trained model
2. Generate predictions
3. Compute and report:
   - Overall accuracy
   - Weighted F1-score
   - Per-class precision, recall, F1 (`classification_report`)
   - Confusion matrix (seaborn heatmap)
4. Identify the 2–3 weakest categories and explain why

---

## Phase 5 — Gradio Demo (All)

**File:** `app/app.py`
**Steps:** Build a Gradio `Interface` that loads `classifier.pkl` and classifies new Arabic complaints. Output **top-3 categories with confidence scores** — better demo UX than a single label, and exposes uncertainty when the model is unsure.

```python
import gradio as gr
import joblib

model = joblib.load('models/classifier.pkl')
CATEGORIES = ['التوصيل', 'الجو والمكان', 'السعر والقيمة', 'النظافة',
              'جودة الطعام', 'خدمة الموظفين', 'دقة الطلب', 'عامة', 'وقت الانتظار']

def predict(text):
    if not text or len(text.strip()) < 3:
        return {}
    probs = model.predict_proba([text])[0]
    # return top-3 categories with confidence
    pairs = sorted(zip(CATEGORIES, probs), key=lambda x: -x[1])[:3]
    return {cat: float(score) for cat, score in pairs}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label='شكوى العميل', rtl=True),
    outputs=gr.Label(num_top_classes=3, label='التصنيف'),
    title='تصنيف الشكاوى — Arabic Complaint Classifier',
    description='ادخل شكوى عربية وسيتم تصنيفها الى احدى الفئات التسع',
)
demo.launch()
```

**Confidence threshold:** if top prediction < 0.30, default to عامة (low-confidence fallback). Configurable in `app/app.py`.

---

## Phase 6 — Deployment (Meshal + Feras)

**Goal:** Public website where anyone can type an Arabic complaint and receive a category.

**Architecture:**
- `app/api.py` — FastAPI service (`POST /predict`), deployed as Render Web Service
- `app/app.py` — Gradio interface that calls the live FastAPI endpoint, deployed as second Render Web Service

**Meshal's steps:**
1. Complete `app/api.py`:
   - Load `models/classifier.pkl` at startup
   - Expose `POST /predict` — accepts `{"text": "..."}`, returns `{"category": "...", "confidence": 0.XX}`
   - Add CORS middleware so Gradio frontend can call it
2. Complete `app/app.py`:
   - Gradio Interface that posts to live API URL (not local)
   - RTL layout hint for Arabic input
3. Create `render.yaml` blueprint defining both services
4. Upload `models/classifier.pkl` as Render persistent disk or GitHub release asset
5. Deploy both services on Render.com free tier
6. Verify end-to-end: Arabic complaint → API returns correct category

**Done when:** A public URL exists where anyone can classify Arabic complaints and the services stay running.

---

## Known Issues & Design Decisions

### Single-label classification
A complaint can span multiple categories ("الأكل بارد والتوصيل تأخر" hits 3). The current model picks the dominant theme. Multi-label support is planned for v2.

### Category overlap
Some categories have natural overlap:
- **التوصيل** vs **وقت الانتظار** — both about time, but waiting *in restaurant* vs delivery
- **خدمة الموظفين** vs **دقة الطلب** — staff might cause order errors
- **النظافة** vs **الجو والمكان** — both touch the physical space

The model handles this by learning context. Per-class F1 reflects difficulty.

### Synthetic data
~7% of training rows are template-generated (for the 4 small categories). To prevent inflated scores, val/test sets are 100% real data. Synthetic templates are visible in `data/text/synthetic_complaints.csv` for audit.

### Dialect bias
- production_dataset: KSA/Gulf dialect dominant
- RES1 ambiance data: Lebanese/Levantine dialect
- Scraped Play Store data: Saudi
- Synthetic templates: MSA + Gulf-leaning

Model may underperform on Egyptian/Moroccan Arabic.

### Negative-only training
All training data are complaints. The model will misclassify positive reviews into a complaint category. A "not-a-complaint" detector is recommended for production deployment.

---

## What NOT to Commit

- Raw or processed data files (`data/`)
- Trained model files (`models/*.pkl`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

See `.gitignore`.
