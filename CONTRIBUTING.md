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

## Member Tasks

### Feras — Leader + Data Collection + Evaluation

**Data Collection Goal:** Collect 150+ labeled Arabic complaint texts per category (1,200 total minimum).

**Data Collection Steps:**
1. Read `sample_complaints.csv` (shared on Drive) to understand the format and tone
2. Collect real or realistic Arabic complaints for each of the 8 categories
3. Save as `data/raw/complaints_raw.csv` with columns: `category_id`, `category_ar`, `text`
4. Upload the file to the shared Google Drive folder — do NOT commit raw data to GitHub

**Target per category:** 150+ rows
**Total target:** 1,200+ rows

**Evaluation Goal (after model is trained):** Report model performance on the held-out test set.

**Evaluation Steps:**
1. Load `data/processed/test.csv` and `models/classifier.pkl`
2. Generate predictions
3. Compute and report:
   - Overall accuracy
   - Weighted F1-score
   - Per-class precision, recall, F1 (`classification_report`)
   - Confusion matrix (plot with seaborn heatmap)
4. Identify the 2–3 weakest categories and document why

---

### Rima + Mohammed — Model Training

**Branch:** Each creates their own (`rima/model-training`, `mohammed/model-training`)

**Goal:** Train TF-IDF + classifier and save the best model.

**Pipeline (no pre-trained LLMs — TF-IDF only):**
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

pipeline_lr = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline_svm = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('clf', LinearSVC())
])
```

**Steps:**
1. Load `data/processed/train.csv` and `val.csv`
2. Train both pipelines on training set
3. Evaluate both on validation set using weighted F1
4. Document: which model won and by how much (val F1 difference)
5. Save best model: `import joblib; joblib.dump(best_model, 'models/classifier.pkl')`

---

### Lana + Khowla — Data Processing (Text Cleaning + Labeling & Splitting)

**Branch:** Each creates their own (`lana/data-processing`, `khowla/data-processing`)

**Lana — Text Cleaning:**

**Goal:** Clean and normalize raw Arabic complaint text.

**Steps:**
1. Load `data/raw/complaints_raw.csv`
2. Use CAMeL Tools to normalize and tokenize Arabic text:
   ```python
   from camel_tools.utils.normalize import normalize_unicode
   from camel_tools.tokenizers.word import simple_word_tokenize
   ```
3. Remove punctuation, extra whitespace, non-Arabic characters

**Khowla — Labeling & Splitting:**

**Goal:** Verify category assignments and encode labels, then create stratified train/val/test splits.

> **Labeling ownership note:** The data team (Lana + Khowla) owns ALL labeling decisions — both category assignment and integer encoding. The raw CSV from Feras includes `category_id` set by convention; Khowla verifies each row's category is correct before encoding.

**Steps:**
1. Encode `category_id` as integer labels (0–7)
2. Split: 70% train / 15% val / 15% test (stratified)
3. Save to `data/processed/`: `train.csv`, `val.csv`, `test.csv`

---

### Meshal — API Integration

**Branch:** `meshal/api-integration`

**Goal:** Build a REST API to serve the trained model.

**Steps:**
1. Load `models/classifier.pkl`
2. Build a REST API endpoint (Flask or FastAPI):
   - `POST /predict` — accepts `{ "text": "..." }`, returns `{ "category": "..." }`
3. Save as `app/api.py`

---

### All Members — Demo

**Collective task** (no dedicated branch — done on `main` after merge)

**Goal:** Run the Gradio interface together for the final presentation.

**Steps:**
1. Each member tests their own category predictions using the Gradio interface
2. Present results collectively

**Gradio interface (already implemented in `app/app.py`):**
```python
import gradio as gr
import joblib

model = joblib.load('models/classifier.pkl')
categories = ['خدمة العملاء', 'التوصيل والشحن', 'جودة المنتج',
              'الفواتير والدفع', 'المرتجعات والاسترداد',
              'الموقع والتطبيق', 'العروض والخصومات', 'التوصيل المتأخر']

def predict(text):
    pred = model.predict([text])[0]
    return categories[pred]

demo = gr.Interface(fn=predict, inputs='text', outputs='text',
                    title='Arabic Complaint Classifier')
demo.launch()
```

---

## What NOT to Commit

- Raw or processed data files (`data/`)
- Trained model files (`models/*.pkl`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

See `.gitignore` for the full list.
