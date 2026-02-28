# Contributing Guide

## Branch Naming

Each member works on their own branch:

```
member1/data-collection
member2/preprocessing
member3/model-training
member4/evaluation
member5/demo
```

Create your branch from `main`:
```bash
git checkout main
git pull origin main
git checkout -b member2/preprocessing
```

Open a Pull Request to `main` when your phase is complete.

---

## Member Tasks

### Member 1 — Data Collection

**Goal:** Collect 150+ labeled Arabic complaint texts per category (1,200 total minimum).

**Steps:**
1. Read `sample_complaints.csv` (shared on Drive) to understand the format and tone
2. Collect real or realistic Arabic complaints for each of the 8 categories
3. Save as `data/raw/complaints_raw.csv` with columns: `category_id`, `category_ar`, `text`
4. Upload the file to the shared Google Drive folder
5. Do NOT commit raw data to GitHub

**Target per category:** 150+ rows
**Total target:** 1,200+ rows

---

### Member 2 — Preprocessing

**Goal:** Clean Arabic text and prepare train/val/test splits.

**Steps:**
1. Load `data/raw/complaints_raw.csv`
2. Use CAMeL Tools to normalize and tokenize Arabic text:
   ```python
   from camel_tools.utils.normalize import normalize_unicode
   from camel_tools.tokenizers.word import simple_word_tokenize
   ```
3. Remove punctuation, extra whitespace, non-Arabic characters
4. Encode `category_id` as integer labels (0–7)
5. Split: 70% train / 15% val / 15% test (stratified)
6. Save to `data/processed/`: `train.csv`, `val.csv`, `test.csv`

---

### Member 3 — Model Training

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
4. Save best model: `import joblib; joblib.dump(best_model, 'models/classifier.pkl')`

---

### Member 4 — Evaluation

**Goal:** Report model performance on the held-out test set.

**Steps:**
1. Load `data/processed/test.csv` and `models/classifier.pkl`
2. Generate predictions
3. Compute and report:
   - Overall accuracy
   - Weighted F1-score
   - Per-class precision, recall, F1 (`classification_report`)
   - Confusion matrix (plot with seaborn heatmap)
4. Identify the 2–3 weakest categories and document why

---

### Member 5 — Demo (Gradio)

**Goal:** Build a simple Arabic text → predicted category web demo.

**Steps:**
1. Load `models/classifier.pkl`
2. Build Gradio interface:
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
3. Save as `app/app.py`

---

## What NOT to Commit

- Raw or processed data files (`data/`)
- Trained model files (`models/*.pkl`)
- Environment files (`.env`)
- Python cache (`__pycache__/`)

See `.gitignore` for the full list.
