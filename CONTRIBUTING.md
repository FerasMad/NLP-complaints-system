# Contributing Guide

## Getting Started

Follow these steps **once** to set up your environment before doing anything else.

### 1. Clone the repository

Open a terminal (or Git Bash on Windows) and run:

```bash
git clone https://github.com/FerasMad/NLP-complaints-system.git
cd NLP-complaints-system
```

If you have already cloned it before, just pull the latest changes instead:

```bash
git pull origin main
```

### 2. Open the setup notebook in Google Colab

All work is done in **Google Colab** — no local Python installation is needed.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Open notebook → GitHub**
3. Paste the repo URL: `https://github.com/FerasMad/NLP-complaints-system`
4. Open `notebooks/00_setup.ipynb`
5. Run all cells top-to-bottom — this mounts your Drive, installs dependencies, and pulls the repo

> Run `00_setup.ipynb` **once at the start of every Colab session**.

### 3. Create your branch

After cloning, create your personal branch (replace with your own branch name from the table below):

```bash
git checkout main
git pull origin main
git checkout -b lana/data-processing
```

Push your branch to GitHub so others can see it:

```bash
git push -u origin lana/data-processing
```

### 4. Open your assigned notebook

Each member has a dedicated notebook in the `notebooks/` folder. Open it in Colab the same way as step 2.

### 5. Submit your work

When your task is complete, open a **Pull Request** on GitHub from your branch into `main`. Tag the team lead for review.

---

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

## Phase 2 — Data Processing

### Lana — Text Cleaning

**Notebook:** `notebooks/02_text_cleaning.ipynb`
**Branch:** `lana/data-processing`

**Steps:**
1. Run `00_setup.ipynb` first (once per session) to mount Drive and pull the repo
2. Open `02_text_cleaning.ipynb` and run all cells top-to-bottom
3. The notebook loads `data/text/complaints_labeled.csv` and `data/text/complaints_unlabeled.csv` from the repo
4. It applies Arabic normalization (tashkeel removal, alef/ya/ta-marbuta normalisation, punctuation removal)
5. Saves two cleaned CSVs to your Drive: `data/processed/complaints_labeled_clean.csv` and `data/processed/complaints_unlabeled_clean.csv`
6. Commit and push **only the notebook** on your branch (`lana/data-processing`) — never commit the CSV files

**Done when:** `complaints_labeled_clean.csv` and `complaints_unlabeled_clean.csv` exist in the shared Drive folder.

---

### Khowla — Labeling & Splitting

**Notebook:** `notebooks/03_labeling_splitting.ipynb`
**Branch:** `khowla/data-processing`
**Requires:** Lana's output in `data/processed/` (coordinate with Lana first)

**Steps:**
1. Run `00_setup.ipynb` first (once per session)
2. Open `03_labeling_splitting.ipynb` and run all cells top-to-bottom
3. The notebook loads `complaints_labeled_clean.csv` from the shared Drive
4. Displays label distribution — confirm all 8 categories are present and roughly balanced
5. Encodes category names → integers (0–7), saves `label_map.json`
6. Performs a stratified 70/15/15 train/val/test split
7. Saves `train.csv`, `val.csv`, `test.csv` to `data/processed/` on Drive
8. Commit and push **only the notebook** on your branch (`khowla/data-processing`)

**Done when:** `train.csv`, `val.csv`, `test.csv`, and `label_map.json` exist in the shared Drive folder.

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
