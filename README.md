# Arabic NLP Complaints Classification System

## Goal

Build a machine learning pipeline that classifies Arabic customer complaints into 9 actionable categories. The system is trained entirely on data collected and processed by the team — no pre-trained language models are used.

## Model Approach

**TF-IDF + Classical ML Classifier (no LLMs, no AraBERT)**

- Text vectorization: TF-IDF with Arabic-aware tokenization (CAMeL Tools)
- Classifiers: Logistic Regression and SVM — best model selected by validation weighted F1
- Class imbalance handled via `class_weight='balanced'`
- Evaluation: Accuracy, Weighted F1, per-class report, Confusion Matrix

## Categories

| ID | Arabic | Description |
|----|--------|-------------|
| 0 | التوصيل | Delivery: late, missing, wrong address, driver issues |
| 1 | الجو والمكان | Ambiance: noise, seating, decor, parking |
| 2 | السعر والقيمة | Price/value complaints |
| 3 | النظافة | Cleanliness, hygiene |
| 4 | جودة الطعام | Food quality: taste, freshness, portion |
| 5 | خدمة الموظفين | Staff behavior, attitude |
| 6 | دقة الطلب | Order accuracy: wrong/missing items |
| 7 | عامة | General fallback |
| 8 | وقت الانتظار | In-restaurant wait time |

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| scikit-learn | TF-IDF, classifiers, evaluation |
| CAMeL Tools | Arabic text normalization and tokenization |
| pandas / numpy | Data handling |
| google-play-scraper | Arabic review scraping (Saudi food delivery apps) |
| Gradio | Demo web interface |
| FastAPI + uvicorn | REST API serving the trained model |
| Render.com | Cloud hosting |

## Team Roles

| Member | Role |
|--------|------|
| Feras | Leader + Evaluation |
| Lana | Text Cleaning |
| Khowla | Labeling & Splitting |
| Rima | Model Training |
| Mohammed | Model Training |
| Meshal | API Integration + Deployment |

## Dataset

**95,340 labeled Arabic complaint texts** across 9 categories.

Sources:
- 88,415 rows from a production Arabic restaurant complaint dataset
- 1,008 scraped from Saudi food delivery apps (HungerStation, Jahez, Mrsool, Talabat) for التوصيل
- 565 from large-arabic-sentiment RES1 negative reviews for الجو والمكان
- 6,925 synthetic template-generated rows to balance the small categories

Splits (stratified 70/15/15):
- `data/processed/train.csv` — 66,735 rows
- `data/processed/val.csv` — 14,298 rows
- `data/processed/test.csv` — 14,307 rows

## Folder Structure

```
NLP-complaint-system/
├── data/
│   ├── text/              # Source CSVs (labeled, unlabeled, synthetic)
│   ├── raw/               # Scraped Play Store reviews
│   └── processed/         # Cleaned + split train/val/test + label_map.json
├── models/                # classifier.pkl after training
├── notebooks/             # Per-phase Colab notebooks
├── src/                   # Data preparation scripts
│   ├── scrape_reviews.py
│   ├── build_dataset.py
│   ├── generate_synthetic.py
│   └── split_dataset.py
├── app/                   # Gradio demo + FastAPI service
├── README.md
├── CONTRIBUTING.md
└── requirements.txt
```

## Status

**Phases 0–2 complete.** Phase 3 (model training) is current.
