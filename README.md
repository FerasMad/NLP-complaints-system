# Arabic NLP Complaints Classification System

## Goal

Build a machine learning pipeline that classifies Arabic customer complaints into a ist predefined categories. The system is trained entirely on data collected by the team — no pre-trained language models are used.

## Model Approach

**TF-IDF + Classical ML Classifier (no LLMs, no AraBERT)**

- Text vectorization: TF-IDF (with Arabic-aware tokenization via CAMeL Tools)
- Classifiers: Logistic Regression and SVM — best model selected by validation F1
- Evaluation: Accuracy, Weighted F1-score, Confusion Matrix

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| scikit-learn | TF-IDF vectorizer, classifiers, evaluation |
| CAMeL Tools | Arabic text normalization and tokenization |
| pandas / numpy | Data handling |
| Gradio | Demo web interface |


## Team Roles

| Member | Role |
|---|---|
| Feras | Leader + Data Collection + Evaluation |
| Lana | Data Processing — Text Cleaning |
| Khowla | Data Processing — Labeling & Splitting |
| Rima | Model Training |
| Mohammed | Model Training |
| Meshal | API Integration |

## Timeline (Phases)

- **Phase 0** (current): Project setup, team onboarding, data reference
- **Phase 1**: Data collection — 150+ complaints per category (1,200 total)
- **Phase 2**: Preprocessing — cleaning, normalization, train/val/test split
- **Phase 3**: Model training — TF-IDF + LR/SVM, save best model
- **Phase 4**: Evaluation — metrics, confusion matrix, weak category report
- **Phase 5**: Demo — Gradio interface

## Folder Structure (will be populated in later phases)

```
NLP-complaint-system/
├── data/
│   ├── raw/          # Raw collected complaints (not committed)
│   └── processed/    # Cleaned, encoded, split data (not committed)
├── models/           # Saved .pkl model files (not committed)
├── notebooks/        # Per-member Jupyter notebooks (added per phase)
├── src/              # Python scripts (added per phase)
├── app/              # Gradio app (Phase 5)
├── README.md
├── CONTRIBUTING.md
└── .gitignore
```
