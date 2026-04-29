# Model Card — Arabic Complaints Classifier

## Overview

A 9-class text classifier for Arabic restaurant complaints, trained from scratch using TF-IDF features and classical ML (Logistic Regression / Linear SVM). No pre-trained language models are used.

- **Domain:** Arabic restaurant and food-delivery complaints
- **Task:** Multi-class single-label classification
- **Output:** One of 9 categories + confidence
- **Model file:** `models/classifier.pkl` (after Phase 3)

## Categories

| ID | Category | Description |
|----|----------|-------------|
| 0 | التوصيل | Delivery |
| 1 | الجو والمكان | Ambiance |
| 2 | السعر والقيمة | Price/value |
| 3 | النظافة | Cleanliness |
| 4 | جودة الطعام | Food quality |
| 5 | خدمة الموظفين | Staff service |
| 6 | دقة الطلب | Order accuracy |
| 7 | عامة | General fallback |
| 8 | وقت الانتظار | Wait time |

## Architecture

```
Input text
  → TF-IDF word n-grams (1-2, 20K features)
  → TF-IDF char_wb n-grams (3-5, 20K features) [for Arabic morphology]
  → FeatureUnion
  → Classifier (LR or LinearSVC, class_weight='balanced')
  → Argmax over 9 categories
```

## Intended Use

- Restaurant owners categorizing customer feedback
- Food delivery platforms triaging complaints
- Researchers studying Arabic NLP

## Out of Scope

- Sentiment analysis (model assumes input IS a complaint)
- Languages other than Arabic
- Other domains (e.g. hotels, telecom) without retraining
- Real-time fraud detection or safety-critical decisions

## Training Data

- 88,531 real Arabic complaints across 9 categories
- 6,860 synthetic template-generated rows for the 4 smallest categories
- 95,391 rows total — see `DATA_CARD.md` for full provenance

Stratified 70/15/15 train/val/test. Synthetic rows are in **train only** — val and test are 100% real to avoid evaluation inflation.

## Evaluation

Reported metrics are computed on `data/processed/test.csv` (real data only):
- Overall **weighted F1**
- Per-class precision, recall, F1
- Confusion matrix

**Baseline:** majority class (جودة الطعام) → ~47% accuracy. The model must beat this.

**Realistic target:** weighted F1 ≥ 0.78 with current data + TF-IDF + balanced weights.

## Limitations

1. **Single-label:** complaints spanning multiple categories get one label
2. **Negative-only training:** positive reviews will be force-categorized
3. **Dialect bias:** strongest on Saudi/Gulf Arabic; weaker on North African dialects
4. **Synthetic templates:** ~50% of training data for التوصيل / دقة الطلب is template-generated and may not reflect natural language fully
5. **Small test set for new categories:** دقة الطلب has only 22 real test rows — F1 is high-variance on this class

## Ethical Considerations

- **No personal information** in training data — texts are public app reviews and aggregated complaint records
- **Fairness:** bias audit not yet performed across dialects
- **Misclassification cost is low** — the model only routes complaints, no automated decisions are made on its output

## Maintenance

- Retrain when new complaint patterns emerge (e.g. new delivery apps)
- Monitor F1 on production samples; alert if drift > 5%
- Owner: AI Club — see `CONTRIBUTING.md` for team
