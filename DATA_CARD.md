# Data Card — Arabic Complaints Dataset

## Summary

95,391 labeled Arabic restaurant complaints across 9 categories, assembled from 4 sources and split into train/val/test for model training.

## Files

| Path | Rows | Description |
|------|------|-------------|
| `data/text/complaints_labeled.csv` | 95,391 | Master labeled dataset (cleaned) |
| `data/text/synthetic_complaints.csv` | 6,860 | Synthetic supplements (separate for audit) |
| `data/raw/play_store_reviews.csv` | 3,738 | Raw scraped reviews (1-2 star) |
| `data/processed/train.csv` | 68,826 | Training set (real + synthetic) |
| `data/processed/val.csv` | 13,277 | Validation (real only) |
| `data/processed/test.csv` | 13,288 | Test (real only) |
| `data/processed/label_map.json` | — | Category → integer encoding |

## Schema

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | Cleaned Arabic complaint text |
| `category` | string | One of 9 Arabic category names |
| `label` | int (0–8) | Encoded category (in processed/* only) |
| `priority` | string | عالية / متوسطة / منخفضة (auxiliary metadata, not used for training) |
| `source` | string | `production`, `play_store`, `res1`, `synthetic` |

## Sources

### 1. production_dataset.csv (88,478 cleaned rows)
- Origin: pre-existing Arabic restaurant complaint dataset, automatically labeled by keyword rules
- Categories: 6 (renamed to align with 9-category schema: الخدمة → خدمة الموظفين, التأخير → وقت الانتظار, السعر → السعر والقيمة)
- Quality: real text, automatic labels — likely contains some label noise
- License: internal

### 2. play_store_reviews.csv (1,008 رتب التوصيل + 137 دقة الطلب)
- Origin: 1-2 star Arabic reviews scraped from Saudi Google Play Store
- Apps: HungerStation, Jahez, Mrsool, Talabat
- Method: `google-play-scraper` with `lang='ar'`, `country='sa'`
- Date: 2026-04 (current scrape)
- License: public Google Play data

### 3. RES1.csv negative reviews (567 rows → الجو والمكان)
- Origin: hadyelsahar/large-arabic-sentiment-analysis-resources GitHub repo
- Filter: polarity = -1 AND ambiance keywords match
- Dialect: predominantly Lebanese/Levantine
- License: research use (CC BY)

### 4. synthetic_complaints.csv (6,860 rows)
- Origin: `src/generate_synthetic.py` template generator
- Distribution: التوصيل (1,492) + دقة الطلب (2,363) + الجو والمكان (1,933) + عامة (1,072)
- Purpose: balance the small categories so each class has ≥ 2,500 train+val samples
- All synthetic rows are flagged `source=synthetic` and EXCLUDED from val/test

## Final Distribution (95,391 rows)

| Category | Real | Synthetic | Total |
|----------|------|-----------|-------|
| جودة الطعام | 45,084 | 0 | 45,084 |
| السعر والقيمة | 17,391 | 0 | 17,391 |
| خدمة الموظفين | 15,242 | 0 | 15,242 |
| وقت الانتظار | 4,234 | 0 | 4,234 |
| النظافة | 3,440 | 0 | 3,440 |
| عامة | 1,428 | 1,072 | 2,500 |
| التوصيل | 1,008 | 1,492 | 2,500 |
| الجو والمكان | 567 | 1,933 | 2,500 |
| دقة الطلب | 137 | 2,363 | 2,500 |

## Preprocessing

Applied to every row before saving:
1. Remove tashkeel (Arabic diacritics, U+064B–U+065F)
2. Normalize alef forms: أ إ آ ٱ → ا
3. Normalize ya: ى → ي
4. Normalize ta-marbuta: ة → ه
5. Strip punctuation and symbols (keeps Arabic letters, digits — both Arabic and Latin — and ASCII letters)
6. Lowercase ASCII
7. Collapse whitespace

## Known Biases & Limitations

- **Class imbalance:** جودة الطعام is 47% of training data
- **Dialect:** Saudi/Gulf and Levantine dominant; underrepresents Egyptian and Maghrebi
- **Synthetic skew:** التوصيل / دقة الطلب train sets are >50% synthetic — model may overfit to template patterns
- **Real test counts are small for new categories:** دقة الطلب (22 real test rows), الجو والمكان (86) — F1 on these will have wide confidence intervals
- **All complaint, no positive samples:** model has no concept of "this is praise, not a complaint"

## Recommended Use

- For training Arabic complaint classifiers within similar restaurant/food-service domains
- Not recommended for sentiment analysis tasks (no positive class)
- Not recommended for non-Arabic text
