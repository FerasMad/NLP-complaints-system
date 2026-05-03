# Data Card — Arabic Restaurant Complaints

~98,000 labeled Arabic restaurant complaints across 8 categories. Synthetic and augmented sources are train-only — never present in val/test. Evaluation is always on real data.

## Schema

| ID | Arabic | Description |
|----|--------|-------------|
| 0 | التوصيل | Delivery — late, missing, wrong address, driver issues |
| 1 | السعر والقيمة | Price and value |
| 2 | النظافة | Cleanliness, hygiene |
| 3 | جودة الطعام | Food quality — taste, freshness, portion |
| 4 | خدمة الموظفين | Staff service and behavior |
| 5 | دقة الطلب | Order accuracy — wrong/missing items |
| 6 | عامة | General fallback |
| 7 | وقت الانتظار | In-restaurant wait time |

> الجو والمكان (ambiance) was dropped after audit found only 2 of 171 val+test samples were truly clean ambiance. See [REPORT.md](REPORT.md) §"Why we dropped".

## Sources

| Source | Rows | Type | Where it goes |
|---|---:|---|---|
| `production_dataset.csv` | ~88,000 | Real (pre-existing labeled corpus) | train + val + test |
| Google Play scrape (HungerStation, Jahez, Mrsool, Talabat) | ~14,789 | Real (keyword-filtered into التوصيل + دقة الطلب) | train + val + test |
| Template-generated synthetic | ~5,000 | Synthetic | train only |
| MarianMT back-translation (Ar→En→Ar) — عامة | 1,410 | Synthetic | train only |
| Multi-model agreement pseudo-labels | 1,771 | Pseudo-labeled | train only |
| EDA augmented (دقة الطلب + عامة boost) | ~3,000 | Synthetic | train only |

## Splits

Stratified 70/15/15 on real samples only. Synthetic and augmented sources appended to train only.

| Split | Total | Notes |
|---|---:|---|
| Train | ~70,000 | Real + synthetic + augmented + pseudo-labeled |
| Val | ~13,976 | Real only |
| Test | ~13,986 | Real only, never used for selection until final eval |

Per-category test counts: جودة الطعام 6,764 · السعر والقيمة 2,610 · خدمة الموظفين 2,287 · التوصيل 850 · وقت الانتظار 636 · النظافة 516 · عامة 215 · دقة الطلب 108.

## Preprocessing

`clean()` in [src/rebuild_dataset.py](src/rebuild_dataset.py):

1. Remove tashkeel (U+064B–U+065F)
2. Normalize alef forms: أ إ آ ٱ → ا
3. Normalize ya: ى → ي
4. Normalize ta-marbuta: ة → ه
5. Strip punctuation/symbols (keep Arabic letters, ASCII letters, digits)
6. Lowercase ASCII
7. Collapse whitespace

## Source provenance

**`production_dataset.csv`** — pre-existing labeled corpus with 6 categories. Renamed: الخدمة → خدمة الموظفين, التأخير → وقت الانتظار, السعر → السعر والقيمة.

**Google Play scrape** ([src/scrape_reviews.py](src/scrape_reviews.py)) — 1–3 star Arabic reviews, NEWEST + MOST_RELEVANT sorts, paginated. 14,789 unique reviews. Keyword filters in [src/rebuild_dataset.py](src/rebuild_dataset.py): `DELIVERY_KW` (توصيل، مندوب، سائق، ديليفري…) and `ORDER_KW` (ناقص، غلط، نسوا، بدلوا، خلطوا…).

**Template synthetic** ([src/generate_synthetic.py](src/generate_synthetic.py)) — 60–80 dialectal templates per category, Saudi/Gulf markers (وش، ياليت، بس) and code-switching. `source="synthetic"`.

**Back-translation** ([src/augment_general.py](src/augment_general.py)) — Helsinki-NLP `opus-mt-ar-en` + `opus-mt-en-ar` paraphrases of real عامة rows. `source="augmented_bt"`.

**Pseudo-labels** ([src/pseudo_label.py](src/pseudo_label.py)) — top-2 bake-off models, both ≥0.90 confidence, agreed class. Limited to under-supported targets (التوصيل, دقة الطلب, عامة). 1,771 added. `source="pseudo_labeled"`.

**EDA augmentation** ([src/eda_augment_classes.py](src/eda_augment_classes.py)) — random word swap/delete/insert applied to دقة الطلب and عامة to lift their floor. `source="eda_augmented"`.

## Known biases

- **Saudi/Gulf dialect heavy.** Cross-dialect canary: Saudi 67%, Levantine 60%, Egyptian 50%, MSA 50%.
- **Class imbalance.** جودة الطعام is 47% of test data. Class-weighted loss compensates.
- **Synthetic templates are detectable.** Mitigated by excluding from val/test.
- **Tokenizer normalization is lossy.** Tashkeel and dialect-specific letter forms erased.
- **Pseudo-labels reflect model bias.** Mitigated by two-model agreement + ≥0.90 confidence + restricted to under-supported targets.

## Train-only sources

[src/split_dataset.py](src/split_dataset.py) excludes from val/test: `synthetic`, `augmented_bt`, `pseudo_labeled`, `eda_augmented`.

Real sources (`production`, `play_store`, `res1`) are split 70/15/15 stratified by category.

## Audit artifacts

- [data/processed/ambiance_audit.csv](data/processed/ambiance_audit.csv) — the 171-row audit that justified dropping ambiance
- [models/error_analysis/errors.csv](models/error_analysis/errors.csv) — every test error with heuristic classification
- [models/error_analysis/confusion_matrix.csv](models/error_analysis/confusion_matrix.csv) — full confusion matrix
