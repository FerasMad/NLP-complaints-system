# Ambience Data Collection Workflow

This document covers data acquisition for the v5 `الجو والمكان` experiment.
It does not change the 8-class production model or promote weak labels to
final labels.

## Workflow

1. Import raw reviews into a review queue with one of:
   - `src/data/import_arama_ambience.py` for manually downloaded AraMA/AraMAMS files.
   - `src/data/import_hf_datasets_for_ambience.py` for vetted Hugging Face review datasets.
   - `src/data/import_kaggle_or_local_reviews.py` for local Kaggle or other Arabic review files.
   - `src/data/collect_google_places_ambience.py` only when `GOOGLE_PLACES_API_KEY` is configured.
2. Each importer writes the same four files:
   - `ambience_candidates.csv`
   - `hard_negative_candidates.csv`
   - `review_needed.csv`
   - `source_summary.json`
3. Reviewers manually label the queues using the exact boundaries in
   [AMBIENCE_BOUNDARIES.md](AMBIENCE_BOUNDARIES.md).
4. Approved real rows go into `data/text/complaints_labeled_ambience.csv`.
   Keep this separate from the production `data/text/complaints_labeled.csv`.
5. Split only after manual review. Never put synthetic or weak unreviewed rows
   into validation or test.

Use the source ranking, license notes, and ethics guidance in
[AMBIENCE_DATA_SOURCES.md](AMBIENCE_DATA_SOURCES.md). Use the 19-column schema
in [DATA_SCHEMA_AMBIENCE.md](DATA_SCHEMA_AMBIENCE.md) for all queue files and
approved ambience data.

## Minimum Data Bar Before Training

Do not train a revived ambience model until the reviewed set has at least:

- 500 real approved ambience positives.
- 800 real hard negatives covering food, service, cleanliness, price, delivery,
  order accuracy, and wait time boundaries.
- 300 manually reviewed test rows.

Synthetic contrastive rows are useful for train-only robustness work, but they
must not enter validation or test.

## CLI Examples

AraMA/AraMAMS local import:

```bash
python src/data/import_arama_ambience.py \
  --input data/raw/arama_reviews.csv \
  --text-col text \
  --aspect-col aspect \
  --output-dir data/processed/ambience/arama
```

Hugging Face import:

```bash
python src/data/import_hf_datasets_for_ambience.py \
  --output-dir data/processed/ambience/huggingface
```

Generic Kaggle/local import:

```bash
python src/data/import_kaggle_or_local_reviews.py \
  --input data/raw/kaggle_reviews.csv \
  --source-name kaggle_arabic_reviews \
  --text-col review \
  --output-dir data/processed/ambience/kaggle
```

Optional Google Places API collector:

```bash
GOOGLE_PLACES_API_KEY=... python src/data/collect_google_places_ambience.py \
  --query "مطاعم الرياض" \
  --output-dir data/processed/ambience/google_places
```

If the API key is missing, the Google collector exits cleanly and prints:

```text
GOOGLE_PLACES_API_KEY not found. Skipping optional Google Places collector.
```

Google Places API returns only a limited number of reviews per place. Treat it
as a supplemental source, not a high-yield primary source.

## Manual Work Remaining

- Download AraMA/AraMAMS files locally and verify their license terms.
- Download Kaggle datasets locally and keep their license metadata.
- Configure a Google Places API key only if the team chooses to use paid API
  collection.
- Run the human labeling pass, including `review_needed.csv` discussion.
- Run `src/evaluation/evaluate_ambience.py` before making any accuracy claims.
