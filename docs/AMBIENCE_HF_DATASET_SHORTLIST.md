# HuggingFace Datasets Shortlist for Ambience Data

Pre-vetted candidates plus auto-discovered hits. **None of these are downloaded automatically.** Read each dataset's card, check the license, and sample a few rows in a notebook before committing disk space.

## Recommendation

Don't lean on HF datasets as the bulk source. The realistic order of value (high to low):

1. **Hand-written by the team** — ~240 rows, zero labeling noise, 1 day of work. Best ROI.
2. **Google Places API** — filtered Arabic restaurant reviews, free up to ~100k req/mo, official terms-compliant access. Largest potential volume.
3. **AraMA / AraMAMS academic seed** — small (~hundreds) but already aspect-tagged. Use as gold seed, not bulk.
4. **HF datasets** — useful for MSA / cross-dialect canary sets and for general Arabic sentiment grounding, but ambience signal is almost always weak.

## Shortlist

| Dataset | Language | License | Ambience signal | Notes |
|---|---|---|---|---|
| `arbml/Arabic_News_Restaurant_Reviews` | ar | unknown — check card | low | Arabic news/restaurant reviews. Aspect labels not present, so you'd need to filter ambience candidates yourself with src/filter_ambience_candidates.py. |
| `Hello-SimpleAI/HC3-Arabic` | ar | CC BY-SA 4.0 | low | Generic Q&A in Arabic; not restaurant-specific. Skip for ambience. |
| `ajgt_twitter_ar` | ar | MIT | low | Arabic Jordanian Twitter sentiment. Useful as Levantine canary if you want cross-dialect testing, but no aspect labels. |
| `labr` | ar | see card | low | Large Arabic Book Reviews (LABR). Restaurant signal absent. Useful for general MSA polarity if you want a separate canary. |
| `FBK/AraSenTi` | ar | see card | low | MSA sentiment; few restaurant rows. Skip. |
| `BRAD` | ar | see card | low | Book reviews. Skip for ambience. |
| `google-research-datasets/poem_sentiment` | multi | Apache-2.0 | none | Skip. |
| `(not on HF) Google Places API` | ar (filterable) | Google Maps Platform terms | high | Single best source for in-restaurant reviews. Filter with lang=ar + city=Riyadh/Jeddah/Dammam, then run src/filter_ambience_candidates.py. Free up to ~100k requests/mo. |
| `(not on HF) AraMA / AraMAMS` | ar | research-only (check papers) | high | Arabic Multi-Aspect ABSA dataset(s). The 'environment' aspect is the closest to ambience. Dataset is small (likely <1k ambience rows) — use as gold seed, not bulk. |
| `(your team) hand-written` | ar (Saudi/Gulf) | MIT (your repo) | highest | Each labeler writes 10-15 examples per ambience subtype using BOUNDARIES.md as the rulebook. ~240 rows for 1 day of team work. No labeling noise. |

## How to evaluate a candidate dataset

1. Read the dataset card on HF Hub. Confirm license allows your use.
2. Sample 100-500 rows: `datasets.load_dataset('hub_id', split='train[:500]')`.
3. Run `src/filter_ambience_candidates.py` over the sample. If ambience-bucket size is < 5% of the sample, the dataset is too noisy to be worth full download.
4. If the sample looks promising, download the full dataset to `data/raw/hf_<dataset_name>/` and run the filter again.
5. Hand-label the candidates that pass the filter using `docs/AMBIENCE_BOUNDARIES.md`.