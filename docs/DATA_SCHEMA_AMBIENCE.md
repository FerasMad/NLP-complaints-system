# Ambience Data Schema

The columns we use for ambience-experiment data files. This is the canonical
schema for anything in `data/processed/ambience/` and for the outputs of
`src/filter_ambience_candidates.py` and `src/data/generate_ambience_synthetic.py`.

The 8-class production data files keep their existing columns
(`text, category, label, priority, source`) — this richer schema is for
the v5 experiment only.

## Columns

| Column | Required | Type | Notes |
|---|---|---|---|
| `id` | yes | int / str | Unique row id within the file. Filter outputs use the original input row index; synthetic outputs use `g{NNN}-pos` / `g{NNN}-neg`. |
| `text` | yes | str | The Arabic complaint, surface form (no normalization applied). Cleaning happens at training time. |
| `category` | yes | str | One of the 9-class schema. The deciding label. |
| `label` | yes once split | int | Integer label index from `label_map.json`. Filled in by `split_dataset.py --schema-version 9class_ambience`. |
| `source` | yes | str | One of the values in the table below. Determines whether the row can appear in val/test. |
| `source_detail` | no | str | Free-text refinement of `source` — e.g. `google_places_api/riyadh/2025-04`. |
| `city` | no | str | Saudi/Gulf city if known: `الرياض`, `جدة`, `الخبر`, etc. |
| `restaurant_name` | no | str | If known. Optional — most sources don't carry this. |
| `dialect` | no | str | One of: `saudi`, `gulf`, `msa`, `mixed`, `unknown`. Defaults to `unknown` if not set. |
| `weak_label` | no | str | The bucket the filter assigned: `ambience_candidate`, `hard_negative_candidate`, `review_needed`, `none`. NOT the final label. |
| `contains_ambience_keyword` | no | bool | True if the row matched any keyword in `src/config/ambience_keywords.py`. Filter sets this. |
| `ambience_subtype` | no | str | One of the 12 subtypes (see below). Empty string for non-ambience rows. |
| `risk_keyword` | no | str | `\| ` -separated risky-ambiguous words present (e.g. `بارد \| المطعم`). |
| `is_hard_negative` | no | bool | True if this row is a confirmed non-ambience training example. |
| `is_synthetic` | no | bool | True if `source` ∈ {`chatgpt_synthetic`, `ambience_synthetic`, `ambience_eda_augmented`, `synthetic`, `augmented_bt`, `eda_augmented`, `pseudo_labeled`}. Computed; not required at write time. |
| `label_confidence` | no | str | One of: `manual_high`, `manual_medium`, `weak`, `synthetic_reviewed`, `synthetic_unreviewed`. Empty for un-labeled rows. |
| `review_status` | no | str | One of: `unreviewed`, `reviewed`, `approved`, `rejected`, `needs_discussion`. Filter outputs default to `unreviewed`. |
| `notes` | no | str | Free text. Filter outputs put the bucket-decision reason here; labelers append observations. |
| `split` | no once split | str | One of: `train`, `val`, `test`, `unassigned`. Filled in by the splitter. |

## Allowed `source` values

| Source | Where it can appear | Description |
|---|---|---|
| `manual` | train, val, test | Hand-written or hand-labeled by the team. Highest trust. |
| `production` | train, val, test | Real reviews from the production 8-class data, repurposed. |
| `google_places_api` | train, val, test | Google Places API responses, post-filter and post-review. |
| `arama` | train, val, test | AraMA / AraMAMS academic seed data. |
| `play_store` | train, val, test | App-store reviews — usually weak ambience signal. |
| `hf_arabic_reviews` | train, val, test | HuggingFace Arabic restaurant-review datasets. |
| `kaggle_arabic_reviews` | train, val, test | Kaggle restaurant datasets. |
| `chatgpt_synthetic` | **train only** | Generic synthetic generation — NOT used in this project's ambience generator (kept for compatibility). |
| `ambience_synthetic` | **train only** | Output of `src/data/generate_ambience_synthetic.py` (contrastive pairs). |
| `ambience_eda_augmented` | **train only** | Output of `src/augment_ambience.py` (light perturbations of real ambience rows). |
| `synthetic`, `augmented_bt`, `pseudo_labeled`, `eda_augmented` | **train only** | Pre-existing 8-class train-only sources, gated by the leakage test. |

The leakage gate (`tests/test_data_pipeline.py::test_train_only_sources_not_in_val_test`)
enforces the "train only" rows. Adding a new train-only source means adding it
to the `train_only` set in that test as well.

## Allowed `ambience_subtype` values

The 12 subtypes from `BOUNDARIES.md`:

```
temperature_ac        seating_comfort       space_crowding
noise_music           lighting              smell
decor_furniture       parking               bathroom_facilities
outdoor_view          privacy               general_place_vibe
```

Subtypes appear only on rows where `category == "الجو والمكان"`. For all
other categories, `ambience_subtype` is the empty string.

## Example rows

A filter output (`ambience_candidates.csv`):

| id | text | weak_label | ambience_subtype | risk_keyword | review_status | notes |
|---|---|---|---|---|---|---|
| 42 | المكان حار والمكيف ما يبرد | ambience_candidate | temperature_ac | حار | unreviewed | ambience phrase/token present, no hard-negative cue |

A filter output (`hard_negative_candidates.csv`):

| id | text | weak_label | ambience_subtype | risk_keyword | hard_negative_hits | notes |
|---|---|---|---|---|---|---|
| 89 | الطاولة وسخه فيها بقع طعام | hard_negative_candidate | seating_comfort | الطاوله | بقع \| بقع طعام | multi-word hard-negative phrase present — confirmed non-ambience |

A synthetic-generator row (`ambience_synthetic_v1.csv`):

| text | category | ambience_subtype | source | is_hard_negative | risk_keyword | contrast_group_id | quality_score | notes |
|---|---|---|---|---|---|---|---|---|
| المكان بارد والمكيف قوي ما نقدر نقعد | الجو والمكان | temperature_ac | ambience_synthetic | False | بارد | g001 | 5 | ambience positive — place temp vs food temp |

## File-level conventions

- Always UTF-8. Output files use `utf-8-sig` (BOM) so they open correctly in
  Excel for hand-labeling.
- Multi-value fields use ` | ` (space-pipe-space) as the separator, never
  comma (which would break CSV parsing).
- Booleans serialize as `True`/`False` (Python's default), not `1`/`0`.
- Empty optional values are the empty string, not `null` / `NaN`.
