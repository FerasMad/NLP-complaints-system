---
language:
  - ar
license: mit
pretty_name: Arabic Restaurant Ambience Annotations
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
tags:
  - arabic
  - nlp
  - ambience
  - restaurant-reviews
  - saudi
  - gulf
  - dialectal-arabic
  - aspect-extraction
  - weak-supervision
---

# Arabic Restaurant Ambience Annotations

A dataset for detecting **ambience / physical-place complaints** in Arabic restaurant reviews — the kind of complaint that's about the room itself (AC, seating, noise, smell, decor, parking) rather than the food, staff, delivery, or price.

Built as part of the [Arabic Restaurant Complaints Classifier](https://github.com/FerasMad/NLP-complaints-system) project. Released because there's almost no public Arabic ambience-labeled data, and we wanted to fix that.

## What's in here

| File | Rows | What it is | License |
|---|---:|---|---|
| `synthetic_contrastive_pairs.csv` | 250 | 125 hand-written contrastive pairs across 12 ambience subtypes. Each pair shares a "risky" word (بارد, حار, قديم, ريحة, زيت) but one row is ambience and the other is the matching non-ambience hard-negative. | MIT (ours) |
| `adversarial_test_set.csv` | 183 | Hand-written adversarial test cases across 13 attack types: clean, boundary, negation, sarcasm, mixed dialect, multi-aspect, very_short, long, out_of_domain, typo, emoji, adversarial. | MIT (ours) |
| `boundary_examples.csv` | 90 | Hand-written boundary cases that map to the canonical labeling rules in `docs/BOUNDARIES.md`. | MIT (ours) |
| `ambience_keywords.json` | – | Saudi/Gulf ambience keyword vocabulary across 12 subtypes + risky-ambiguous tokens + 7 hard-negative keyword sets (food, service, delivery, price, order, hygiene, wait). | MIT (ours) |
| `annotations_qaym_*.csv` | 3,882 total | Filter outputs on [hadyelsahar/ar_res_reviews](https://huggingface.co/datasets/hadyelsahar/ar_res_reviews). **Annotations only — no source text.** Re-join with the source dataset by `id`. | Annotations: MIT (ours). Source text: see hadyelsahar's dataset card. |
| `annotations_hard_*.csv` | 41,752 total | Filter outputs on [HARD](https://huggingface.co/datasets/HARD) (hotel reviews — useful for transfer to restaurant ambience). **Annotations only — no source text.** Re-join by `id`. | Annotations: MIT (ours). Source text: see HARD's dataset card. |
| `complaints_labeled.csv` | **95,391** | **The full production training dataset.** Text + 8-class labels + source + priority. The actual data the production model trained on. Sources: Saudi delivery-app reviews (`production`, ~87K), team-generated synthetic templates (`synthetic`, ~7K), Play Store reviews (`play_store`, ~1.1K), one external source (`res1`, ~570). Released for **research / academic / non-commercial use** only — the labels are 100% the AI Club NLP team's manual work; the underlying review text comes from public Saudi delivery-app reviews. If you need commercial use of the source text, contact the original platforms directly. | Labels + curation: MIT. Source text: research-use only. |
| `production_labels_only.csv` | **95,391** | The 8-class production labels — the actual training labels for the deployed model. Columns: `row_id`, `category`, `source`, `priority`, `text_sha256` (16-char prefix for de-dup), `text_len_chars`, `text_len_words`. **No source text.** Source attribution: collected from public Saudi delivery-app reviews + synthetic templates + scraped Play Store reviews — text redistribution rights uncertain, so labels-only is the safe path. | MIT (the labels are 100% the team's manual work). Source text: not redistributed. |
| `docs/BOUNDARIES.md` | – | The exact labeling rules. The canonical "what is and isn't ambience" reference. | MIT |
| `docs/DATA_SOURCES.md` | – | Tier 1/2/3 source ranking + ethical collection guidance. | MIT |
| `docs/DATA_SCHEMA.md` | – | 19-column schema for ambience-experiment data files. | MIT |

## The 12 ambience subtypes

```
temperature_ac        seating_comfort       space_crowding
noise_music           lighting              smell
decor_furniture       parking               bathroom_facilities
outdoor_view          privacy               general_place_vibe
```

Each ambience-positive row in `synthetic_contrastive_pairs.csv` and `boundary_examples.csv` has an `ambience_subtype` column tagging which of the 12 it belongs to.

## Why this exists

The 8-class production model at the parent project intentionally **dropped** the ambience class in v3 → v4 because labels were ~99% noise. v5 is the experimental retry. The boundary rules are explicit, the keyword vocab is restaurant-grounded, and the synthetic pairs deliberately train the model on what makes ambience different from food temperature, food smell, dirty surfaces, and staff behavior.

If you're building an Arabic aspect-classification model, the boundary cases are where you'll fail. We documented every one we hit.

## Recommended use

**For training a 9-class Arabic restaurant complaint classifier** that includes ambience:
- Use `synthetic_contrastive_pairs.csv` as supervised training data (training-only — never in val/test, by source rule)
- Use `annotations_qaym_*.csv` as a filtering layer on `hadyelsahar/ar_res_reviews` to identify weak-labeled real ambience examples
- Use `boundary_examples.csv` as a known-good test set for boundary cases
- Use `adversarial_test_set.csv` as the hard-test gate before shipping

**For continued pre-training (MLM)** on Arabic restaurant text:
- Use `annotations_qaym_*.csv` and `annotations_hard_*.csv` `id` columns to fetch source text from the original datasets, then run MLM on that text

**For evaluation only:**
- Use `adversarial_test_set.csv` as a held-out test set for any Arabic restaurant ambience model. Don't include it in your training data.

## Loading

```python
from datasets import load_dataset

# Load the synthetic contrastive pairs
ds = load_dataset("FerasMad/arabic-restaurant-ambience", data_files="synthetic_contrastive_pairs.csv")
```

Or just `pd.read_csv` — these are plain CSVs with UTF-8 BOM (Excel-friendly).

## Reproducing the harvested annotations

See `docs/REPRODUCE.md` for step-by-step instructions to recover the full text + annotations by joining `annotations_*.csv` against the upstream HF datasets.

## Boundary rules (the most important section)

Three rules that determine ambience labeling — full table in `docs/BOUNDARIES.md`:

| Rule | Ambience | Not ambience |
|---|---|---|
| Temperature | `المكان حار والمكيف ما يبرد` (room temp / AC) | `الأكل بارد` (food temp) |
| Smell | `ريحة الزيت في المطعم` (environment) | `ريحة الأكل غريبة` (food) |
| Furniture | `الطاولة تهتز` (broken/uncomfortable) | `الطاولة وصخة` (dirty → cleanliness) |
| Bathroom | `السيفون خربان` (broken facility) | `الحمام وصخ` (dirty → cleanliness) |
| Staff vs ambience | n/a | `الموظف بارد` (staff behavior, not room temp) |

If your model can label all 12 boundary examples in `boundary_examples.csv` correctly, it understands ambience.

## Performance benchmarks

The associated model (CAMeLBERT-mix fine-tuned, available at [FerasMad/arabic-complaints-classifier](https://huggingface.co/FerasMad/arabic-complaints-classifier)) achieves on `adversarial_test_set.csv`:

- **Ambience F1: 89.22%** (with threshold tuning + abstain wrapper)
- Ambience precision: 86.67%
- Ambience recall: 91.92%
- Per-attack-type pass rates: clean 89%, boundary 80%, sarcasm 100%, long 87%, multi_aspect 40%, OOD 50%

Multi-aspect remains the hardest case — single-label classification can't decompose "الاكل بارد والمكيف خربان" cleanly. Multi-label retraining is the next step.

## Citation

If you use this dataset, please cite:

```bibtex
@misc{madkhali2026ambience,
  title  = {Arabic Restaurant Ambience Annotations},
  author = {Madkhali, Feras and the AI Club NLP Team},
  year   = {2026},
  url    = {https://huggingface.co/datasets/FerasMad/arabic-restaurant-ambience},
  note   = {Annotations on hadyelsahar/ar_res_reviews and HARD; synthetic contrastive pairs and adversarial test set are original work}
}
```

Also cite the upstream datasets if you use the source text:

- `hadyelsahar/ar_res_reviews` — Arabic restaurant reviews from qaym.com
- `HARD` — Hotel Arabic Reviews Dataset

## License

- **Annotations + synthetic + adversarial + keyword vocab + docs**: MIT (your work)
- **Source text** (when joined back from upstream datasets): per the upstream dataset's license — read each card before redistribution

## Contributing

Issues, label corrections, and contributions are welcome at the parent repo: <https://github.com/FerasMad/NLP-complaints-system>

If you label more rows in any of the candidate sets, please share back so the community gets cleaner training data over time.

---

**Built by:** the NLP team at AI Club, led by Feras Madkhali.
**Built with:** ChatGPT, Claude (Anthropic), Codex (OpenAI) — used for boilerplate and analysis. Schema design, boundary rules, keyword curation, and labeling decisions are the team's work.
