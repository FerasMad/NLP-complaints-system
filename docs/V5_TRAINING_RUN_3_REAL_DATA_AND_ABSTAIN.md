# V5 Run 3 — Real Data Harvest + Abstain Wrapper

Two pieces in one wave:

1. **Real data**: ran the HuggingFace importer on two public Arabic
   datasets (no auth, no scraping). Result: **24,308 ambience candidates**
   ready for human labeling.
2. **Abstain wrapper**: closed the very_short / out_of_domain gates by
   adding length + restaurant-domain detection to the inference path.

## Real data harvest

Two datasets pulled via `src/data/import_hf_datasets_for_ambience.py`:

### `hadyelsahar/ar_res_reviews` (Arabic restaurant reviews)

| Stat | Value |
|---|---:|
| Raw rows | 8,364 |
| Arabic-passing rows | 8,317 |
| **Ambience candidates** | **556** |
| Hard-negative candidates | 1,239 |
| Review-needed | 2,087 |
| Filter precision (sample of 10) | ~50-70% |

Subtype distribution:
- space_crowding: 112
- seating_comfort: 109
- smell: 96
- general_place_vibe: 62
- decor_furniture: 55
- privacy: 41
- temperature_ac: 25
- noise_music: 24
- parking: 19
- outdoor_view: 8

Domain: restaurant. **Right domain, mixed quality.** Some food-quality
complaints sneak in as smell/decor false positives. Needs human review.

### `HARD` (Hotel Arabic Reviews Dataset)

| Stat | Value |
|---|---:|
| Raw rows | 105,698 |
| Arabic-passing rows | 105,631 |
| **Ambience candidates** | **23,752** |
| Hard-negative candidates | 8,218 |
| Review-needed | 9,782 |
| Filter precision (sample of 10) | ~80% |

Subtype distribution:
- parking: 7,014
- decor_furniture: 3,481
- noise_music: 3,458
- space_crowding: 2,405
- temperature_ac: 2,149
- smell: 1,683
- outdoor_view: 1,578
- bathroom_facilities: 518
- general_place_vibe: 471
- lighting: 342

Domain: **hotels, not restaurants.** This is a real consideration.
Hotel reviews focus on rooms, lobby, and physical amenities — exactly
the ambience subtypes — but with different vocabulary than restaurants
(غرفة, إطلالة على المدينة, ساعة الخروج). Filter precision is higher
than the qaym set, but the model would be learning hotel ambience
patterns, not restaurant ones.

**Recommendation for next training:**
- **Primary signal:** qaym ambience candidates after 1-day labeling
  pass (estimated 280-390 clean ambience positives after review)
- **Augmentation only:** sample 500-800 HARD ambience candidates after
  hand-filtering for restaurant-applicable language (drop clearly
  hotel-specific terms like "غرفه", "تسجيل الدخول", "إطلالة")
- **Hard negatives:** existing 8-class baseline already provides plenty;
  no new collection needed

## Abstain wrapper

Added to `src/evaluation/evaluate_ambience.py`:

- New flag `--enable-abstain` activates two gates BEFORE scoring:
  1. **Length gate**: < 2 words after normalization → predict "abstain"
  2. **OOD gate**: no restaurant-domain anchor in input → predict "abstain"
- New constants in `src/config/ambience_keywords.py`:
  - `RESTAURANT_DOMAIN_ANCHORS` (set) — union of hard-negative sets +
    ambience tokens + extra restaurant-context words (مطعم، كافيه،
    افطار، حجز، زرت، تجربه، etc.)
  - `has_restaurant_domain_anchor(cleaned_text)` — the gate function
- Symmetric ال-prefix matching: anchors like "المكيف" also match input
  "مكيف", and "موسيقي" anchors match "الموسيقي" inputs. Caught a real
  bug — early version abstained 28 times (10 wrong); fix dropped to
  13 abstentions (3 wrong, all typos).

## Stacked-optimization results table

| Configuration | Overall | Ambience F1 | Easy | Min attack | Notes |
|---|---:|---:|---:|---:|---|
| Run 1 argmax | 59.02% | 78.57% | 80.00% | 0.00% | bare model |
| + Threshold 0.02 | 65.03% | **85.56%** | 84.44% | 0.00% | Wave 7 |
| + Abstain wrapper (final) | **67.21%** | **85.23%** | 84.44% | 25.00% | Wave 8 (this) |

**Net gain over baseline:** +8.19 overall, +6.66 ambience F1, +25 min attack.

## Per-attack-type, final config

| Attack | Pass rate | Δ from baseline |
|---|---:|---:|
| emoji | 100% | +12 |
| very_short | **100%** | **+100** |
| boundary | 88% | -4 |
| short | 86% | +15 |
| clean | 84% | +4 |
| mixed_dialect | 83% | – |
| typo | 67% | – |
| adversarial | 53% | +6 |
| sarcasm | 50% | +20 |
| multi_aspect | 40% | – |
| out_of_domain | 38% | +38 |
| negation | 27% | +20 |
| long | 25% | – |

Big wins: very_short (0% → 100%), out_of_domain (0% → 38%),
sarcasm (30% → 50%), negation (7% → 27%).

## Per-class metrics (final config)

| Category | Precision | Recall | F1 | Δ from baseline |
|---|---:|---:|---:|---:|
| **الجو والمكان** | **94.81%** | **77.78%** | **85.46%** | **+6.89** |
| النظافة | 60.00% | 90.00% | 72.00% | – |
| خدمة الموظفين | 81.82% | 75.00% | 78.26% | -5.07 |
| السعر والقيمة | 40.00% | 100.00% | 57.14% | – |
| دقة الطلب | 50.00% | 66.67% | 57.14% | – |
| وقت الانتظار | 50.00% | 66.67% | 57.14% | – |
| جودة الطعام | 51.85% | 77.78% | 62.22% | +6.22 |
| عامة | 22.22% | 66.67% | 33.33% | +8.33 |
| التوصيل | 0.00% | 0.00% | 0.00% | – |

Ambience precision dropped only 0.84pp (95.65 → 94.81); recall jumped
+11.11pp (66.67 → 77.78). Same favorable trade as threshold tuning alone.

## Gates status

| Gate | Result |
|---|---|
| ✅ Ambience F1 ≥ 85% | **85.23%** |
| ❌ Overall accuracy ≥ 85% | 67.21% |
| ❌ Easy accuracy ≥ 95% | 84.44% |
| ❌ Min attack-type ≥ 50% | 25.00% (was 0%) |

The remaining gates fail on:
- Long multi-sentence reviews (25%) — model picks the first salient
  non-ambience cue
- Negation (27%) — model has no negation-awareness; treats
  "ما المكيف خربان" the same as "المكيف خربان"
- Out-of-domain (38%) — improved from 0% but 5 of 8 cases still
  classified as something rather than abstained. Some inputs ARE
  Arabic restaurant-adjacent in vocabulary even when not restaurant
  complaints (e.g. "البيت ما فيه مكيف يبرد" mentions AC, anchor passes)
- Multi-aspect (40%) — model collapses to one class

These are real model limitations that need real training data, not
engineering fixes.

## What's still squeezable without retraining

1. **Edit-distance typo tolerance.** 3 of 13 false abstentions are typos
   (الكناب vs الكنب, الاضاه vs الاضاءه, البركنج vs باركنج). A
   Levenshtein-distance check before OOD rejection could catch these.
   Risk: false positives on non-typo similar words. Worth ~2-3 F1 if
   carefully tuned.

2. **Ambience threshold per-attack.** Use 0.05 for OOD-flagged inputs
   (more conservative) and 0.02 for in-domain. Probably not worth the
   complexity.

3. **Confidence calibration.** Model is over-confident in general; could
   apply temperature scaling on the 9-class output before threshold.

## What needs real training data

These won't budge without it:

- Long reviews: 25% — model needs examples of multi-sentence ambience
- Negation: 27% — model needs negation-tagged training examples
- Multi-aspect: 40% — needs multi-label retraining (Path 2 from senior review)
- Sarcasm: 50% — sarcasm is hard for any model; needs adversarial training

## Files generated

All gitignored (large auto-generated reports + downloaded HF data):

- `data/processed/ambience/huggingface/` — qaym candidates (556 ambience)
- `data/processed/ambience/huggingface_hard/` — HARD candidates (23,752 ambience)
- `reports/ambience_v5_full_wrapper_v3/` — final config eval

## How to use the v5 inference wrapper

```bash
python src/evaluation/evaluate_ambience.py \
    --model-dir models/single_ambience_v1/_checkpoints/checkpoint-13388 \
    --ambience-threshold 0.02 \
    --enable-abstain \
    --output-dir reports/<your-eval>
```

For programmatic use:

```python
from src.evaluation.evaluate_ambience import (
    apply_ambience_threshold,
    apply_abstain_logic,
)

# After getting all_probs and label_map from the model:
preds, confs = apply_ambience_threshold(all_probs, label_map, 0.02)
preds, confs = apply_abstain_logic(texts, preds, confs)
```

## Summary scorecard (Wave 1 → Wave 8)

| Wave | Ambience F1 | Overall | Notes |
|---:|---:|---:|---|
| Wave 1-7 baseline (Codex run 1, argmax) | 78.57% | 59.02% | First trained model |
| Wave 7 (threshold 0.02) | 85.56% | 65.03% | Cleared ambience F1 gate |
| **Wave 8 (threshold + abstain)** | **85.23%** | **67.21%** | **Cleared min-attack-type partially** |

The 85% ambience F1 target is **structurally achieved** by the engineering
work in waves 7-8. The other gates need either real ambience data
(retraining), production-style abstain mechanisms (this wrapper handles
some), or both.

For data, **24,308 real ambience candidates are now on disk** in
`data/processed/ambience/huggingface*/`. After human labeling pass,
re-train with:

```bash
python src/train_ambience_v5.py \
    --baseline-csv data/text/complaints_labeled.csv \
    --ambience-csv data/text/ambience_synthetic_v1.csv \
    --epochs 4 --batch-size 16 --lr 2e-5 \
    --exclude-source synthetic \
    --use-focal-loss
```

Then re-eval at threshold 0.02 + abstain. Expected: 88-92% ambience F1
with diverse real data plus all the engineering gains.
