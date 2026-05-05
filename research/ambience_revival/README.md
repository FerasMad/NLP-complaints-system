# Ambience Revival — Research Sandbox

Exploratory work toward a possible v5 schema that re-introduces an `الجو والمكان` (ambience / physical space) class. **This is sandboxed research, not part of the production model.**

## Why is this sandboxed?

The production model is 8-class and intentionally does not include ambience. The v3 → v4 transition dropped ambience after a 171-sample audit found:

- Only ~1.2% of training labels were genuine ambience (2 of 171)
- 47% were mislabeled (rater disagreement)
- 40% were multi-aspect with ambience as a fragment, not the focus

Dropping the class gained **+7.7% macro F1** and **+43 points** on min-class F1. See `documentation.pdf` §8 and the team auto-memory note `feedback_drop_broken_features.md`.

## What's different now (why retry might work)

Three things have changed since the v3 drop:

1. **PySarf morphology** — `والمندوبين` matches `مندوب` via stem. Fewer surface-form misses on aspect words.
2. **Aspect extraction layer** — we can spot ambience phrases in text even when the model is over-confident on a different class. This is direct evidence of multi-aspect, not a heuristic.
3. **Keyword rescue infrastructure** — defensive post-processing for unambiguous phrases. We already use it for `الحمام → النظافة`.

These didn't exist in v3. They make the labeling discipline easier to enforce and the model failure modes easier to catch.

## What's in this folder

| File | Purpose |
|---|---|
| `BOUNDARIES.md` | The exact rules for what is and is not ambience, with boundary examples |
| `DATA_SOURCES.md` | Where to ethically source ambience data (no TOS-violating scraping) |
| `keywords.py` | Saudi/Gulf ambience vocabulary, risky words, hard-negative cues, ambience subtypes |
| `filter_ambience.py` | CLI script that triages a CSV into ambience-candidates / hard-negatives / review-needed |
| `test_examples.csv` | 80 hand-written Saudi/Gulf test cases for evaluating any future ambience classifier |

## Workflow (proposed, not yet executed)

1. Take an existing CSV of Arabic complaints (any source).
2. Run `filter_ambience.py` on it. You'll get three CSVs split by signal strength.
3. Hand-label the `ambience_candidates.csv` rows using `BOUNDARIES.md` as the rulebook.
4. Hand-label a sample of `hard_negative_candidates.csv` to confirm those are genuinely non-ambience.
5. If you end up with **300+ clean ambience rows + balanced negatives**, the retraining is worth attempting.
6. If you can't reach that bar, drop the experiment again — the v4 decision was right.

## Non-goals

- This sandbox does **not** build a model.
- It does **not** add `app_payment` or any other class — the production schema stays 8-class plus the optional new `الجو والمكان`.
- It does **not** change `hf_space/`, `app/`, or `models/`.

## Run it

```
python research/ambience_revival/filter_ambience.py \
  --input data/raw/some_reviews.csv \
  --text-col text \
  --output-dir data/processed/ambience_triage
```

Outputs: `ambience_candidates.csv`, `hard_negative_candidates.csv`, `review_needed.csv` (all UTF-8 with BOM, Excel-friendly).
