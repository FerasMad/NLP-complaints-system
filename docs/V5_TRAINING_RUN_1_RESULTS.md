# V5 Ambience Training Run 1 — Results

First end-to-end execution of `src/train_ambience_v5.py` and
`src/evaluation/evaluate_ambience.py`. Run by Codex on user's RTX 4070,
~28 minutes wall-clock. This document captures the actual measured
numbers (vs. earlier predictions in `SENIOR_REVIEW.md`), the failure
modes, and the bugs that surfaced.

## Setup

| | |
|---|---|
| Hardware | RTX 4070 (CUDA 12.4, fp16 enabled) |
| Base model | CAMeL-Lab/bert-base-arabic-camelbert-mix |
| Training time | ~27m56s (4 epochs, batch 16, lr 2e-5) |
| Baseline data | `data/processed/train.csv` (`data/text/complaints_labeled.csv` was missing locally — used the pre-split train.csv as substitute) |
| Ambience synthetic | `data/text/ambience_synthetic_v1.csv` (250 rows: 125 contrastive pairs) |
| Real ambience labels | **Zero** (the central blocker still unresolved) |

## Training metrics (val + test from the auto-split)

| Metric | Value | Notes |
|---|---:|---|
| Test accuracy | 88.16% | High because the 8 production classes dominate |
| Test macro F1 | 72.70% | Pulled down by ambience |
| Test ambience F1 | **0.00%** | **Misleading** — see below |

The 0% ambience F1 is **not a model failure**; it's a split-construction
artifact. The synthetic ambience CSV is in the train-only-leakage gate,
so val and test ended up with zero ambience rows. There were no
positives to predict, so any F1 calculation returns 0.

## Adversarial evaluation (the real test)

Run on the 183-case `tests/fixtures/ambience_adversarial.csv` after
training. Two checkpoints evaluated separately:

### Promoted "best" model (selected by `metric_for_best_model="ambience_f1"`)

| Metric | Value |
|---|---:|
| Overall accuracy | 49.73% |
| Ambience F1 | 70.13% |
| Easy-difficulty accuracy | 68.89% |
| Min attack-type accuracy | 0.00% |

All 4 quality gates failed. **Root cause:** the "best" metric was
ambience_f1, which was always 0 on val (no ambience there), so
"best" was effectively random — Trainer happened to keep the worst
checkpoint.

### Final epoch checkpoint (manually evaluated by Codex as a sanity check)

| Metric | Value |
|---|---:|
| Overall accuracy | 59.02% |
| **Ambience F1** | **78.57%** |
| Easy-difficulty accuracy | 80.00% |
| Min attack-type accuracy | 0.00% |

Still failed gates, but **+8.4 points ambience F1** vs the "best"
checkpoint. This confirms the bug.

## Per-attack-type breakdown (final checkpoint)

| Attack type | Passed | Total | Accuracy | Comment |
|---|---:|---:|---:|---|
| boundary | 23 | 25 | 92% | Crushed it — keyword vocab + contrastive pairs work as designed |
| emoji | 7 | 8 | 88% | Emoji ignored correctly |
| mixed_dialect | 10 | 12 | 83% | Arabic + English code-switch handled |
| clean (easy positives + negatives) | 36 | 45 | 80% | OK but should be ≥95% |
| short | 5 | 7 | 71% | Fine |
| typo | 8 | 12 | 67% | Some typos break tokenization |
| adversarial | 7 | 15 | 47% | Figurative language hard |
| multi_aspect | 6 | 15 | 40% | Known limit — model picks one aspect |
| sarcasm | 3 | 10 | 30% | Sarcasm reads as praise → wrong class |
| long | 2 | 8 | 25% | Multi-sentence ambience reviews go to dominant non-ambience cue |
| negation | 1 | 15 | 7% | Model has no negation-awareness training |
| out_of_domain | 0 | 8 | 0% | Non-restaurant text classified as something |
| very_short | 0 | 3 | 0% | 1-3 word inputs forced into a class |

## Per-class metrics (final checkpoint)

| Category | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| **الجو والمكان** | 66 | 3 | 33 | **95.65%** | **66.67%** | **78.57%** |
| خدمة الموظفين | 10 | 2 | 2 | 83.33% | 83.33% | 83.33% |
| النظافة | 9 | 6 | 1 | 60.00% | 90.00% | 72.00% |
| دقة الطلب | 2 | 2 | 1 | 50.00% | 66.67% | 57.14% |
| السعر والقيمة | 2 | 3 | 0 | 40.00% | 100.00% | 57.14% |
| وقت الانتظار | 2 | 2 | 1 | 50.00% | 66.67% | 57.14% |
| جودة الطعام | 14 | 18 | 4 | 43.75% | 77.78% | 56.00% |
| عامة | 2 | 11 | 1 | 15.38% | 66.67% | 25.00% |
| التوصيل | 0 | 0 | 4 | 0.00% | 0.00% | 0.00% |

**Key reading:** ambience precision is **95.65%** — when the model
commits to ambience, it's almost always right. The problem is
**recall: 66.67%** — it under-fires. The model defaults to
`جودة الطعام` for ambiguous cases (it has 45k training rows vs
ambience's 2500-mostly-noise + 125 v5 contrastive).

## Bugs surfaced (now fixed)

1. **`metric_for_best_model="ambience_f1"` with empty val ambience**
   Selected effectively-random "best" checkpoint. Final checkpoint
   was ~8 points better. **Fix:** detect empty-val-ambience and
   auto-fall-back to `weighted_f1`. New CLI flag `--best-metric`
   for explicit override + `--promote-final-checkpoint` to skip
   best-model-at-end entirely.

2. **`v5_metrics` filtered out all eval keys**
   The dict comprehension `{k: v ... if not k.startswith("eval_")}`
   meant config.json's v5_metrics had only "epoch". **Fix:** strip
   the `eval_` prefix instead of dropping the keys. Also added
   `best_metric_used` and `promoted_final_checkpoint` to the
   config so future eval runs can see which checkpoint they're on.

3. **`save_strategy="epoch"` with no `save_total_limit`**
   Disk filled with 4 × 440MB checkpoints (~1.7GB). **Fix:**
   added `save_total_limit=2`.

## What this confirms

- The infrastructure works end-to-end (train → save → evaluate → reports)
- Synthetic-only ambience training has a real ceiling around **78-80% F1
  on adversarial set**. Predicted in `SENIOR_REVIEW.md` as 70-75%; we
  got 78.57%. Slightly better than predicted but still well short of 85%.
- The model has **excellent ambience precision (95.65%)** but **weak
  recall (66.67%)** — it knows what ambience looks like but errs on the
  side of not firing. This is the right failure mode (vs. v3 which had
  the opposite problem and over-fired).

## What it doesn't change

- The 85% gate still requires real ambience data. No amount of
  synthetic-data tweaking will get recall above ~70% without diverse
  real examples for sarcasm, long-form, multi-aspect, and out-of-domain
  rejection.
- **Path to 85% remains:** Codex's data acquisition pipeline +
  team labeling → 500+ real ambience positives → retrain.

## Next training run — what to change

**Required:**
- Wait for real labeled ambience data (use `src/data/import_*.py` +
  manual review per `docs/AMBIENCE_DATA_COLLECTION.md`)
- Make sure val and test contain real ambience rows (not just train)

**Improved invocation (with the fixes shipped this commit):**

```
python src/train_ambience_v5.py \
    --epochs 4 --batch-size 16 --lr 2e-5 \
    --baseline-csv data/text/complaints_labeled.csv \
    --best-metric ambience_f1
```

If real ambience data is still scant after collection, also pass
`--promote-final-checkpoint` to force keeping the final epoch.

**Likely-helpful additions (not yet implemented):**

- Add a `--exclude-source` flag to drop the v3-era `synthetic` ambience
  rows that we now know are noisy (1933 rows tagged as ambience but
  templated and grammatically wonky)
- Add focal-loss option for the long-tail problem (high-precision
  but low-recall is exactly what focal loss helps with)
- Add an inference-time threshold for the ambience class specifically
  (model is 95% precise — could lower its decision threshold to gain
  recall)

## Files generated by this run (all gitignored)

- `models/single_ambience_v1/` — promoted "best" model + `_checkpoints/`
- `logs/` — Trainer logs
- `reports/ambience/` — adversarial eval against promoted model
- `reports/ambience_checkpoint_13388/` — adversarial eval against final
  epoch checkpoint (the better one)

## Notes for the next runner

The training script now:
- Auto-detects empty val ambience and adjusts metric_for_best_model
- Saves `best_metric_used` and `promoted_final_checkpoint` in
  `config.json:v5_metrics` so post-hoc eval knows which checkpoint
  is loaded
- Limits checkpoints to 2 (~880MB instead of ~1.7GB)
- All other behavior unchanged from Codex's first run
