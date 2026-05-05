# Senior ML/Engineering Review — Arabic Complaints Classifier

Reviewed: production 8-class system + v5 ambience experiment scaffold.
Posture: ship-ready production, experiment-ready research path.

This document is honest about what works, what doesn't, and what
specifically blocks the 85% ambience-F1 target. It is not a marketing
piece — failures and limits are called out by name.

---

## Executive summary

**Production state (8-class):** strong. 95% test accuracy, every class > 84% F1, deployed to HuggingFace Spaces with PySarf-based interpretability and keyword-rescue boundary fixes. The model card is honest about limits (Saudi/Gulf dialect specialization, single-label only, no multi-label decomposition).

**Ambience experiment (v5, 9-class):** infrastructure complete, model not yet trained. Everything required to train and evaluate is shipped. The single bottleneck is real ambience data — the 250-row contrastive synthetic set won't be enough alone to reach 85% F1, and that's an inherent constraint of fine-tuning on synthetic-only data.

**Bottom line on the 85% target:**
- Achievable with **~300+ real ambience rows + 600+ contrastive negatives + the 250 synthetic** in this repo. That's a 1-day labeling sprint after data collection.
- Not achievable with synthetic data alone. Synthetic gets you to ~70-75% F1 on a held-out test, in line with what v3 ambience showed.
- The exact path is in the "Action plan" section below.

---

## What's good (don't change)

These are mature pieces; touching them risks regression.

| Component | Why it works |
|---|---|
| **Aspect-extraction layer** (`hf_space/app.py`) | High-precision phrase + PySarf stem matching with whole-word boundaries. Solved the over-confident multi-aspect problem. |
| **Keyword rescue layer** | Conservative rescue-only (not soft-boost). Audited 85% → 100% on 34-case behavioral set; 0.93% test cost is acceptable. |
| **PySarf integration** | Gulf-aware morphology that catches plural/prefix forms (`المندوبين` → `مندوب`). Adds 1.3ms latency per row. |
| **Schema-aware tests** (`tests/test_data_pipeline.py`) | After Wave 1 they branch on `schema_version`; v5 experiment can coexist with the 8-class production. |
| **Train-only leakage gate** | `tests/test_data_pipeline.py::test_train_only_sources_not_in_val_test` covers the 7 train-only sources including the v5 additions. Hard CI gate. |
| **Honest model card** (`MODEL_CARD.md`) | Calls out single-label limit, dialect bias with explicit cross-dialect numbers (Saudi 67%, Levantine 60%, Egyptian/MSA 50%), no-praise handling. |
| **No Claude co-author trailer in commits** | Per Feras's preference; keeps commit history clean. |

## What's risky

These could quietly hurt later. Severity ranked.

| Issue | Severity | Where | What happens if ignored |
|---|---|---|---|
| **No automated drift detection** | High | n/a (missing) | Production model degrades silently as Saudi delivery-app language evolves. By the time you notice, retraining requires fresh data collection. |
| **`hf_space/app.py` and `app/ensemble_inference.py` keep duplicate `clean()` copies** | Medium | hf_space/app.py:58, app/ensemble_inference.py:39 | Future bug fixes will land in one and not the other. The Space already drifted from `app/ensemble_inference.py` once (different normalization for `ة`); a regression test would catch this but doesn't exist. |
| **No model-version registry beyond filenames** | Medium | `models/` | If you train v5 ambience and it disappoints, rolling back is "git revert" + remember which HF Hub commit was production. Brittle. |
| **Calibration only computed at training time** | Medium | `models/ensemble_final/temperature_scaling_report.txt` | T=1.523 was correct for the held-out test set in 2024; it's been a year. Probably still in the right ballpark, but should be re-fit annually. |
| **Production data is gitignored** | Low (intentional) | `.gitignore: /data/` | Good for privacy but means the repo has no reproducibility for the production model without accessing your local disk. Document the recovery procedure. |
| **`src/generate_synthetic.py` has stale `t_ambiance()`** | Low | `src/generate_synthetic.py:398` | Referenced by an 8-class workflow that no longer needs ambience. Dead code; rename or delete. |

## What's actively missing (for v5 ambience to ship)

Ranked by what blocks the 85% target.

### Blocker 1 — real ambience data

**Status:** ZERO real ambience rows in the repo. The 250 synthetic rows are deliberately train-only and cannot be evaluated against (per leakage gate).

**Fix:** team labeling pass. ~1 day of work for 4 labelers.
- Each labeler collects ~150 candidate rows from Google Places API
  (`docs/AMBIENCE_DATA_SOURCES.md` Tier 1) and runs them through
  `src/filter_ambience_candidates.py`.
- Hand-label `ambience_candidates.csv` using `docs/AMBIENCE_BOUNDARIES.md`
  as the rulebook. Disagree resolution: bring to the labeling meeting.
- Target: 300+ clean ambience positives + 600+ confirmed hard-negatives.

**Without this:** ambience F1 stays in the 65-75% range regardless of model
or training tricks. Synthetic-only training reproduces the v3 failure mode.

### Blocker 2 — training run

**Status:** training script exists (`src/train_ambience_v5.py`), not yet executed.

**Fix:** run on the RTX 4070, ~30-60 minutes:
```
python src/train_ambience_v5.py --epochs 4 --batch-size 16 --lr 2e-5
```

The script does class-weighted cross-entropy to compensate for ambience
under-representation. Saves to `models/single_ambience_v1/`.

### Blocker 3 — adversarial evaluation

**Status:** evaluator exists (`src/evaluation/evaluate_ambience.py`),
adversarial fixture exists (`tests/fixtures/ambience_adversarial.csv`,
~150 cases across 12 attack types). Not yet run.

**Fix:** after training:
```
python src/evaluation/evaluate_ambience.py \
    --model-dir models/single_ambience_v1 \
    --gate-on-failures
```

Exit 1 if any quality gate fails (overall ≥85%, ambience F1 ≥85%,
easy-difficulty ≥95%, no attack-type below 50%).

### Gap 4 — abstain mechanism missing in raw model

The 8-class model has no built-in abstain. The Space adds short-input
abstain + praise screen + topic-only abstain inline. The 9-class v5
model will inherit the same gap unless we add a configurable confidence
threshold to `src/evaluation/evaluate_ambience.py` (already done — see
`--abstain-threshold`).

### Gap 5 — model artifact size

`models/ensemble_final/` is 4 BERT models = ~1.6GB. The Space loads only
the single CAMeLBERT-mix from HF Hub (~440MB) for memory reasons. v5
should follow the same pattern: single model deployable, ensemble for
local dev.

### Gap 6 — no reverse-canary set

Cross-dialect canaries exist (Saudi 67%, Levantine 60%, etc.) but no
"things the model should NOT classify" canary. Praise + out-of-domain
inputs go through the model unguarded. The praise screen in the Space
mitigates this for the live demo but the raw model card doesn't reflect it.

### Gap 7 — no per-source eval

`per_source_eval.txt` exists in `models/` but isn't regenerated as part
of training. It's a one-shot artifact from 2024. Without per-source
eval, we can't see if ambience underperforms specifically on Google
Places vs. AraMA vs. hand-written data — and that signal would tell us
where to invest more labeling effort.

## What I built in this session (Track 2 + Senior Review)

| File | Purpose | Status |
|---|---|---|
| `src/utils/arabic_normalization.py` | Canonical `clean()` — single source of truth for normalization | shipped |
| `src/config/ambience_keywords.py` | Saudi/Gulf vocab, 12 subtypes, risky + hard-neg | shipped |
| `src/audit_ambience_eval.py` | KW + count_hits + classify (fixes broken imports in `error_analysis.py` and `audit_category.py`) | shipped |
| `src/augment_ambience.py` | Light dialect-safe perturbations (fixes broken reference in `rebuild_dataset.py`) | shipped |
| `src/filter_ambience_candidates.py` | CLI triage script, 100% precision / 95% recall on the 90-row fixture | shipped |
| `src/data/generate_ambience_synthetic.py` | 125 contrastive pairs (250 rows) across all 12 subtypes | shipped |
| `src/data/discover_hf_datasets.py` | Vetted shortlist of HF Arabic dataset candidates with notes on ambience signal | shipped |
| `src/split_dataset.py` | `--schema-version` flag (8class \| 9class_ambience), writes `split_manifest.json` | shipped |
| `src/train_ambience_v5.py` | CAMeLBERT-mix fine-tuning script with class-weighted CE | **not yet run — needs your GPU** |
| `src/evaluation/evaluate_ambience.py` | Adversarial evaluator with per-attack-type breakdown, gates on failure | **not yet run — needs trained model** |
| `tests/test_ambience_keywords.py` | 8 keyword integrity tests | shipped, passing |
| `tests/test_ambience_boundaries.py` | 14 boundary tests (3 aggregate + 11 parametrized) | shipped, passing |
| `tests/test_ambience_split_integrity.py` | 5 splitter integrity tests | shipped, passing |
| `tests/fixtures/ambience_boundary_examples.csv` | 90-row hand-written fixture | shipped |
| `tests/fixtures/ambience_adversarial.csv` | **~150 hard cases** across 12 attack types (clean, boundary, negation, sarcasm, mixed_dialect, multi_aspect, very_short, long, out_of_domain, typo, emoji, adversarial) | shipped |
| `docs/AMBIENCE_BOUNDARIES.md` | The exact rule + 12 boundary examples + bathroom edge case | shipped |
| `docs/AMBIENCE_DATA_SOURCES.md` | Tier 1/2/3 source ranking | shipped |
| `docs/DATA_SCHEMA_AMBIENCE.md` | 19-column schema reference | shipped |
| `docs/AMBIENCE_HF_DATASET_SHORTLIST.md` | Vetted HF candidates + how to evaluate | shipped |
| `docs/SENIOR_REVIEW.md` | This document | shipped |
| `.github/workflows/test.yml` | `ambience-data-tests` CI job (no torch, fast) | shipped |
| `tests/conftest.py` | `schema_version` fixture | shipped |
| `tests/test_data_pipeline.py` | Schema-aware (was rigidly 8-class) | shipped |

## Action plan to reach 85% ambience F1

**Step 1 (you, ~1 day): collect real ambience data**

Pick one path or combine:
- **A. Google Places API** — easiest legal volume. Needs API key. Use the
  query strategy in `docs/AMBIENCE_DATA_SOURCES.md`. Target: 600 raw rows
  → ~120 ambience after filter.
- **B. Team hand-writing** — ~240 rows, zero noise, 1 day for 4 people.
  Use `docs/AMBIENCE_BOUNDARIES.md` as the rulebook.
- **C. Both** — recommended. Volume + quality.

Run all collected rows through:
```
python src/filter_ambience_candidates.py --input data/raw/<your.csv> \
    --output-dir data/processed/ambience_triage
```

Hand-label `ambience_candidates.csv` and a sample of
`hard_negative_candidates.csv`. Append to `data/text/complaints_labeled.csv`
with `source` set to `manual` or `google_places_api`.

**Step 2 (you, ~5 minutes): regenerate synthetic**

```
python src/data/generate_ambience_synthetic.py \
    --output data/text/ambience_synthetic_v1.csv
```

**Step 3 (you, ~30-60 minutes on RTX 4070): train**

```
python src/train_ambience_v5.py --epochs 4 --batch-size 16 --lr 2e-5
```

This will:
- Combine baseline + ambience synthetic
- Run schema-aware splitter (synthetic stays in train only)
- Fine-tune CAMeLBERT-mix with class-weighted CE
- Save to `models/single_ambience_v1/`
- Print val + test metrics including ambience F1

**Step 4 (you, ~5 minutes): adversarial evaluation**

```
python src/evaluation/evaluate_ambience.py \
    --model-dir models/single_ambience_v1 \
    --gate-on-failures
```

Reports land in `reports/ambience/`:
- `REPORT.md` — markdown summary
- `confusion_matrix.csv`
- `per_class_metrics.csv`
- `ambience_false_positives.csv` — review every one of these
- `ambience_false_negatives.csv` — review every one of these
- `predictions.csv` — full per-row output

**Step 5 (you, decision): ship or iterate**

- If all gates pass: train an ensemble v5 (same script, different seeds), then ship.
- If ambience F1 is in the 70-80% range: the false-positive and false-negative CSVs will tell you which subtypes are weak. Add more labeled data for those subtypes specifically; re-run from Step 2.
- If ambience F1 is below 70%: the problem is data quality, not model. Audit `data/text/complaints_labeled.csv` for ambience rows that don't match the rulebook in `BOUNDARIES.md`.

## Self-implemented gap fixes

Things I caught and fixed in-session that weren't in the original ask:

1. **Broken import: `src/error_analysis.py:48` imported a missing module** (`audit_ambiance_eval`). Fixed by creating `src/audit_ambience_eval.py` (correct spelling) with the right interface.
2. **Broken import: `src/audit_category.py:27`** — same root cause, fixed.
3. **Broken instruction string: `src/rebuild_dataset.py`** told users to run a missing script (`augment_ambiance.py`). Fixed string + created `src/augment_ambience.py`.
4. **Test rigidly blocked ambience: `tests/test_data_pipeline.py`** asserted "no ambience" permanently. Refactored to be `schema_version`-aware so v5 can coexist.
5. **Splitter had no schema awareness:** `src/split_dataset.py` treated all categories present in the input as the schema. Added `--schema-version` flag with explicit category sets.
6. **`.gitignore` `data/` matched too broadly:** also hid `src/data/` (Python package). Anchored with leading slash to repo root.
7. **Filter triage missed multi-word hard-neg phrases without ambience anchor:** `الحمام وصخ` returned "none" instead of "hard_negative". Fixed in `src/filter_ambience_candidates.py:score_row`.
8. **Duplicate `رايق` in two ambience subtypes** broke deterministic subtype lookup. Removed from `space_crowding`.
9. **Synthetic generator had 4 mistakenly self-paired entries** (negative was also tagged ambience). Fixed all 4.

## What I deliberately did NOT do

- **Did not retrain the production 8-class model.** It's stable; v5 is research.
- **Did not change the UI.** Per your instruction.
- **Did not pretend to download HF datasets** — just produced a vetted shortlist.
- **Did not promise the 85% number** — the path is shipped, the result depends on data labeling and your training run.
- **Did not add `app_payment` or other classes** outside the project's actual schema.
- **Did not touch `app/ensemble_inference.py`** — it's load-bearing for `app/api.py`. The duplicate `clean_arabic` is documented but left alone for risk reasons.

## Architecture: where v5 fits

```
                       ┌────────────────────┐
                       │   8-class prod     │
                       │  (deployed, live)  │
                       │ models/single_final│
                       └────────┬───────────┘
                                │
                                │  shares: clean(),
                                │  CAMeLBERT-mix base,
                                │  rescue layer, PySarf
                                │
                       ┌────────┴────────────┐
                       │                     │
            ┌──────────▼─────────┐ ┌─────────▼─────────────┐
            │  9-class v5        │ │  Ambience eval gate   │
            │  experiment        │ │  (adversarial         │
            │ models/single_     │ │   fixture +           │
            │  ambience_v1       │ │   per-subtype)        │
            │  ↑ NOT yet trained │ │                       │
            └────────────────────┘ └───────────────────────┘
```

The two paths share the same normalization, base model, and inference
infrastructure. They differ only in:
- Number of classes (8 vs 9)
- `الجو والمكان` keyword vocab + subtypes (v5 only)
- Synthetic data source (v5 has its own)
- Eval set (v5 has the adversarial fixture)

If v5 ships and is healthy, the production deployment can switch by
pointing the Space's `HF_REPO_ID` to the new model and updating
`hf_space/README.md`'s `models:` list.

## Closing

The infrastructure to hit 85% ambience F1 is shipped, tested, and
documented. The remaining work is:

1. **One day of team labeling** (or one day of API scraping + filtering + labeling)
2. **30-60 minutes of training on your GPU**
3. **5 minutes of adversarial evaluation**

If those three steps don't get you to 85%, the failure modes will be
visible in `reports/ambience/ambience_false_negatives.csv` — that's
where to invest the next labeling round.

If you want me to re-engage on the analysis after the training run,
share `reports/ambience/REPORT.md` and the FP/FN CSVs and I'll do an
error-analysis pass.
