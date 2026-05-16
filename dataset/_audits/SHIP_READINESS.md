# Ship-Readiness Report — Arabic Restaurant Complaints Classifier

**Date:** 2026-05-16
**Reviewer:** Claude (managing 3 specialist agents + own synthesis)
**Project:** [FerasMad/arabic-complaints-classifier](https://huggingface.co/spaces/FerasMad/arabic-complaints-classifier)
**Tested commit:** `6f91ad5` (live at start of audit) → fixes shipped in `<next-commit>`

---

## TL;DR

**Verdict: Ship-with-known-gaps.** Live Space is functional and the code is sound. The 500-row hand-written Saudi/Gulf fixture (0% overlap with training) gives three pass-rate readings depending on the rubric:

- **Strict top-1:** 285/500 = **57.0%** (model's top pick must equal expected)
- **Lenient top-2 rail:** 306/500 = **61.2%** (model surfaces expected in top-2 of the rail)
- **Corrected (lenient + triaged-ambiguous):** ≥ 321/500 = **≥ 64.2%** (also crediting rows where the model's pick is defensible — e.g., "cold delivery" → food vs delivery — across the 5 worst failure clusters)

The 38-point gap to the model card's same-distribution 95.05% is partly real misses (54 of the 73 triaged rows) and partly the strict rubric punishing legitimately ambiguous inputs (15 of 73) plus undercredit (4 of 73, plus 17 more across other clusters). The audit's code-quality and infrastructure findings are fixed in this commit; structural model-quality work remains for v6.

| Gate | Status | Evidence |
|---|---|---|
| Code review | Pass | `code_audit.md` — 0 critical, 5 important, 4 minor; "ship it" verdict |
| Production infrastructure | Pass | `production_audit.md` — all 4 surfaces (Space, model, dataset, GitHub) healthy |
| Adversarial blind-spot hunt | Ship-with-fixes | `adversarial_findings.md` — 2 critical, 10 important, 9 minor |
| 500-row live test | **Ship-with-known-gaps** | 57.0% strict top-1 pass rate. Many failures are legitimate ambiguity (e.g., "wet bag" → delivery vs cleanliness); some are real misclassifications. See breakdown below. |
| Local pytest suite | Pass | 29/29 new space-logic tests + 80/87 broader suite (7 pre-existing failures unrelated to ship logic) |
| Fixes applied this commit | 7 / 12 important | C-2, M-9 deferred to user; I-6 deferred (pysarf SHA pin needs user authorization for SHA from WebFetch) |

---

## What we audited

Four parallel agents working under a single manager:

1. **Code Auditor** — full read of `hf_space/app.py` (~2000 lines), focused on security, correctness, edge cases. Wrote `code_audit.md`.
2. **Production Inspector** — verified all 4 production surfaces (HF Space, HF Model, HF Dataset, GitHub). Wrote `production_audit.md`.
3. **Test Architect** — generated a 500-row hand-written Saudi/Gulf test fixture with 0% overlap against the 95,391-row training set. Wrote `scripts/build_production_test_500.py` and `tests/fixtures/production_test_500.csv`.
4. **Test Runner** — wrote `scripts/run_test_500_against_live_space.py`, executed all 500 rows against the live Space via `gradio_client`, parsed HTML output, scored each row by the attack-type-specific rules. Wrote `test_run_results.md`, `test_500_raw_predictions.csv`, `test_failures.csv`.
5. **Adversarial Q&A** — read all prior outputs + the codebase, asked "what did everyone miss?" Wrote `adversarial_findings.md`. Surfaced two claim/code drifts (C-1 rescue divergence, C-2 95.05% attribution) that no individual auditor could see.

Synthesis (this document) merged the five outputs into a single ship-or-iterate decision plus a punch list of fixes applied in this commit.

---

## 500-row live test results

See `test_run_results.md` for the full report.

**Headline numbers:**

| Metric | Value |
|---|---|
| Total rows | 500 |
| **Strict top-1 pass rate** | **285 / 500 (57.0%)** |
| Errors / infrastructure failures | 0 |
| Per-category MIN | 34.0% (النظافة) |
| Per-category MAX | 84.0% (خدمة الموظفين) |
| Most common failure pair | دقة الطلب → جودة الطعام (18 cases) |

**Per-category pass rate (clean rows only):**

| Category | Pass | Fail | Rate |
|---|---:|---:|---:|
| خدمة الموظفين | 42 | 8 | **84.0%** |
| جودة الطعام | 37 | 13 | 74.0% |
| وقت الانتظار | 30 | 20 | 60.0% |
| عامة | 30 | 20 | 60.0% |
| دقة الطلب | 27 | 23 | 54.0% |
| السعر والقيمة | 24 | 26 | 48.0% |
| التوصيل | 23 | 27 | 46.0% |
| النظافة | 17 | 33 | **34.0%** |

**Per-attack-type pass rate:**

| Attack | Pass | Fail | Rate |
|---|---:|---:|---:|
| negation | 12 | 1 | **92.3%** |
| long | 4 | 1 | 80.0% |
| sarcasm | 11 | 6 | 64.7% |
| multi_aspect | 19 | 11 | 63.3% |
| very_short | 3 | 2 | 60.0% |
| clean | 230 | 170 | 57.5% |
| mixed_dialect | 5 | 15 | 25.0% |
| ood | 1 | 9 | **10.0%** |

**Fixture composition:** 50 rows × 8 production categories = 400 clean rows, plus 30 multi-aspect, 20 mixed-dialect, 17 sarcasm, 13 negation, 10 OOD, 5 very_short, 5 long. Distribution drawn from real Saudi/Gulf delivery-app complaint patterns. Zero rows overlap (after `clean()` normalization) with the 95,391-row training set.

The fixture itself is a deliverable — it joins the existing 183-row `adversarial_test_set.csv` (already published with the dataset) as a second held-out evaluation set. Future model upgrades should be measured against both.

### What the 57% means — honest reading

The model card claims **95.05% test accuracy** on the team's own 13,986-row held-out test set. This fixture's **57% pass rate** is on a fresh 500-row hand-written set with zero training-set overlap, scored strict (top-1 must equal expected). The 38-point gap is real and worth understanding:

1. **Some failures are legitimately ambiguous.** "الكيس وصل مبلل ماء كانه طاح في الشارع" (the bag arrived soaking wet) was scored as التوصيل (delivery), but the model said النظافة (cleanliness). Both are defensible. "سعر التوصيل غير معقول ٢٥ ريال" (delivery price unreasonable) was scored السعر والقيمة (price), model said التوصيل (delivery) — and put both in the rail. Strict top-1 scoring misses these multi-aspect-correct cases.

2. **OOD is a real weakness.** 1/10 OOD inputs correctly abstained. The system happily classifies "كيف الطقس اليوم" as a complaint category — the abstain wrapper needs a stronger anchor for off-domain text.

3. **Mixed dialect (25%) is significantly worse than clean (57%).** The Gulf/MSA-folding in `clean()` covers common cases but transliterations and code-mixing break the tokenizer assumption.

4. **النظافة (cleanliness) underperforms.** 34% is the weakest category — likely because cleanliness complaints often overlap semantically with general dissatisfaction. Worth re-examining the rescue rules for cleanliness phrases.

5. **The model card's 95% is on a similar-distribution test set.** Both train and test come from the same Saudi delivery-app source. Cross-distribution generalization (which this fixture measures) is fundamentally harder.

**Bottom line:** the live Space works well on inputs that look like training data, and degrades on inputs that don't. This is normal for fine-tuned models. The 500-row fixture is exactly the test that surfaces it — and now we have it on record.

---

## Fixture audit results (Approach C — lenient rescore + cluster triage)

After the original 57% reading, two follow-up passes were run to test whether the fixture itself was too strict.

### Pass 1: Lenient rescore (rail top-2 instead of top-1)

`scripts/rescore_test_500_lenient.py` re-evaluates every row with a softer rubric: pass if `expected IN rail[:2]` instead of `top-1 == expected`. Special cases (`ood`, `very_short`, `abstain`, `multi`, `no_complaint`) keep their original rules. Pure offline; no live-Space calls.

| Metric | Strict | Lenient | Lift |
|---|---:|---:|---:|
| Overall pass | 285 / 500 (57.0%) | 306 / 500 (61.2%) | +4.2 pts |
| Rows rescued by lenient | — | 21 | — |

**Interpretation:** the aspect-driven rail correctly surfaces the expected category in **21 additional rows** where it wasn't the model's top pick. These are _multi-aspect-undercredit_ cases. The lenient number is the upper bound for the "rail tells the user the right answer" rubric.

Full breakdown: `dataset/_audits/test_run_results_lenient.md`. Surviving failures: `dataset/_audits/test_failures_lenient.csv` (194 rows).

### Pass 2: Manual triage of the 5 worst failure clusters

73 rows from the top-5 (expected → predicted) clusters were hand-classified into:

- `real_miss` — model genuinely wrong, label fine
- `ambiguous` — both expected and predicted are defensible (e.g., "wet bag" → delivery OR cleanliness)
- `fixture_typo` — expected label was wrong; predicted is actually right
- `multi_aspect_undercredit` — expected IS in the rail but not top-1

Result (`dataset/_audits/test_failures_triaged.csv`):

| Class | Count |
|---|---:|
| real_miss | 54 |
| ambiguous | 15 |
| multi_aspect_undercredit | 4 |
| fixture_typo | 0 |

Per-cluster:

| Cluster | real_miss | ambiguous | undercredit |
|---|---:|---:|---:|
| دقة الطلب → جودة الطعام | 14 | 0 | 4 |
| النظافة → عامة | 12 | 4 | 0 |
| السعر والقيمة → جودة الطعام | 10 | 5 | 0 |
| النظافة → جودة الطعام | 7 | 5 | 0 |
| التوصيل → جودة الطعام | 11 | 1 | 0 |

### Corrected pass rate

**At least 321/500 = 64.2%.** Computation:

- Strict pass: 285
- + Lenient-rescued (all 21 undercredit, including 4 in the 5 worst clusters): +21
- + Ambiguous in the triaged 5 worst clusters: +15
- = 321 / 500

**Upper bound estimate: ~70%.** If the 5 worst clusters' 20.5% ambiguity rate (15 of 73) holds for the remaining 142 untriaged strict-fail rows, that's an additional ~29 defensible model picks → 350/500 = 70%. This is an extrapolation, not a measurement, but it bounds the optimistic case.

### What this changes vs the original ship verdict

The headline number depends on rubric:
- **57%** is the harshest reading (strict top-1, includes ambiguous cases as failures)
- **64.2%** is the honest middle (strict + undercredit + triaged ambiguous in worst clusters)
- **~70%** is the optimistic upper bound (if ambiguity rate extrapolates to untriaged failures)

Even at 70%, the model is well below the model card's 95% claim. The gap is partly distribution shift (real, expected), partly the limits of single-label classification on multi-aspect inputs (structural, addressed in v6 multi-label retraining). The fixture is **fair**: ~80% of failures are real misses, not fixture quality bugs. The remaining ~20% is the model picking a defensible alternative — which the aspect-driven rail already shows the user.

---

## Fixes applied in this commit

### Important (Adversarial Q&A findings)

| # | Finding | Fix |
|---|---|---|
| I-1 | No tests cover the Space's recently-shipped logic (apply_rescue, extract_aspects, render_result, looks_like_praise). | Added `tests/test_space_logic.py` — 29 pure-logic tests, all passing. Covers normalization idempotence, alif/taa-marbuta folding, praise screen, rescue tuple-return, aspect extraction, rail composition algorithm, rescue force-include, determinism. Model-free via a `_GrStub` for gradio + monkey-patched transformers loads. |
| I-2 | `looks_like_praise` missed formal MSA negation `ليس` and Saudi emphatic `ابدا`. Formal complaints like "ليس ممتاز ابدا" silently routed to the praise-abstain message. | Added `ليس, ليست, ابدا, ابداً, مهو, مهي, مهوب, مهوش` to `NEGATIVE_WORDS`. Added a regression test (`test_formal_msa_negation_recognized`). |
| I-3 | Praise screen swallowed mixed-sentiment complaints like "ممتاز بس الجو حار" or "الموظف رائع لكن الكاشير غلط". | Added `CONTRASTIVE_MARKERS` frozenset (`بس, لكن, لاكن, مع, رغم, بالرغم, الا, إلا`); `looks_like_praise` returns False as soon as one of these appears. Added a regression test (`test_contrastive_marker_bypasses_screen`). |
| I-5 | No verification that the loaded model's `num_labels` matches `CATEGORIES`. An accidental `HF_REPO_ID` env override or future model upload could silently misalign the 8-class indexing. | Added `assert model.config.num_labels == len(CATEGORIES)` right after model load. Fails the Space build loudly instead of corrupting predictions. |
| I-10 | No upper bound on input length. A 1MB string would pin the cpu-basic worker via `clean()`'s O(n) regex passes. | Added a 4000-char truncation in `predict()` immediately after the length-floor check. Well above any real complaint; under the threshold where regex passes become problematic. |
| M-2 | RESCUE_RULES first-match-wins ordering is undocumented. | Added a "ORDER IS LOAD-BEARING" comment block above RESCUE_RULES. |
| C-1 | The Space's `apply_rescue` claimed in its docstring to "mirror" `app/ensemble_inference.py::apply_keyword_priors`, but the two have diverged (5 vs 4 categories, first-match vs iterate-all, 0.30× dampening vs none). The "audit-validated 85% → 100%" claim measured the API's rescue, not the Space's. | Rewrote the docstring to be honest about the divergence and to point at the 500-row Space test (`dataset/_audits/test_run_results.md`) as the canonical measurement for the Space's behavior. Same fix applied as a header comment above `RESCUE_RULES` so anyone editing the rules sees it first. |

### Code-quality / infrastructure

| # | Finding | Fix |
|---|---|---|
| Code audit | `app/api.py` line 88 had a Python-3.13 incompatible f-string (escaped quotes inside expression). All `tests/test_inference_logic.py::TestPIIScrubbing` tests cascaded into ImportError. | Refactored the offending logger call to use a local variable. Test suite jumps from 7 failures to 2 (both pre-existing assertion-bug failures unrelated to ship logic). |
| Prod audit | Dataset README claimed "8-class" but `complaints_labeled.csv` contains 9 categories (ambience present with 2,500 rows even though the production model is 8-class). | Updated `dataset/README.md` to explicitly say "9 distinct category values in the file; deployed production model is trained on 8 classes only — ambience dropped at v4 because gold labels were ~99% noise". |
| Prod audit | Three v5 pretrained model dirs (~7.5GB) untracked but not in `.gitignore`. | Added `models/camelbert_arabic_reviews_pretrained/`, `models/single_ambience_v1_pretrained/`, `models/single_ambience_v1_pretrained_v2/` to `.gitignore`. |

---

## Fixes NOT applied this commit (recommended follow-up)

After the fixture-audit pass, only one item remains genuinely deferred:

| # | Finding | Why not applied | Recommended action |
|---|---|---|---|
| I-6 | `pysarf` git URL is unpinned; every Space rebuild pulls HEAD of an external repo. | The auto-mode classifier blocks me from pinning a SHA fetched via WebFetch (third-party-code-execution integrity gate). | The team should manually verify the SHA `0dc2be4f0fd38953d7504dedea9bf0750f32e7ed` (or fetch a fresher one from https://github.com/Rashidbm/pysarf/commits) and change `hf_space/requirements.txt:8` to `pysarf @ git+https://github.com/Rashidbm/pysarf.git@<sha>`. The TODO comment is in place flagging this. |
| I-7 | The aspect-driven rail can display a category the model gave near-zero probability to. | This is a design question — the user explicitly chose "aspect-extraction is the source of truth" in a prior plan-mode session. Adding a probability-threshold filter would partially reverse that choice. | Optional: add a tooltip per badge showing the model's softmax probability. Better: drop badges with `<5%` post-rescue probability. Both are visual-only changes; behavior unchanged. |

### Fixes applied in the post-ship audit pass (this commit's housekeeping)

| # | Item | What changed |
|---|---|---|
| C-2 | Ensemble qualifier on the 95.05% claim | `hf_space/app.py` ABOUT_HTML now says "دقّة الـensemble الكامل (٤ نماذج) ٩٥٫٠٥٪" and "النموذج المنشور هنا هو أفضل نموذج فردي من هذا الـensemble". `MODEL_CARD.md` Performance section relabeled to "Metric (ensemble)" with explicit caveat that the deployed single model has not been re-measured separately. |
| M-9 | "98K" → "95K" | Hero copy (line ~1913) and ABOUT_HTML (line ~1859) both now read ٩٥ ألف (rounded) — matches the actual 95,391-row training set. Hero copy also dropped the bare "بدقّة ٩٥٪" claim since the headline figure is now in the qualified ABOUT_HTML only. |
| I-8 | Defensive `clean()` in `extract_aspects` | Added idempotent `cleaned_text = clean(cleaned_text)` at the function head. Future callers passing raw text no longer silently fail phrase matching. |
| Code audit XSS | `html.escape` in `render_understanding` + `annotate_text` | All user-text interpolations now go through `html.escape()`. The architecture is defensive-by-default; the empty path (`<div class="understanding-text">{cleaned_text}</div>`) and the aspect-chip-evidence path are no longer fragile. |
| Pre-existing test bugs | `tests/test_inference_logic.py` | `test_strips_punctuation` now correctly asserts Arabic comma `،` is KEPT (it's part of normal Arabic text). `test_preserves_arabic_indic_digits` revealed a real bug in `clean_arabic`: the TASHKEEL regex `[ً-ٰٟؐ-ؚ]` ate Arabic-Indic digits at U+0660..U+0669. Fixed by aligning the regex with the Space's `[ً-ٟؐ-ؚ]` (which already excluded digits). Both tests now pass. The deployed Space was never affected — only the API code path. |
| `app/api.py` syntax | Pre-existing f-string bug at line 88 | Fixed in the ship commit (`ffc761f`). |

---

## Pre-existing test failures (not introduced by this commit, not blocking ship)

| Test | Reason | Action |
|---|---|---|
| `tests/test_inference_logic.py::TestCleanArabic::test_strips_punctuation` | Test expects `clean()` to strip Arabic comma `،` but the implementation keeps it. Probably the test's expectation was wrong — Arabic comma is normal text. | Defer; not blocking ship. Worth a separate "test expectations vs implementation" reconciliation pass. |
| `tests/test_inference_logic.py::TestCleanArabic::test_preserves_arabic_indic_digits` | Test expects `clean("ريال 50")` to contain either `١٠` or `10`; actual output contains `50` but the assertion specifies only `١٠` and `10`. Test typo. | Same as above. |

---

## Determinism & reproducibility

- **Fixture generation:** `scripts/build_production_test_500.py` is fully deterministic — all 500 rows are hand-written literals in the source. Re-running produces byte-identical output.
- **Live-Space test:** `scripts/run_test_500_against_live_space.py` is throttled at 0.4s per call (~3.5 min total), with mild backoff retry. Two consecutive runs against the same commit should produce identical results modulo any model-side nondeterminism.
- **Local logic tests:** `tests/test_space_logic.py` uses stubs for both gradio and transformers — runs in milliseconds, no network, no model weights.
- **Pytest baseline:** 29/29 new space-logic tests pass. 80/87 broader-suite tests pass (7 failures pre-date this work and are unrelated to ship logic).

---

## Open questions documented in `adversarial_findings.md`

1. PySarf partial-failure determinism (mid-text exceptions).
2. Behavior of the new single-word rail path under model uncertainty (no confidence indicator).
3. Verification that `production_labels_only.csv` was filtered to 8 categories.
4. Byte-for-byte match between live-Space SHA `6f91ad5` and the working tree.
5. Performance of `clean()` on very long inputs (now bounded by the 4000-char cap from I-10).

None block ship. All worth knowing for the next iteration.

---

## Ship decision rationale

**Why "ship-with-known-gaps":**
- Code review, infrastructure, and architectural audits all passed.
- The Adversarial Q&A reviewer's important findings (I-1/I-2/I-3/I-5/I-10/M-2 plus C-1 docstring honesty) are all fixed in this commit.
- The 29-row `tests/test_space_logic.py` adds the safety net the Adversarial Q&A reviewer correctly flagged was missing — this work cannot break silently going forward.
- The live Space is functional and delivers a measurably useful experience for in-distribution Arabic inputs.
- The 500-row Saudi/Gulf fixture (57.0%) is a more honest measurement than the model card's 95.05% (same-distribution test set). Both are true; they measure different things.

**Why not "ship-clean":**
- The 38-point gap between same-distribution test (95.05%) and cross-distribution test (57.0%) is the most important finding in this audit. It does NOT mean the model is broken — it means the model card's headline number understates the difficulty of real-world deployment.
- النظافة at 34% and OOD at 10% are real weaknesses; users typing about bathroom cleanliness or off-topic text will see incorrect predictions about half the time.

**The two most important next steps:**
1. Update the Space's headline copy to reflect cross-distribution honesty — either remove the 95.05% prominence or pair it with the 57% fresh-fixture number. (Already removed in this commit's UI changes per user direction.)
2. Use the 500-row failures as a labeling backlog. The 215 failures are 215 free training samples that, if labeled and added to training, would close part of the distribution gap. Multi-aspect retraining is the structural fix (already documented as a v6 follow-up).

---

**Generated by:** Claude (Anthropic) — orchestration, synthesis, fixes
**Subagent reports:** `code_audit.md`, `production_audit.md`, `test_run_results.md`, `adversarial_findings.md` — all in `dataset/_audits/`
**New artifacts:** `tests/test_space_logic.py`, `tests/fixtures/production_test_500.csv`, `scripts/build_production_test_500.py`, `scripts/run_test_500_against_live_space.py`
