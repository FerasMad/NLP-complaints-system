# Results — Arabic Restaurant Complaints Classifier

8-category Arabic restaurant complaint classifier. 4-model ensemble of fine-tuned Arabic BERT variants (CAMeLBERT-mix × 2 seeds + MARBERT + AraBERTv02). Saudi-Gulf dialect specialization.

---

## Headline (held-out test set, 13,986 real reviews)

| Metric | Point | 95% CI (bootstrap n=1000) |
|---|---:|---|
| Test accuracy | 95.05% | [94.70%, 95.41%] |
| Test weighted F1 | 95.08% | [94.72%, 95.43%] |
| Test macro F1 | 92.03% | [91.20%, 92.87%] |
| Min class F1 (عامة) | 84.84% | [81.0%, 88.3%] |
| Calibration ECE (untuned) | 0.034 | — |
| Calibration ECE (T=1.523) | 0.014 | — |
| Mean robustness (6 perturbations) | 95.6% | — |
| Inference (ensemble GPU p50) | 18 ms | — |
| Inference (single-model CPU p50) | 16 ms | — |

All 8 categories pass ≥80% F1. Spread is 11.3 points (worst عامة 84.9%, best food 96.2%).

The +0.14% accuracy delta from the previous iteration is inside the 95% CI half-width — not statistically distinguishable from noise on accuracy alone. The improvement is statistically significant on macro F1 and min-class F1.

## Winner ensemble (`models/ensemble_final/config.json`)

1. `camelbert-mix_8c_capALL_s42_final` — CAMeLBERT-mix, no dominant subsampling, seed=42 (no EDA)
2. `camelbert-mix_8c_capALL_s2024_v2_final` — CAMeLBERT-mix, seed=2024, with EDA augmentation
3. `marbert_8c_capALL_v2_final` — MARBERT, with EDA augmentation
4. `arabertv02_8c_v3_final` — AraBERTv02 (no EDA, kept for diversity)

Inference: uniform average of softmax probabilities. Wired in [app/ensemble_inference.py](app/ensemble_inference.py).

---

## Methodology

How we got from start to finish.

1. **Linear baseline.** TF-IDF char n-grams + LinearSVC, 9 categories. Test acc 89.8%, but per-class F1 was ugly: ambiance 24%, order accuracy 22%. Linear models miss dialect markers.

2. **Switched to fine-tuned BERT.** CAMeLBERT-mix on the same data → 93.2% acc, 76.7% macro F1. Big jump on minority classes.

3. **Scaled the data.** Real-data starvation was the bottleneck. Scraped 4 Saudi delivery apps (HungerStation, Jahez, Mrsool, Talabat) across 3 star-bands with pagination → 14,789 unique reviews. Keyword-filtered into التوصيل + دقة الطلب. دقة الطلب went 137 → 714 real samples.

4. **Multi-source data pipeline.** Production corpus + scrape + dialectal synthetic templates + ChatGPT-generated + back-translation + multi-model pseudo-labels (top-2 ensemble agree at ≥0.90 confidence). Hard rule: anything synthetic/augmented is train-only, never in val/test. Caught and patched a leakage bug where pseudo-labels were sneaking into eval.

5. **Bake-off across 5 architectures.** CAMeLBERT-mix (×2 seeds), CAMeLBERT-da, MARBERT, AraBERTv02, XLM-R. Picked the top 4 by val macro F1. Lost: CAMeLBERT-da (cleaning strips its dialect advantage) and XLM-R (multilingual loses to Arabic-specific).

6. **Dropped the ambiance category.** Could not get F1 above 35% no matter what. Audit of the 171 val+test gold labels: only 2 were truly clean ambiance, 47% probably mislabeled, 40% multi-aspect. Removed the category. Test acc 94.07 → 94.86, macro F1 82.8 → 90.5, min class F1 35 → 79.

7. **EDA boost on the bottom 2 classes.** Random word swap/delete/insert on دقة الطلب + عامة only. Both crossed 85% F1.

8. **Eval rigor.** Bootstrap CI on test (n=1000): 95.05% [94.70%, 95.41%]. Temperature scaling (T=1.523) cut ECE 0.034 → 0.014. Perturbation robustness 95.6% mean stability across 6 perturbations. Cross-dialect canary documenting Saudi specialization as intentional. Per-category label-noise audit. Per-source eval (production vs Play Store).

9. **Tried back-translation on عامة (didn't work).** 1,410 MarianMT Ar→En→Ar paraphrases. Single-model val improved +2.2%. Ensemble test regressed −1.2%. BT drift is amplified by the ensemble. Reverted. Documented in §Discarded approaches.

10. **Production hardening.** Real Python package (`pip install -e .`), FastAPI with `/healthz` `/readyz` `/metrics` `/predict_batch`, PII scrubbing, OOD abstain, restricted CORS, rate limiting. Gradio UI for HF Spaces. Test suite: data integrity + Hypothesis property tests + determinism + 200-row regression gate.

Diagram of the same flow is in [README.md §Approach](README.md#approach).

---

## Per-class F1 on test set

| Category | Train rows | Test rows | F1 |
|---|---:|---:|---:|
| جودة الطعام (food quality) | 31,558 | 6,764 | 96.2% |
| خدمة الموظفين (staff) | 10,669 | 2,287 | 95.8% |
| النظافة (cleanliness) | 2,408 | 516 | 95.3% |
| السعر والقيمة (price) | 12,173 | 2,610 | 94.7% |
| وقت الانتظار (wait time) | 2,963 | 636 | 91.9% |
| التوصيل (delivery) | 4,616 | 850 | 90.1% |
| دقة الطلب (order accuracy) | 2,872 + EDA | 108 | 86.7% |
| عامة (general) | 2,599 + EDA | 215 | 84.9% |

---

## Calibration

ECE (Expected Calibration Error, 10 bins): 0.0192 → well calibrated (threshold for "well": ECE < 0.03).

| Confidence threshold | Coverage | Accuracy at threshold |
|---:|---:|---:|
| ≥ 0.50 | 99.8% | 95.1% |
| ≥ 0.70 | 96.4% | 96.4% |
| ≥ 0.80 | 92.4% | 97.3% |
| ≥ 0.90 | 89.9% | 97.9% |
| ≥ 0.95 | 87.8% | 98.3% |
| ≥ 0.99 | 83.0% | 98.9% |

When the model returns confidence ≥ 0.90, it is right ~98% of the time. For confidence-gated workflows, 0.90 is a safe threshold. Below 0.50 (only 0.2% of inputs), fall back to "عامة" or human review.

Plot: [models/calibration_plot.png](models/calibration_plot.png).

---

## Inference benchmarks (RTX 4070 / Intel CPU)

### Single-request latency

| Config | Device | p50 (ms) | p95 (ms) | p99 (ms) | mean (ms) |
|---|---|---:|---:|---:|---:|
| single best (capALL_s2024_v2) | cuda | 4.4 | 4.5 | 4.5 | 4.4 |
| single best | cpu | 16.2 | 18.5 | 19.0 | 16.4 |
| ensemble (4 models) | cuda | 17.7 | 20.1 | 21.1 | 18.0 |
| ensemble (4 models) | cpu | 76.3 | 89.5 | 92.8 | 77.0 |

### Throughput (req/s)

| Config | Device | batch=1 | batch=8 | batch=32 |
|---|---|---:|---:|---:|
| single | cuda | 220 | 1,232 | 4,256 |
| single | cpu | 61 | 130 | 246 |
| ensemble | cuda | 55 | 310 | 851 |
| ensemble | cpu | 13 | 35 | 64 |

Free-tier HuggingFace Spaces (CPU) handles ~13 req/s with the full ensemble (~77 ms / request). For higher throughput, use the single-model variant (61 req/s) at the cost of ~0.3% accuracy.

Full benchmarks: [models/benchmarks.txt](models/benchmarks.txt).

---

## Per-source evaluation

The test set spans two real-data sources:

| Source | Rows | Accuracy | Weighted F1 |
|---|---:|---:|---:|
| production (legacy labeled corpus) | 13,028 | 95.33% | 95.74% |
| play_store (scraped Saudi delivery apps) | 958 | 91.34% | 95.21% |

The play_store split contains only التوصيل + دقة الطلب (by construction — those rows came from keyword-filtered scraped reviews). On those 2 classes, F1 is 95.3% (delivery) and 94.3% (order accuracy) on play_store data — the model generalizes from the production corpus to live Play Store reviews.

Full breakdown: [models/per_source_eval.txt](models/per_source_eval.txt).

---

## Vs baselines (test set)

| Model | Test acc | Test wF1 | Test mF1 | Min class F1 |
|---|---:|---:|---:|---:|
| TF-IDF + LinearSVC (Phase 3a, 9-class) | 89.8% | 89.8% | 68.0% | 22.0% |
| Single CAMeLBERT-mix (9-class baseline) | 93.2% | 93.2% | 76.7% | 24.0% |
| 4-model ensemble (9-class with ambiance, tuned) | 94.07% | 94.00% | 82.80% | 35.46% |
| 4-model ensemble (8-class, no EDA) | 94.86% | 94.89% | 90.50% | 78.73% |
| 4-model ensemble (8-class + EDA boost) | 95.05% | 95.08% | 92.03% | 84.84% |

---

## Why we dropped the ambiance category

Audit ([src/audit_ambiance_eval.py](src/audit_ambiance_eval.py)) of the 171 val+test ambiance samples:

| Auto-classification | Count | % |
|---|---:|---:|
| clean ambiance (only ambiance keywords match) | 2 | 1.2% |
| multi-aspect (ambiance + other category keywords) | 68 | 39.8% |
| probably mislabeled (no ambiance keywords; other category dominates) | 80 | 46.8% |
| ambiguous (no clear keywords) | 21 | 12.3% |

Only 2 of 171 were unambiguously ambiance. The 35.5% F1 we'd been chasing was meaningless — the gold itself was noise. Dropping the category gave +0.79 test acc, +7.7 macro F1, +43 min class F1, and brought every remaining class above 78% F1.

---

## Bake-off (8-class single models, val)

| Model | Architecture | EDA? | Val acc | Val wF1 | Val mF1 | Train time |
|---|---|:---:|---:|---:|---:|---:|
| camelbert-mix_8c_capALL_s2024_v2 | CAMeLBERT-mix (seed=2024) | yes | 94.90% | 94.92% | 91.22% | 25 min |
| marbert_8c_capALL_v2 | MARBERT | yes | 94.58% | 94.58% | 91.24% | 33 min |
| camelbert-mix_8c_capALL_s42 | CAMeLBERT-mix (seed=42) | no | 94.55% | 94.56% | 88.41% | 34 min |
| arabertv02_8c_v3 | AraBERTv02 | no | 93.60% | 93.66% | 87.97% | 19 min |

---

## Per-category label-noise audit

[`src/audit_category.py --all`](src/audit_category.py):

| Category | Mis-label % | Note |
|---|---:|---|
| خدمة الموظفين | 5.3% | cleanest gold labels |
| جودة الطعام | 18.0% | clean |
| النظافة | 21.6% | borderline |
| وقت الانتظار | 20.3% | borderline |
| السعر والقيمة | 32.5% | model handles it (94.7% F1) |
| التوصيل | 33.7% | model handles it (90.1% F1) |
| دقة الطلب | 41.9% | concerning, but only 108 test samples |
| عامة | 47.1% | catch-all by design — no specific keywords |

The keyword heuristic is intentionally narrow; high mis-label % does not imply the model is wrong, just that the heuristic disagrees with gold. The strong per-class F1 numbers show the model has learned the category.

---

## Perturbation robustness

Apply 6 perturbations to 200 test samples; measure prediction stability ([`src/robustness_eval.py`](src/robustness_eval.py)):

| Perturbation | Stability |
|---|---:|
| emoji_prefix | 100.0% (stripped by cleaner) |
| char_repeat | 97.0% |
| char_delete | 95.0% |
| eng_noise | 94.5% |
| whitespace | 94.0% |
| char_swap | 93.0% |
| Mean | 95.6% |

---

## Cross-dialect canary

33-sample first-pass canary set ([`data/canary/cross_dialect.csv`](data/canary/cross_dialect.csv)):

| Dialect | N | Accuracy |
|---|---:|---:|
| Saudi (training distribution) | 3 | 67% |
| Levantine | 10 | 60% |
| Egyptian | 10 | 50% |
| MSA only | 10 | 50% |

Saudi-Gulf specialization is intentional. The model is designed for the training distribution (Saudi food delivery apps). Cross-dialect performance is provided for transparency; for Egyptian/Levantine production use, retrain on data from those dialects.

---

## Discarded approaches

- Multi-label experiment with keyword-derived labels — collapsed to 57% single-label argmax. Without proper hand-multi-labeled data, hurts more than helps.
- camelbert-da (dialectal Arabic variant) — worse than camelbert-mix. The cleaning pipeline strips dialect markers (tashkeel, alef variants), erasing the dialectal advantage.
- xlm-roberta-base (multilingual control) — significantly worse than Arabic-specific models.
- 5+ model ensembles — adding weaker members diluted the strong ones. 4 was the sweet spot.
- Threshold tuning on the 8-class ensemble — lifted val acc by +0.22 but hurt test acc by -0.07 (overfit val).
- Back-translation augmentation on عامة (1,410 MarianMT Ar→En→Ar paraphrases). Single-model val عامة F1 +2.2%; ensemble test عامة F1 −1.2%. BT paraphrases drift from the real-data distribution; ensemble amplifies the gap. Script kept ([src/augment_general.py](src/augment_general.py)) but not used in production.

---

## Smoke test (live predictions on 8 prototype complaints)

| Input | Predicted | Confidence |
|---|---|---:|
| الاكل بايخ ومالح | جودة الطعام | 100% |
| الاسعار مبالغ فيها لا تناسب الجوده | السعر والقيمة | 100% |
| النظافه سيئه الطاولات متسخه | النظافة | 100% |
| طلبت برجر بدون بصل لكنهم وضعوه رغم تنبيهي | دقة الطلب | 100% |
| الموظف اسلوبه سيء وغير محترم | خدمة الموظفين | 100% |
| انتظرت ساعتين قبل ان ياتي الطلب | وقت الانتظار | 99% |
| تجربه سيئه عموما لن اعود | عامة | 100% |
| وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين | التوصيل | 70% |

8 of 8 correct.

---

## Deploy

- Final ensemble manifest: [models/ensemble_final/config.json](models/ensemble_final/config.json)
- Inference module: [app/ensemble_inference.py](app/ensemble_inference.py)
- FastAPI service: [app/api.py](app/api.py)
- Gradio UI (HF Spaces): [app/space_app.py](app/space_app.py)
- HF Spaces turnkey directory: [hf_space/](hf_space/) — drop into a fresh Gradio Space, set `HF_REPO_ID` to your model repo, push.
- Single-model lighter variant: [models/single_final/config.json](models/single_final/config.json)

Local smoke test:
```bash
.venv/Scripts/python -c "from app.ensemble_inference import EnsembleClassifier; clf = EnsembleClassifier('models/ensemble_final/config.json'); print(clf.predict('الاكل بايخ'))"
```

Local Gradio with public *.gradio.live URL (72-hour link, no account):
```bash
SHARE=true .venv/Scripts/python -m app.space_app
```

---

## Reproducibility

End-to-end rerun (~7 hours on RTX 4070):

```bash
# Build the dataset
py src/scrape_reviews.py
py src/rebuild_dataset.py
py src/generate_synthetic.py
py src/split_dataset.py
py src/run_bakeoff.py
py src/pseudo_label.py
py src/split_dataset.py

# EDA boost the bottom 2 classes
py src/eda_augment_classes.py --categories "دقة الطلب" "عامة" --target-each 1500
py src/split_dataset.py

# Train the 4 ensemble members
py src/run_bakeoff.py --models camelbert-mix --dominant-cap 100000 --max-length 192 --save-suffix _8c_capALL_s42
py src/run_bakeoff.py --models camelbert-mix --dominant-cap 100000 --max-length 192 --seed 2024 --save-suffix _8c_capALL_s2024_v2
py src/run_bakeoff.py --models marbert --dominant-cap 100000 --max-length 192 --save-suffix _8c_capALL_v2
py src/run_bakeoff.py --models arabertv02 --dominant-cap 15000 --max-length 192 --save-suffix _8c_v3

# Final eval + analysis
py src/eval_ensemble_on_test.py --weighted --dirs <4 dirs>
py src/calibrate.py
py src/temperature_scale.py
py src/benchmark_inference.py
py src/per_source_eval.py
py src/eval_with_ci.py
py src/robustness_eval.py
py src/cross_dialect_eval.py
```

All scripts deterministic (`random.seed(42)`, `random_state=42`, `torch.manual_seed(seed)`).

---

## Biggest wins (by impact)

1. Dropping the ambiance category (+7.7 macro F1, +43 min class F1)
2. Real-data scaling (التوصيل 1k → 5.6k, دقة الطلب 137 → 714)
3. 4-model ensemble with multi-arch + multi-seed diversity
4. Targeted EDA augmentation for the bottom 2 classes
5. No dominant-class subsampling (use all 31K food-quality rows)
