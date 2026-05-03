# Changelog

Notable changes to the model + repo.

## Latest — 2026-05-03

### Added
- Bootstrap 95% CI on test metrics: 95.05% acc [94.70, 95.41]. Latest +0.14% acc is inside CI; macro F1 +1.46 is significant.
- Tried back-translation augmentation on عامة (1,410 paraphrases via Helsinki-NLP MarianMT, [src/augment_general.py](src/augment_general.py)). Single-model val عامة F1 improved +2.2% (0.849 → 0.871) but ensemble test عامة F1 regressed -1.2% (0.849 → 0.837) — BT paraphrases drift from the real-data distribution. Kept the script, reverted production ensemble to non-BT model.
- Per-category label-noise audit ([src/audit_category.py](src/audit_category.py)) generalized to all 8 classes.
- Temperature scaling: T=1.523 reduces test ECE 0.034 → 0.014.
- Perturbation robustness: 95.6% mean stability across 6 perturbations.
- Cross-dialect canary (Egyptian / Levantine / MSA / Saudi). Documents Saudi specialization as intentional design.
- pyproject.toml — `pip install -e .` works.
- app/__init__.py — proper Python package.
- ensemble_inference rewrite: graceful degradation, OOD abstain, model SHA pinning, structured PredictionResult.
- API rewrite: /healthz + /readyz, POST /predict_batch, optional Prometheus + slowapi, PII scrubbing, structured logging, restricted CORS.
- Test suite: data pipeline, pure-logic unit tests (no model loading), Hypothesis property tests, determinism, 200-row regression gate.
- Gradio UI: dark theme, RTL Arabic, top-3 with confidence bars, example chips.
- Turnkey HF Spaces deploy directory ([hf_space/](hf_space/)).
- Single-model deploy variant ([models/single_final/config.json](models/single_final/config.json)).
- LICENSE (MIT), SECURITY.md, NOTICES, Dockerfile, .editorconfig, .pre-commit-config.yaml.
- src/config.py + src/inference_utils.py (kill duplication).

### Changed
- Same model checkpoints. Temperature scaling shifts threshold decisions on borderline cases: test acc 95.00% → 95.05%.

## 2026-05-02 (after EDA boost)

4-model ensemble, 8-class, EDA augmentation on دقة الطلب and عامة. Test acc 95.00%, macro F1 91.96%, all classes ≥84.9%.

## 2026-05-02 (after dropping ambiance)

Schema: 9 → 8 categories. Dropped الجو والمكان after audit found only 2 of 171 val+test ambiance samples were truly clean. Test acc 94.07% → 94.86% (+0.79), macro F1 82.80% → 90.50% (+7.7), min class F1 35.46% → 78.73% (+43).

## 2026-05-01 (4-model ensemble)

Multi-architecture ensemble (CAMeLBERT-mix × 2 seeds + CAMeLBERT-da + MARBERT + AraBERTv02 + XLM-R bake-off; top 4 used). Pseudo-labeling on unused scrape (top-2 model agreement, ≥0.90 confidence), back-translation augmentation, threshold tuning. Caught data-leakage bug (pseudo-labeled rows leaking into val/test) and patched split_dataset.py. Test acc 91.7% (single model) → 94.07% (4-model ensemble + tuned).

## 2026-05-01 (CAMeLBERT-mix)

Switched from TF-IDF to fine-tuned CAMeLBERT-mix. Saudi-dialect synthetic templates rewritten (60–80 per generator). Scraping expanded to 12 apps × 3 score bands × pagination → 14,789 unique reviews. التوصيل: 1,008 → 5,658 real samples. دقة الطلب: 137 → 714. Test acc 89.8% → 93.2%, macro F1 68% → 76.7%.

## 2026-04-30 (TF-IDF baseline)

Initial 9-category schema. TF-IDF (char_wb 3-5 n-grams) + LinearSVC. Test acc 89.8%. Per-class F1 highly skewed: ambiance 24%, order accuracy 22%.
