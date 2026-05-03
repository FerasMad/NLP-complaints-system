# Model Card — Arabic Restaurant Complaints Classifier

## Model overview

- **Name:** Arabic Restaurant Complaints Classifier — 4-model ensemble (8-class)
- **Date:** 2026-05-03
- **Type:** Text classification — single-label multiclass (8 categories)
- **Base architectures (4-model ensemble):**
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix` (seed=42)
  - `CAMeL-Lab/bert-base-arabic-camelbert-mix` (seed=2024, multi-seed for diversity)
  - `UBC-NLP/MARBERT` (Twitter-trained, dialect-rich)
  - `aubmindlab/bert-base-arabertv02`
- **Inference:** Average of softmax probabilities; argmax prediction. Optional temperature (T=1.5225) for calibration.
- **Output:** category label (one of 8) + confidence + top-3 with confidences. Returns `category=null` with `abstain_reason` when input isn't Arabic enough.

## Intended use

- Triage of Arabic restaurant complaints (Saudi/Gulf dialect — by design) to route to the correct operational category.
- Suitable for: customer-feedback dashboards, app-review triage, demo/portfolio use.
- **Not** suitable for: medical, legal, financial, or safety-critical decisions; high-stakes content moderation; languages other than Arabic.

## Categories

| ID | Arabic | Description |
|----|--------|-------------|
| 0 | التوصيل | Delivery — late, missing, wrong address, driver issues |
| 1 | السعر والقيمة | Price and value |
| 2 | النظافة | Cleanliness, hygiene |
| 3 | جودة الطعام | Food quality — taste, freshness, portion |
| 4 | خدمة الموظفين | Staff service and behavior |
| 5 | دقة الطلب | Order accuracy — wrong/missing items |
| 6 | عامة | General fallback |
| 7 | وقت الانتظار | In-restaurant wait time |

> Originally 9 categories. الجو والمكان (ambiance) was dropped after audit revealed only 2 of 171 val+test ambiance samples were truly clean ambiance — see REPORT.md §"Why we dropped".

## Performance (held-out test set, 13,986 real reviews)

| Metric | Point | 95% CI (bootstrap n=1000) |
|---|---:|---|
| Accuracy | 95.05% | [94.70%, 95.41%] |
| Weighted F1 | 95.08% | [94.72%, 95.43%] |
| Macro F1 | 92.03% | [91.20%, 92.87%] |
| Min class F1 (عامة) | 84.84% | [81.0%, 88.3%] |
| Calibration ECE (T=1.523) | 0.014 | — |
| Mean robustness (6 perturbations) | 95.6% | — |

Per-class F1 in [REPORT.md](REPORT.md). Confusion matrix and error analysis at [models/error_analysis/](models/error_analysis/).

## Training data

- **Size:** ~98,000 labeled rows (8 categories) after dropping ambiance.
- **Sources:** production Arabic restaurant complaints dataset, scraped Saudi food delivery app reviews (HungerStation, Jahez, Mrsool, Talabat), template-generated synthetic, back-translation augmented (MarianMT), ChatGPT-generated synthetic, multi-model-agreement pseudo-labeled, EDA-augmented.
- **Split:** stratified 70/15/15 on real samples only. Synthetic and augmented sources are train-only — never in val/test.
- See [DATA_CARD.md](DATA_CARD.md) for full provenance, biases, and known issues.

## Training details

- **Loss:** Class-weighted CrossEntropyLoss (balanced)
- **Optimizer:** AdamW (default Hugging Face Trainer)
- **Schedule:** Cosine LR with 10% warmup
- **Epochs:** 5
- **Batch size:** 16 (effective 32 with gradient_accumulation_steps=2)
- **Max sequence length:** 192
- **Mixed precision:** fp16 on RTX 4070 (12 GB VRAM)
- **Dominant-class subsampling:** None (uses all 31K food-quality rows)
- **Reproducibility:** all seeded (`random.seed(42)`, `torch.manual_seed(seed)`)
- **Compute per model:** ~25–35 min on a single RTX 4070; 4 models total ~2 hours

## Limitations

- **Single-label.** Real complaints are often multi-aspect (e.g., "cold food + late driver + rude staff"). Model picks the dominant aspect; the other signals appear in the top-3 list with non-trivial confidence.
- **Saudi-specialized by design.** Cross-dialect canary: Saudi 67%, Levantine 60%, Egyptian/MSA-only 50%. For Egyptian/Levantine production use, retrain on data from those dialects.
- **Min-support classes have wider CIs.** دقة الطلب (108 test samples) and عامة (215 test samples) per-class F1 95% CI widths are ±5-7%.
- **Tokenizer normalization is lossy.** Tashkeel removed, alef variants normalized — some dialect-specific signals are erased.

## Ethical considerations

- Trained on real user-generated content (Play Store reviews, production complaints). All texts were already public.
- The API includes a PII scrubber (phone/email/URL masking) before any logged input.
- Predictions can be wrong — use top-3 + confidence as a triage signal, not as a verdict for consequential action.
- The model has no notion of severity or urgency — it only categorizes the topic.

## Maintenance & retraining

- Retrain every 3–6 months on new scraped reviews to handle distribution drift.
- Pipeline is fully reproducible — see [REPORT.md](REPORT.md) §"Reproducibility".
- Pseudo-labeling can be re-run after each new bake-off to expand train data without manual labeling.
- Regression test (`tests/test_regression.py`) blocks PRs that drop accuracy below 92%.

## Deployment

- **Recommended:** the 4-model ensemble at [models/ensemble_final/config.json](models/ensemble_final/config.json) (~1.8 GB total disk).
- **Lighter option:** single best model `models/bakeoff/camelbert-mix_8c_capALL_s2024_v2_final` (~440 MB, val acc 94.90%, test acc 94.86%). Manifest at [models/single_final/config.json](models/single_final/config.json).
- HF Spaces turnkey bundle in [`hf_space/`](hf_space/) — drop into a Gradio Space, set `HF_REPO_ID`, push.
