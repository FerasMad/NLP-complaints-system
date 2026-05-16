---
language:
  - ar
license: mit
library_name: transformers
pipeline_tag: text-classification
base_model: CAMeL-Lab/bert-base-arabic-camelbert-mix
tags:
  - arabic
  - nlp
  - text-classification
  - bert
  - saudi
  - dialectal-arabic
  - complaints
  - restaurant-reviews
  - aspect-classification
  - ambience-detection
metrics:
  - accuracy
  - f1
model-index:
  - name: arabic-complaints-classifier
    results:
      - task:
          type: text-classification
          name: Arabic Restaurant Complaint Classification
        dataset:
          type: custom
          name: Arabic Restaurant Complaints (held-out test set)
        metrics:
          - type: accuracy
            value: 0.9505
            name: Test accuracy
          - type: f1
            value: 0.9203
            name: Test macro F1
          - type: f1
            value: 0.9508
            name: Test weighted F1
widget:
  - text: "الاكل بايخ ومالح والطبخ مو متقن"
    example_title: "Food quality"
  - text: "انتظرت ساعتين قبل ان ياتي الطلب"
    example_title: "Wait time"
  - text: "الموظف اسلوبه سيء وغير محترم"
    example_title: "Staff service"
  - text: "النظافه سيئه الطاولات متسخه"
    example_title: "Cleanliness"
---

# Arabic Restaurant Complaints Classifier

Single-label classification of Arabic restaurant complaints into 8 actionable categories. Specialized for Saudi-Gulf dialect; trained on ~98K reviews from Saudi delivery platforms (HungerStation, Jahez, Mrsool, Talabat) plus production data.

This model card covers the **production 8-class model** (deployed) and the **v5 ambience experiment** (research-stage 9-class model with a re-introduced الجو والمكان class).

Live demo: <https://huggingface.co/spaces/FerasMad/arabic-complaints-classifier>
Source: <https://github.com/FerasMad/NLP-complaints-system>
Companion dataset (v5 work): <https://huggingface.co/datasets/FerasMad/arabic-restaurant-ambience>

## Categories

| Arabic | English | Test F1 |
|---|---|---:|
| جودة الطعام | Food quality | 96.2% |
| خدمة الموظفين | Staff service | 95.8% |
| النظافة | Cleanliness | 95.3% |
| السعر والقيمة | Price / value | 94.7% |
| وقت الانتظار | Wait time | 91.9% |
| التوصيل | Delivery | 90.1% |
| دقة الطلب | Order accuracy | 86.7% |
| عامة | General (no specific aspect) | 84.9% |

## Performance

Held-out test set of 13,986 real reviews, never seen in training. **These numbers are for the full 4-model ensemble.** The deployed Space uses the single best CAMeLBERT-mix model from this ensemble; its standalone accuracy on this test set has not been re-measured separately.

| Metric (ensemble) | Value |
|---|---:|
| Accuracy | 95.05% |
| Weighted F1 | 95.08% |
| Macro F1 | 92.03% |
| Min class F1 (عامة) | 84.84% |
| Bootstrap 95% CI (accuracy) | [94.70%, 95.41%] |
| ECE (after temperature scaling, T=1.523) | 0.014 |

The keyword-rescue layer in the deployed Space trades 0.93% test accuracy for a 15-point gain on a 34-case behavioral audit (85% → 100%). See the GitHub repo for the rescue logic and audit set.

### Cross-distribution evaluation (post-ship)

A 500-row hand-written Saudi/Gulf fixture (0% overlap with training) was scored against the live Space (commit 6f91ad5):

| Rubric | Pass rate |
|---|---:|
| Strict top-1 | 57.0% |
| Lenient (expected in top-2 rail) | 61.2% |
| Corrected (lenient + triaged-ambiguous) | ≥ 64.2% |

The fixture and audit reports are at `dataset/_audits/` in the GitHub repo. The 30-point gap to the same-distribution test accuracy is partly real misses (~75% of failures), partly legitimately ambiguous inputs (~20%), and partly multi-aspect-undercredit (~5%). See `dataset/_audits/SHIP_READINESS.md` for the full reading.

## Intended use

Triage and routing of Arabic restaurant feedback for Saudi/Gulf operators. Use cases:

- Customer-feedback dashboards: tag each complaint and route to the right team
- Product analytics: aggregate complaint volume per category over time
- Quality programs: prioritize improvement areas by complaint share

## Out-of-scope / known limitations

- **Single-label only.** The model picks one category. Multi-aspect complaints ("food was cold AND staff was rude") are not natively decomposed — the deployed Space adds a heuristic display layer for these. A true multi-label retrain is the next planned improvement.
- **Dialect bias.** Trained almost entirely on Saudi/Gulf dialect. Cross-dialect canary scores (out of ~50 single-aspect probes per dialect):
  - Saudi: ~67%
  - Levantine: ~60%
  - Egyptian / MSA: ~50%
  - For non-Gulf dialects, treat predictions as advisory.
- **Domain bound to restaurants.** The model was trained on restaurant complaints only. Don't apply to other product categories without retraining.
- **No ambience category in production model.** v3 had a "الجو والمكان" class; v4 dropped it after a manual audit found ~99% of the gold labels were noise. The production 8-class model does not predict ambience. **A 9-class v5 experiment exists** that re-introduces ambience — see "V5 ambience experiment" section below. The v5 model achieves **89.22% ambience F1** on a hand-crafted adversarial fixture but is research-stage, not deployed.
- **No abstain mechanism in the raw model.** The deployed Space adds short-input abstain (length < 3 chars) and reframes "عامة" predictions as "no specific aspect detected." If you call the model directly, you won't get those guards.
- **PII handling is the caller's responsibility.** The model has no built-in PII scrubbing.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

REPO = "FerasMad/arabic-complaints-classifier"
LABELS = [
    "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
    "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
]

tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()

text = "الاكل بايخ ومالح والطبخ مو متقن"
inputs = tok(text, return_tensors="pt", truncation=True, max_length=192)
with torch.no_grad():
    probs = model(**inputs).logits.softmax(-1)[0]
top_idx = int(probs.argmax())
print(f"{LABELS[top_idx]}: {probs[top_idx]:.2%}")
# جودة الطعام: 98.4%
```

For temperature-scaled probabilities, divide logits by `T = 1.523` before softmax. For the keyword-rescue layer and aspect-extraction interpretability, use the deployed Space or copy `hf_space/app.py` from the GitHub repo.

## Training details

- **Base model:** CAMeL-Lab/bert-base-arabic-camelbert-mix
- **Architecture:** BERT base (110M params) + classification head over 8 labels
- **Sequence length:** 192 tokens
- **Training data:** ~98K Arabic complaints from Saudi delivery platforms + production data, manually labeled by the AI Club NLP team
- **Augmentation:** EDA (Easy Data Augmentation) for under-represented classes (`دقة الطلب`, `عامة`) only
- **Calibration:** Temperature scaling with T=1.523, fit on the validation set by NLL minimization

The deployed system uses an ensemble of 4 BERT variants (CAMeLBERT-mix × 2 seeds, MARBERT, AraBERTv02) — see the GitHub repo. This single model is the lightest deployable variant and is what powers the public HF Space (free-tier memory constraint).

## Evaluation methodology

- Held-out test set of 13,986 reviews, sampled per-source and stratified per-class
- Bootstrap 95% CI computed over 1000 resamples
- Calibration assessed via ECE on the test set
- Cross-dialect canary set written by hand to probe dialect generalization
- Behavioral audit: 34 hand-written test cases (single + multi-aspect) covering all 8 categories — this is what the deployed rescue layer optimizes for

## V5 ambience experiment (9-class, research-stage)

After the 8-class production model shipped, the v3-dropped ambience class
was re-attempted from scratch with new infrastructure: PySarf morphology,
weak-labeled real data from public HF datasets, MLM continued pre-training
on 45K Arabic restaurant + hotel reviews, focal loss, and threshold-tuned
inference. Three models exist (all research-stage, not deployed):

| Model | Adversarial Ambience F1 | Real-world test Ambience F1 | Best for |
|---|---:|---:|---|
| Run 4 (`models/single_ambience_v1_pretrained`) | **89.22%** | 69.19% | Adversarial benchmark |
| Run 5 (`models/single_ambience_v1_pretrained_v2`) | 88.24% | **94.09%** | Real-world deployment |
| Run 4 + Run 5 ensemble (softmax averaging) | 88.15% | (~80-85% expected) | Best real-world balance |

All three apply a post-hoc decision threshold (`ambience_threshold ∈ [0.005, 0.02]`)
and an abstain wrapper (length gate + restaurant-domain OOD gate) at inference.

### V5 evaluation methodology

- **Adversarial fixture:** 183 hand-written cases across 13 attack types
  (clean, boundary, negation, sarcasm, mixed dialect, multi-aspect, short,
  very_short, long, out_of_domain, typo, emoji, adversarial). Released as
  part of the [companion dataset](https://huggingface.co/datasets/FerasMad/arabic-restaurant-ambience).
- **Real-world val/test:** 15% / 15% stratified splits of the combined
  baseline (95K rows) + 5,000 weak-labeled HARD ambience candidates.

### V5 ambience class metrics (Run 4, threshold 0.01 + abstain)

| Metric | Value |
|---|---:|
| Ambience precision | 86.67% |
| Ambience recall | 91.92% |
| Ambience F1 | 89.22% |
| Per-attack-type pass rate (worst) | out_of_domain 50% |
| Per-attack-type pass rate (best) | sarcasm + emoji + short + very_short = 100% |

### V5 known limitations

- **Multi-aspect cases stuck at 40%** — single-label classification can't
  decompose "الاكل بارد والمكيف خربان". Multi-label retraining is the
  only structural fix.
- **Out-of-domain handling at 50%** — abstain wrapper catches half. A
  dedicated OOD classifier as a preprocessing gate would close more.
- **Trained on weak-labeled real data**, not gold-labeled. The 1,256
  qaym + HARD ambience candidates have ~50-80% precision per sampling.
  Hand-labeling would push F1 higher.

### Reproducing v5

Full waves 7-10 documented at:

- `docs/V5_TRAINING_RUN_1_RESULTS.md` — initial baseline (78.57% F1)
- `docs/V5_TRAINING_RUN_2_THRESHOLD_TUNING.md` — threshold to 85.56%
- `docs/V5_TRAINING_RUN_3_REAL_DATA_AND_ABSTAIN.md` — 24K real candidates harvested
- `docs/V5_TRAINING_RUN_4_PRETRAINED.md` — MLM + weak labels = 89.22%
- `docs/V5_TRAINING_RUN_5_HARD_DATA.md` — 5x data, mixed signal

## Credits

Built by the NLP team at AI Club:

- **Feras Madkhali** — team lead, training (Phase 3), evaluation, deployment
- **Lana** — text cleaning pipeline (`clean()`)
- **Khowla** — labeling and split decisions
- **Rima & Mohammed** — schema design, baseline trials
- **Meshal** — deployment side

Special thanks to **Rashidbm** for [PySarf](https://github.com/Rashidbm/pysarf) — the Arabic morphology engine used in the deployed Space's aspect-extraction layer.

## License

[MIT](https://github.com/FerasMad/NLP-complaints-system/blob/main/LICENSE). Upstream model licenses (CAMeLBERT, MARBERT, AraBERTv02) apply to the base architecture — see the GitHub repo's `NOTICES` file.

## Citation

```bibtex
@misc{madkhali2026arcomplaints,
  title  = {Arabic Restaurant Complaints Classifier},
  author = {Madkhali, Feras and the AI Club NLP Team},
  year   = {2026},
  url    = {https://github.com/FerasMad/NLP-complaints-system},
  note   = {Saudi-Gulf dialect specialization. 8-class production model
            + 9-class v5 ambience experiment (89.22% adversarial F1).
            Companion dataset at huggingface.co/datasets/FerasMad/arabic-restaurant-ambience}
}
```
