# How to Reproduce the Full Annotated Dataset

The published `annotations_qaym_*.csv` and `annotations_hard_*.csv` files contain **filter decisions only — no source text** (legally clean republication). To get the full text + annotations together, join back to the upstream HuggingFace datasets.

## Quick recipe

```python
import pandas as pd
from datasets import load_dataset

# 1. Load our annotations (filter decisions only)
ann = pd.read_csv("annotations_qaym_ambience_candidates.csv", encoding="utf-8-sig")

# 2. Load the upstream source dataset
ds = load_dataset("hadyelsahar/ar_res_reviews", split="train")
src_df = ds.to_pandas()

# 3. Join. Our `id` (or `row_id`) is the row index in the upstream dataset.
src_df = src_df.reset_index(drop=False).rename(columns={"index": "row_id"})
joined = ann.merge(src_df, on="row_id", how="left")

# joined has both the source `text` (and any source metadata) plus our filter
# decisions: weak_label, ambience_subtype, risk_keyword, etc.
print(joined.head())
```

## Per-source recipes

### qaym (`hadyelsahar/ar_res_reviews`)

```python
import pandas as pd
from datasets import load_dataset

ds = load_dataset("hadyelsahar/ar_res_reviews", split="train")
src_df = ds.to_pandas().reset_index(drop=False).rename(columns={"index": "row_id"})

for bucket in ["ambience_candidates", "hard_negative_candidates", "review_needed"]:
    ann = pd.read_csv(f"annotations_qaym_{bucket}.csv", encoding="utf-8-sig")
    full = ann.merge(src_df, on="row_id", how="left")
    full.to_csv(f"qaym_{bucket}_full.csv", index=False, encoding="utf-8-sig")
    print(f"{bucket}: {len(full)} rows joined")
```

Source dataset: <https://huggingface.co/datasets/hadyelsahar/ar_res_reviews>
Source attribution: reviews from qaym.com, collected by Hady ElSahar.

### HARD (Hotel Arabic Reviews Dataset)

```python
import pandas as pd
from datasets import load_dataset

ds = load_dataset("HARD", split="train")
src_df = ds.to_pandas().reset_index(drop=False).rename(columns={"index": "row_id"})

for bucket in ["ambience_candidates", "hard_negative_candidates", "review_needed"]:
    ann = pd.read_csv(f"annotations_hard_{bucket}.csv", encoding="utf-8-sig")
    full = ann.merge(src_df, on="row_id", how="left")
    full.to_csv(f"hard_{bucket}_full.csv", index=False, encoding="utf-8-sig")
```

Source dataset: <https://huggingface.co/datasets/HARD>
HARD is a hotel-review dataset; useful for ambience because hotel reviews focus on the physical place. Hotel-only terms (غرفة, الفندق, الإقامة, etc.) appear frequently — filter as needed for restaurant transfer.

## Reproducing the filter from scratch

If you want to re-run our filtering pipeline on a fresh download (or on different data), the full parent project is at:

<https://github.com/FerasMad/NLP-complaints-system>

Specifically:

```bash
git clone https://github.com/FerasMad/NLP-complaints-system.git
cd NLP-complaints-system
pip install -r requirements.txt

# Pull qaym and run the filter
python src/data/import_hf_datasets_for_ambience.py \
    --dataset hadyelsahar/ar_res_reviews \
    --output-dir data/processed/ambience/huggingface

# Pull HARD and run the filter
python src/data/import_hf_datasets_for_ambience.py \
    --dataset HARD \
    --output-dir data/processed/ambience/huggingface_hard
```

Outputs match the published annotations exactly (deterministic seed).

## What to do with the candidates

These are **filter outputs, not gold labels**. The filter has high precision on hand-written test data (~100%) but drops to ~50-70% on real reviews because real users mix multiple complaint types in a single review.

Recommended workflow:

1. **Triage by hand.** Open the candidate CSVs in Excel, read each row, set the final category. Use `dataset/docs/BOUNDARIES.md` as the rulebook.
2. **Approved rows** → `data/text/complaints_labeled_ambience.csv` (per the parent project's schema).
3. **Train.** See the parent project's `src/train_ambience_v5.py`.

Even unlabeled, the candidates are useful as **weak supervision** — train on them with `label_confidence=weak` and the model averages out the noise. Our v5 model used 556 weak qaym candidates + 5,000 filtered HARD candidates this way.

## Don't do this

- **Don't republish raw upstream text** without checking the source dataset's license. The annotations files in this repo deliberately exclude text for that reason.
- **Don't treat filter decisions as gold labels** for evaluation. Always use a hand-labeled held-out set (`adversarial_test_set.csv` is one).
- **Don't fine-tune purely on hotel-domain text** if your target is restaurants. The HARD candidates need a hotel-term filter (see `src/data/build_v5_training_data.py:--include-hard` in the parent repo).
