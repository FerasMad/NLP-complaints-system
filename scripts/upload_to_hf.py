"""Upload the single best CAMeLBERT-mix model to HuggingFace Hub.

Pure-Python — no bash, no shell escaping. Works in any terminal (PowerShell,
Git Bash, cmd, Linux, macOS).

Auth: set HF_TOKEN env var, or run `huggingface-cli login` first.

Usage:
    python scripts/upload_to_hf.py FerasMad
    python scripts/upload_to_hf.py FerasMad my-custom-repo-name
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, whoami

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "models" / "bakeoff" / "camelbert-mix_8c_capALL_s2024_v2_final"


MODEL_CARD = """---
language: ar
license: mit
tags:
- arabic
- classification
- restaurant-complaints
- saudi
- dialectal-arabic
- bert
base_model: CAMeL-Lab/bert-base-arabic-camelbert-mix
---

# Arabic Restaurant Complaints Classifier (CAMeLBERT-mix, 8-class)

Fine-tuned [CAMeLBERT-mix](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix) for 8-class Arabic restaurant complaint classification. Single-best member of the production 4-model ensemble at https://github.com/FerasMad/NLP-complaints-system.

## Performance

| Metric | Value |
|---|---:|
| Test accuracy | 94.86% |
| Test weighted F1 | 94.92% |
| Test macro F1 | 91.22% |
| Min class F1 | 81.6% |

The full 4-model ensemble achieves 95.05% test accuracy. This single model is 4x faster and 4x smaller for ~0.2% accuracy trade-off.

## Categories (8)

0. التوصيل (delivery)
1. السعر والقيمة (price)
2. النظافة (cleanliness)
3. جودة الطعام (food quality)
4. خدمة الموظفين (staff)
5. دقة الطلب (order accuracy)
6. عامة (general)
7. وقت الانتظار (wait time)

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("{repo}")
model = AutoModelForSequenceClassification.from_pretrained("{repo}")

text = "الاكل بايخ ومالح والطبخ مو متقن"
inputs = tok(text, return_tensors="pt", truncation=True, max_length=192)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits[0], dim=-1)
print(probs.argmax().item())  # 3 (جودة الطعام)
```

## Training

- Base: `CAMeL-Lab/bert-base-arabic-camelbert-mix`
- 5 epochs, batch 16 x grad_accum 2, AdamW, cosine LR with 10% warmup
- Class-weighted CrossEntropy loss
- max_length 192
- All 31K dominant-class samples (no subsampling)
- EDA augmentation on minority classes (دقة الطلب, عامة)
- Seed: 2024

Training data: ~98K labeled Arabic complaints from production data + scraped Saudi food delivery app reviews.

## Limitations

- Single-label classification — for multi-aspect complaints, use top-3.
- Saudi/Gulf dialect best; cross-dialect performance not measured.
- Not for safety-critical decisions.

## License

MIT (see https://github.com/FerasMad/NLP-complaints-system/blob/main/LICENSE).
"""


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/upload_to_hf.py <hf-username> [repo-name]")
        return 1

    user = sys.argv[1]
    repo_name = sys.argv[2] if len(sys.argv) > 2 else "arabic-complaints-classifier"
    repo_id = f"{user}/{repo_name}"

    if not SOURCE.exists():
        print(f"ERROR: source model dir not found: {SOURCE}")
        return 1

    print("Checking authentication...")
    try:
        info = whoami()
        print(f"  Logged in as: {info['name']}")
    except Exception as e:
        print(f"ERROR: not authenticated. Run `huggingface-cli login` or set HF_TOKEN env var.")
        print(f"  Detail: {e}")
        return 1

    api = HfApi()

    print(f"\nCreating repo: {repo_id}")
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"  -> https://huggingface.co/{repo_id}")

    # Write the model card to the source dir temporarily so it gets uploaded
    readme_path = SOURCE / "README.md"
    readme_existed = readme_path.exists()
    if readme_existed:
        original = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(MODEL_CARD.format(repo=repo_id), encoding="utf-8")

    try:
        files = sorted(p.name for p in SOURCE.iterdir() if p.is_file())
        total_mb = sum(p.stat().st_size for p in SOURCE.iterdir() if p.is_file()) / 1e6
        print(f"\nUploading {len(files)} files ({total_mb:.0f} MB) from {SOURCE.name}/")
        for f in files:
            print(f"  {f}")

        api.upload_folder(
            folder_path=str(SOURCE),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model",
        )
        print(f"\nDone. Model live at https://huggingface.co/{repo_id}")
        print("\nNext: create a Gradio Space at https://huggingface.co/new-space")
        print(f"  Name it: {repo_name}")
        print(f"  Then push hf_space/ contents to that Space repo.")
        return 0
    finally:
        # Restore original README if there was one, else remove ours
        if readme_existed:
            readme_path.write_text(original, encoding="utf-8")
        else:
            readme_path.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
