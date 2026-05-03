#!/usr/bin/env bash
# Push the single-best CAMeLBERT-mix model to HuggingFace Hub.
#
# Prereq: pip install huggingface_hub && huggingface-cli login
#
# Usage:
#   bash scripts/upload_to_hf.sh <hf-username> [repo-name]
#   bash scripts/upload_to_hf.sh FerasMad arabic-complaints-camelbert-mix

set -euo pipefail

USER="${1:-}"
REPO_NAME="${2:-arabic-complaints-camelbert-mix}"

if [ -z "$USER" ]; then
    echo "Usage: $0 <hf-username> [repo-name]"
    exit 1
fi

REPO="$USER/$REPO_NAME"
SOURCE="models/bakeoff/camelbert-mix_8c_capALL_s2024_v2_final"

if [ ! -d "$SOURCE" ]; then
    echo "Source model dir not found: $SOURCE"
    exit 1
fi

if ! command -v huggingface-cli >/dev/null; then
    echo "huggingface-cli not installed. Run: pip install huggingface_hub"
    exit 1
fi

echo "Creating HF model repo: $REPO"
huggingface-cli repo create "$REPO_NAME" --type model --y || echo "(repo may already exist; continuing)"

CLONE_DIR="/tmp/hf_upload_$$"
echo "Cloning to $CLONE_DIR"
git clone "https://huggingface.co/$REPO" "$CLONE_DIR"

echo "Copying model artifacts..."
cp -r "$SOURCE/." "$CLONE_DIR/"

cat > "$CLONE_DIR/README.md" <<EOF
---
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

\`\`\`python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("$REPO")
model = AutoModelForSequenceClassification.from_pretrained("$REPO")

text = "الاكل بايخ ومالح والطبخ مو متقن"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192)
with torch.no_grad():
    probs = torch.softmax(model(**inputs).logits[0], dim=-1)
print(probs.argmax().item())  # 3 (جودة الطعام)
\`\`\`

## Training

- Base: \`CAMeL-Lab/bert-base-arabic-camelbert-mix\`
- 5 epochs, batch 16 × grad_accum 2, AdamW, cosine LR with 10% warmup
- Class-weighted CrossEntropy loss
- max_length 192
- All 31K dominant-class samples (no subsampling)
- EDA augmentation on minority classes (دقة الطلب, عامة)
- Seed: 2024

Training data: ~98K labeled Arabic complaints from production data + scraped Saudi food delivery app reviews. Full provenance in [DATA_CARD.md](https://github.com/FerasMad/NLP-complaints-system/blob/main/DATA_CARD.md).

## Limitations

- Single-label classification — for multi-aspect complaints, use top-3.
- Saudi/Gulf dialect best; cross-dialect performance not measured.
- Not for safety-critical decisions.

## License

MIT (see https://github.com/FerasMad/NLP-complaints-system/blob/main/LICENSE).
EOF

cd "$CLONE_DIR"
git lfs install
git lfs track "*.safetensors" "*.bin"
git add .
git commit -m "Upload single-model checkpoint"
echo "Pushing to HF Hub..."
git push

echo ""
echo "Done. Model at https://huggingface.co/$REPO"
echo "Next: deploy the Space — see hf_space/HOW_TO_DEPLOY.md step 2-3"
