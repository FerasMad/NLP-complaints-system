# Deploy to HuggingFace Spaces

This directory is turnkey. Three steps from a fresh HF account to a live public Space.

## Prerequisites

1. HF account → https://huggingface.co/join
2. Write token → https://huggingface.co/settings/tokens (role: write)

## 1. Push the model to HF Hub (~10 min)

```bash
pip install huggingface_hub
huggingface-cli login

huggingface-cli repo create arabic-complaints-camelbert-mix --type model

cd /tmp
git clone https://huggingface.co/<your-username>/arabic-complaints-camelbert-mix
cd arabic-complaints-camelbert-mix
cp -r "<project>/models/bakeoff/camelbert-mix_8c_capALL_s2024_v2_final/." .

cat > README.md <<'EOF'
---
language: ar
license: mit
tags: [arabic, classification, restaurant-complaints, saudi, dialectal-arabic, bert]
base_model: CAMeL-Lab/bert-base-arabic-camelbert-mix
---
# Arabic Restaurant Complaints Classifier (CAMeLBERT-mix, 8-class)

Fine-tuned CAMeLBERT-mix on 8 categories of Arabic restaurant complaints. Single-best member of the production 4-model ensemble at https://github.com/FerasMad/NLP-complaints-system.

Test accuracy 94.86% (single). Full ensemble 95.05%. Per-class F1 ≥84%.
EOF

git lfs install
git lfs track "*.safetensors" "*.bin"
git add . && git commit -m "Add model" && git push
```

## 2. Create the Space (1 min)

1. https://huggingface.co/new-space
2. Name: `arabic-complaints-classifier`
3. License: MIT
4. SDK: Gradio
5. Hardware: `cpu-basic` (free)
6. Visibility: Public

## 3. Push the app (1 min)

```bash
git clone https://huggingface.co/spaces/<your-username>/arabic-complaints-classifier /tmp/space
cp hf_space/* /tmp/space/
cd /tmp/space

# Point at your model repo
sed -i 's|FerasMad/arabic-complaints-camelbert-mix|<your-username>/arabic-complaints-camelbert-mix|g' app.py

git add . && git commit -m "Deploy" && git push
```

Build takes ~3 min. First load ~30 s (cold), then instant.

## Update the model later

Push new weights as a new commit on the HF Hub model repo. The Space picks them up on next cold start. Pin a specific revision via the `HF_REVISION` env var in Space settings.

## Troubleshooting

**"OSError: model not found"** — Model repo not public. Either make it public or add `HF_TOKEN` as a secret in Space settings.

**Want the full 4-model ensemble?** Use `app/space_app.py` from this repo instead of `hf_space/app.py`, and push all 4 model dirs to HF Hub. ~1.8 GB total.
