"""Perturbation robustness evaluation.

Apply 6 perturbations to test samples and measure prediction stability:
  1. Char swap (random adjacent swap)
  2. Char delete (drop random non-Arabic chars)
  3. Char repeat (extend a vowel like aaaaa)
  4. Emoji prefix (add 🍕 or similar)
  5. Whitespace insertion (extra spaces)
  6. Mixed-case English noise (insert "OK" or "wow")

For each: sample 200 test rows, perturb, check if prediction unchanged.
Higher stability rate = more robust model.

Outputs models/robustness_report.txt.
"""
from __future__ import annotations

import gc
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
LABEL_MAP_JSON = ROOT / "data" / "processed" / "label_map.json"
ENSEMBLE_CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_TXT = ROOT / "models" / "robustness_report.txt"

with open(LABEL_MAP_JSON, encoding="utf-8") as _f:
    NUM_LABELS = len(json.load(_f))
BATCH_SIZE = 64
N_SAMPLES = 200
random.seed(42)


def predict_argmax(model_dirs, texts, device, max_length, biases):
    """Run ensemble, return argmax category indices."""
    stack = np.zeros((len(model_dirs), len(texts), NUM_LABELS), dtype=np.float32)
    for k, d in enumerate(model_dirs):
        tokenizer = AutoTokenizer.from_pretrained(str(d))
        model = AutoModelForSequenceClassification.from_pretrained(str(d)).to(device)
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i : i + BATCH_SIZE]
                enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(device)
                logits = model(**enc).logits
                stack[k, i : i + len(batch)] = torch.softmax(logits, dim=-1).cpu().numpy()
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    avg = stack.mean(axis=0) + biases
    return avg.argmax(axis=1)


# ---- Perturbations ----
def perturb_char_swap(text: str) -> str:
    chars = list(text)
    if len(chars) < 4:
        return text
    i = random.randint(1, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def perturb_char_delete(text: str) -> str:
    chars = list(text)
    if len(chars) < 4:
        return text
    i = random.randint(0, len(chars) - 1)
    return "".join(chars[:i] + chars[i + 1:])


def perturb_char_repeat(text: str) -> str:
    """Extend a random non-space char by 3-5 repeats (like 'مررررره')."""
    chars = list(text)
    if len(chars) < 2:
        return text
    candidates = [i for i, c in enumerate(chars) if c.strip()]
    if not candidates:
        return text
    i = random.choice(candidates)
    n = random.randint(3, 5)
    return "".join(chars[: i + 1] + [chars[i]] * n + chars[i + 1:])


def perturb_emoji(text: str) -> str:
    return random.choice(["🍕 ", "😡 ", "🙄 ", "😤 ", "👎 "]) + text


def perturb_whitespace(text: str) -> str:
    chars = list(text)
    insertions = max(1, len(chars) // 20)
    for _ in range(insertions):
        i = random.randint(0, len(chars))
        chars.insert(i, " ")
    return "".join(chars)


def perturb_eng_noise(text: str) -> str:
    noise = random.choice([" wow ", " ok ", " hmm ", " yeah "])
    i = random.randint(0, len(text))
    return text[:i] + noise + text[i:]


PERTURBATIONS = {
    "char_swap": perturb_char_swap,
    "char_delete": perturb_char_delete,
    "char_repeat": perturb_char_repeat,
    "emoji_prefix": perturb_emoji,
    "whitespace": perturb_whitespace,
    "eng_noise": perturb_eng_noise,
}


def main():
    with open(ENSEMBLE_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    model_dirs = [ROOT / m for m in cfg["models"]]
    max_length = int(cfg.get("max_length", 192))
    biases_dict = cfg.get("tuned_biases", {})

    with open(LABEL_MAP_JSON, encoding="utf-8") as f:
        label_map = json.load(f)
    biases = np.zeros(NUM_LABELS, dtype=np.float32)
    for cat, idx in label_map.items():
        biases[idx] = float(biases_dict.get(cat, 0.0))

    test_df = pd.read_csv(TEST_CSV)[["text", "label", "category"]]
    sample = test_df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sample: {len(sample)} rows")

    texts = sample["text"].astype(str).tolist()
    print("Predicting on original texts...")
    base_preds = predict_argmax(model_dirs, texts, device, max_length, biases)

    results: dict[str, dict] = {}
    for name, pfn in PERTURBATIONS.items():
        print(f"\nPerturbation: {name}")
        perturbed = [pfn(t) for t in texts]
        new_preds = predict_argmax(model_dirs, perturbed, device, max_length, biases)
        stable = (base_preds == new_preds).mean()
        results[name] = {
            "stability_rate": float(stable),
            "n_changed": int((base_preds != new_preds).sum()),
        }
        print(f"  Stability: {stable:.4f}  ({results[name]['n_changed']} changed)")

    # Save report
    lines = ["# Perturbation robustness", ""]
    lines.append(f"Sample: {N_SAMPLES} test rows. Stability = fraction of predictions unchanged after perturbation.")
    lines.append("")
    lines.append("| Perturbation | Stability | Changed |")
    lines.append("|---|---:|---:|")
    for name, r in results.items():
        lines.append(f"| {name} | {r['stability_rate']:.4f} | {r['n_changed']}/{N_SAMPLES} |")
    lines.append("")
    avg_stab = np.mean([r["stability_rate"] for r in results.values()])
    lines.append(f"**Mean stability across all 6 perturbations: {avg_stab:.4f}**")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- > 0.90 = robust (predictions barely change)")
    lines.append("- 0.80-0.90 = acceptable")
    lines.append("- < 0.80 = brittle, consider augmenting training with similar perturbations")
    lines.append("")
    weakest = min(results.items(), key=lambda x: x[1]["stability_rate"])
    lines.append(f"Weakest perturbation: **{weakest[0]}** ({weakest[1]['stability_rate']:.4f})")

    txt = "\n".join(lines)
    print()
    print(txt)
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    print(f"\nSaved -> {OUT_TXT}")


if __name__ == "__main__":
    main()
