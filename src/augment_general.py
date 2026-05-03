"""Back-translation augmentation for the عامة (general) category.

Reads existing real-source عامة rows from data/text/complaints_labeled.csv,
runs each through Ar→En→Ar via Helsinki-NLP MarianMT, appends paraphrased
variants with source="augmented_bt".

These are train-only (split_dataset.py excludes augmented_bt from val/test).

Note: tried in production. Single-model val عامة F1 improved (+2.2%) but the
ensemble test عامة F1 regressed (-1.2%). BT paraphrases drift from the
real-data distribution; ensemble amplifies the gap. Script kept for future
experimentation (e.g. smaller doses, different MT pair).
"""
import csv
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LABELED = Path("data/text/complaints_labeled.csv")
TARGET_CATEGORY = "عامة"
GOOD_SOURCES = {"production", "play_store", "res1"}  # real sources only
BATCH_SIZE = 16


def load_originals():
    rows, seen = [], set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("category") != TARGET_CATEGORY:
                continue
            if r.get("source") not in GOOD_SOURCES:
                continue
            t = r.get("text", "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "category": TARGET_CATEGORY, "priority": "متوسطة", "source": "augmented_bt"})
    return rows, seen


def main():
    print(f"Loading real {TARGET_CATEGORY} from {LABELED}...")
    originals, existing = load_originals()
    print(f"  {len(originals)} real originals to paraphrase")
    if not originals:
        return

    import torch
    from transformers import MarianMTModel, MarianTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading Helsinki-NLP/opus-mt-ar-en ...")
    ar_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
    ar_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en").to(device).eval()

    print("Loading Helsinki-NLP/opus-mt-en-ar ...")
    en_ar_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
    en_ar = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(device).eval()

    @torch.no_grad()
    def translate(model, tok, texts):
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            gen = model.generate(**enc, max_length=160, num_beams=2)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"    {i + len(batch)}/{len(texts)}")
        return out

    print("\nAr -> En...")
    en_texts = translate(ar_en, ar_en_tok, [r["text"] for r in originals])
    print("\nEn -> Ar...")
    paraphrased = translate(en_ar, en_ar_tok, en_texts)

    out_rows = []
    for orig, para in zip(originals, paraphrased):
        para = para.strip()
        if not para or len(para) < 5 or para == orig["text"] or para in existing:
            continue
        existing.add(para)
        out_rows.append({"text": para, "category": TARGET_CATEGORY, "priority": "متوسطة", "source": "augmented_bt"})

    print(f"\nProduced {len(out_rows)} unique paraphrased {TARGET_CATEGORY} variants")
    with open(LABELED, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(out_rows)
    print(f"Appended to {LABELED}")


if __name__ == "__main__":
    main()
