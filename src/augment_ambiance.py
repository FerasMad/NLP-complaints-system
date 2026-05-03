"""
Back-translation augmentation for الجو والمكان (ambiance).

Ambiance is the bottleneck (567 real samples, 42% F1) and can't be scraped from
delivery apps. This script:
  1. Pulls the 567 real ambiance complaints from data/text/complaints_labeled.csv
  2. Translates Arabic -> English -> Arabic via Helsinki-NLP MarianMT
  3. Appends paraphrased variants with source="augmented_bt"
     (so split_dataset.py keeps them train-only)

Runs once. ~10-15 min on RTX 4070 the first time (downloads ~600MB of NMT
weights). Re-runs are faster.

If GPU isn't available, falls back to CPU (~30 min). If models can't be
downloaded, prints a clear error and exits without modifying the dataset.
"""
import csv
import os
import sys

# Force UTF-8 for Arabic on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LABELED = "data/text/complaints_labeled.csv"
TARGET_CATEGORY = "الجو والمكان"
BATCH_SIZE = 16


def load_originals():
    rows = []
    seen = set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("category") != TARGET_CATEGORY:
                continue
            if r.get("source") == "synthetic":
                continue
            if r.get("source") == "augmented_bt":
                continue
            t = r.get("text", "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            rows.append({
                "text": t,
                "category": r["category"],
                "priority": r.get("priority", "متوسطة"),
                "source": "augmented_bt",
            })
    return rows, seen


def main():
    print(f"Loading existing real ambiance complaints from {LABELED}...")
    originals, existing_texts = load_originals()
    print(f"  {len(originals)} originals to paraphrase")

    if not originals:
        print("No originals to augment. Exiting.")
        return

    # Late imports — only fail at run time if user lacks the deps
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except Exception as e:
        print(f"ERROR: missing transformers/torch: {e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading Helsinki-NLP/opus-mt-ar-en (Arabic -> English)...")
    try:
        ar_en_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ar-en")
        ar_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ar-en").to(device).eval()
    except Exception as e:
        print(f"ERROR: cannot load Arabic->English model: {e}")
        print("Skipping augmentation. Re-run with internet to download.")
        sys.exit(1)

    print("Loading Helsinki-NLP/opus-mt-en-ar (English -> Arabic)...")
    try:
        en_ar_tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
        en_ar = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar").to(device).eval()
    except Exception as e:
        print(f"ERROR: cannot load English->Arabic model: {e}")
        sys.exit(1)

    @torch.no_grad()
    def translate(model, tok, texts):
        out = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
            gen = model.generate(**enc, max_length=160, num_beams=2)
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            out.extend(decoded)
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"    {i + len(batch)}/{len(texts)}")
        return out

    print("\nArabic -> English...")
    en_texts = translate(ar_en, ar_en_tok, [r["text"] for r in originals])

    print("\nEnglish -> Arabic...")
    paraphrased = translate(en_ar, en_ar_tok, en_texts)

    # Filter trivial / low-quality
    out_rows = []
    for orig, para in zip(originals, paraphrased):
        para = para.strip()
        if not para or len(para) < 5:
            continue
        if para == orig["text"]:
            continue  # no change → skip
        if para in existing_texts:
            continue  # collision with another original or earlier output
        existing_texts.add(para)
        out_rows.append({
            "text": para,
            "category": TARGET_CATEGORY,
            "priority": "متوسطة",
            "source": "augmented_bt",
        })

    print(f"\nProduced {len(out_rows)} unique paraphrased ambiance variants")

    # Append to labeled file
    with open(LABELED, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(out_rows)
    print(f"Appended to {LABELED}")


if __name__ == "__main__":
    main()
