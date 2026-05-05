"""
Rebuild data/text/complaints_labeled.csv using only files that exist locally.

The original src/build_dataset.py references data files at C:/Users/Feras/...
that aren't on this machine. This rebuilder:
  1. Loads existing complaints_labeled.csv (it's already merged, source of truth)
  2. Strips out old synthetic rows (will be replaced by fresh generation)
  3. Loads new scraped Play Store reviews + filters with extended keyword lists
  4. Deduplicates against existing texts
  5. Writes new complaints_labeled.csv (real-only at this point)

Run order for a full rebuild:
  py src/rebuild_dataset.py        # this script — scraped real data only
  py src/generate_synthetic.py     # appends synthetic
  py src/augment_ambience.py       # appends back-translated ambience
  py src/split_dataset.py          # produces train/val/test
"""
import csv
import os
import re
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

LABELED = "data/text/complaints_labeled.csv"
PLAY = "data/raw/play_store_reviews.csv"

# Extended keyword lists — broader recall on scraped data than original build_dataset.py
DELIVERY_KW = [
    "توصيل", "تأخر", "تاخر", "لم يصل", "ما وصل", "ماوصل", "مندوب", "سائق",
    "الطلب ضاع", "بارد وصل", "وصل بارد", "عنوان خاطئ", "عنوان غلط",
    "رسوم التوصيل", "وقت التوصيل", "موعد التوصيل", "لم أستلم", "ما استلمت",
    "التوصيل بطيء", "ديليفري", "delivery", "التتبع", "تتبع الطلب",
    "الطلب لم يصل", "الشحن", "ما وصلت",
    # extended:
    "ضاع الطلب", "ضايع", "وين الطلب", "وقت طويل", "نص ساعه", "ساعه", "ساعتين",
    "خدمه التوصيل", "السائق", "الديليفري", "رسوم", "متاخر", "تاخير",
    "وصل متأخر", "تجاهل التوصيل", "غلط في العنوان", "اخطأ في العنوان",
]

ORDER_KW = [
    "ناقص", "ناقصة", "ناقصه", "غلط", "خاطئ", "بدلوا", "بدّلوا", "غيروا",
    "وصلني غير", "طلبي مختلف", "مو اللي طلبت", "نسوا", "ناسين",
    "أضافوا", "حذفوا", "طلبت بدون", "وجبة ناقصة", "صنف ناقص",
    "الكمية غلط", "غير صحيح", "الطلب مختلف", "أخطأوا", "إضافات غير",
    # extended:
    "وجبه ناقصه", "صنف ناقص", "الكميه غلط", "ناقصه نصف", "نقص في الطلب",
    "اخطاء في الطلب", "اخطاؤا", "خلطوا", "خلط", "الطلب فيه نقص",
    "ما اللي طلبت", "مو طلبي", "ما طلبت", "غلطوا", "وصل غلط",
    "بدل بدل", "بدون", "نسي", "نسيان", "اضافوا غلط",
]

# Lana's cleaning (vendored from build_dataset.py)
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
NON_ARABIC = re.compile(r"[^؀-ۿa-zA-Z0-9٠-٩\s]")
WHITESPACE = re.compile(r"\s+")
ARABIC_CHAR = re.compile(r"[؀-ۿ]")


def clean(text):
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}))
    t = NON_ARABIC.sub(" ", t)
    t = WHITESPACE.sub(" ", t).strip().lower()
    return t


def is_arabic(text):
    if not text or len(text) < 5:
        return False
    arabic_chars = len(ARABIC_CHAR.findall(text))
    return arabic_chars / max(len(text), 1) > 0.3


def matches_any(text, keywords):
    return any(kw in text for kw in keywords)


def load_existing_real():
    """Load all rows from labeled file EXCEPT synthetic ones."""
    rows = []
    seen = set()
    with open(LABELED, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
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
                "source": r.get("source", "unknown"),
            })
    return rows, seen


def load_new_scraped(seen_texts):
    """Filter new Play Store reviews into delivery + order_accuracy."""
    delivery, order = [], []
    if not os.path.exists(PLAY):
        print(f"  ! No {PLAY}, skipping")
        return delivery, order
    with open(PLAY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            text = r.get("text", "").strip()
            if not is_arabic(text):
                continue
            cleaned = clean(text)
            if not cleaned or len(cleaned) < 5:
                continue
            if cleaned in seen_texts:
                continue
            # delivery first (more specific intent), then order accuracy
            if matches_any(text, DELIVERY_KW):
                delivery.append({"text": cleaned, "category": "التوصيل", "priority": "متوسطة", "source": "play_store"})
                seen_texts.add(cleaned)
            elif matches_any(text, ORDER_KW):
                order.append({"text": cleaned, "category": "دقة الطلب", "priority": "متوسطة", "source": "play_store"})
                seen_texts.add(cleaned)
    return delivery, order


def main():
    print("=== Rebuilding complaints_labeled.csv ===\n")

    print("[1] Loading existing labeled rows (excluding synthetic + augmented)...")
    existing, seen = load_existing_real()
    print(f"    {len(existing)} real rows kept")

    print("\n[2] Filtering new scraped Play Store reviews...")
    delivery, order = load_new_scraped(seen)
    print(f"    new التوصيل: +{len(delivery)}")
    print(f"    new دقة الطلب: +{len(order)}")

    final = existing + delivery + order

    print(f"\n[3] Final row count: {len(final)} real rows")
    print("    distribution:")
    for cat, n in Counter(r["category"] for r in final).most_common():
        print(f"      {cat}: {n}")

    print(f"\n[4] Writing {LABELED}...")
    with open(LABELED, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writeheader()
        writer.writerows(final)
    print("    done.")
    print()
    print("Next steps:")
    print("  py src/generate_synthetic.py     # appends new synthetic")
    print("  py src/augment_ambience.py       # appends back-translated ambience")
    print("  py src/split_dataset.py          # produces train/val/test")


if __name__ == "__main__":
    main()
