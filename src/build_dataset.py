"""
Build the labeled complaints dataset for all 9 categories.

Sources:
- production_dataset.csv (88K rows, 6 existing categories — renamed where needed)
- play_store_reviews.csv (scraped 1-2 star reviews → التوصيل + دقة الطلب)
- RES1.csv negative reviews → الجو والمكان

Output: data/text/complaints_labeled.csv with columns text, category, priority
"""
import csv
import re
import os
from collections import Counter

# paths
PROD = "C:/Users/Feras/Complaint-system/data/production_dataset.csv"
PLAY = "data/raw/play_store_reviews.csv"
RES1 = "C:/Users/Feras/Complaint-system/data/large-arabic-sentiment/datasets/RES1.csv"
OUT = "data/text/complaints_labeled.csv"

# rename map for legacy categories
RENAME = {
    "الخدمة": "خدمة الموظفين",
    "التأخير": "وقت الانتظار",
    "السعر": "السعر والقيمة",
}

# keyword lists for new categories
DELIVERY_KW = [
    "توصيل", "تأخر", "لم يصل", "ما وصل", "ماوصل", "مندوب", "سائق",
    "الطلب ضاع", "بارد وصل", "وصل بارد", "عنوان خاطئ", "عنوان غلط",
    "رسوم التوصيل", "وقت التوصيل", "موعد التوصيل", "لم أستلم", "ما استلمت",
    "التوصيل بطيء", "ديليفري", "delivery", "التتبع", "تتبع الطلب",
    "الطلب لم يصل", "الشحن", "ما وصلت",
]

ORDER_KW = [
    "ناقص", "ناقصة", "غلط", "خاطئ", "بدلوا", "بدّلوا", "غيروا",
    "وصلني غير", "طلبي مختلف", "مو اللي طلبت", "نسوا", "ناسين",
    "أضافوا", "حذفوا", "طلبت بدون", "وجبة ناقصة", "صنف ناقص",
    "الكمية غلط", "غير صحيح", "الطلب مختلف", "أخطأوا", "إضافات غير",
]

AMBIANCE_KW = [
    "ديكور", "تصميم", "الجو", "أجواء", "إضاءة", "ضوضاء", "ضجيج",
    "صخب", "كراسي", "طاولات", "مقاعد", "الجلسة", "موقف", "باركنج",
    "حار", "سخونة", "التكييف", "مكيفات", "ضيق", "مساحة", "رائحة",
    "ريحة", "ازدحام", "زحمة", "مزدحم", "خارجي", "تراس", "شرفة",
    "موسيقى صاخبة", "واجهة المطعم",
]


# ---------- Lana's cleaning ----------
TASHKEEL = re.compile(r"[ً-ٰٟؐ-ؚ]")
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


# ---------- step 1: production data ----------
def load_production():
    rows = []
    with open(PROD, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            text = r.get("text", "").strip()
            cat = r.get("category", "").strip()
            prio = r.get("priority", "متوسطة").strip()
            if not text or not cat:
                continue
            cat = RENAME.get(cat, cat)
            rows.append({"text": text, "category": cat, "priority": prio, "source": "production"})
    return rows


# ---------- step 2: play store → delivery + order accuracy ----------
def load_play_store():
    delivery, order = [], []
    with open(PLAY, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            text = r.get("text", "").strip()
            if not is_arabic(text):
                continue
            # delivery first (more specific intent), then order accuracy
            if matches_any(text, DELIVERY_KW):
                delivery.append({"text": text, "category": "التوصيل", "priority": "متوسطة", "source": "play_store"})
            elif matches_any(text, ORDER_KW):
                order.append({"text": text, "category": "دقة الطلب", "priority": "متوسطة", "source": "play_store"})
    return delivery, order


# ---------- step 3: RES1 negative → ambiance ----------
def load_res1_ambiance():
    rows = []
    with open(RES1, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if str(r.get("polarity", "")).strip() != "-1":
                continue
            text = r.get("text", "").strip()
            if not is_arabic(text):
                continue
            if matches_any(text, AMBIANCE_KW):
                rows.append({"text": text, "category": "الجو والمكان", "priority": "متوسطة", "source": "res1"})
    return rows


def main():
    print("Loading production_dataset.csv...")
    prod = load_production()
    print(f"  {len(prod)} rows")

    print("\nLoading Play Store scraped reviews...")
    delivery, order = load_play_store()
    print(f"  التوصيل: {len(delivery)}")
    print(f"  دقة الطلب: {len(order)}")

    print("\nLoading RES1 ambiance complaints...")
    ambiance = load_res1_ambiance()
    print(f"  الجو والمكان: {len(ambiance)}")

    # clean each set separately, then merge with new categories overriding production
    def clean_set(rows):
        out = []
        for r in rows:
            t = clean(r["text"])
            if t and len(t) >= 3:
                out.append({"text": t, "category": r["category"], "priority": r["priority"], "source": r["source"]})
        return out

    prod_c = clean_set(prod)
    new_c = clean_set(delivery + order + ambiance)
    print(f"\nAfter cleaning — production: {len(prod_c)}, new: {len(new_c)}")

    # build text → row map: new category rows take priority
    by_text = {}
    for r in prod_c:
        by_text[r["text"]] = r
    overrides = 0
    for r in new_c:
        if r["text"] in by_text and by_text[r["text"]]["category"] != r["category"]:
            overrides += 1
        by_text[r["text"]] = r

    final = list(by_text.values())
    print(f"Overrides (production row relabeled to new category): {overrides}")
    print(f"After cleaning + merge: {len(final)} rows")

    # report distribution
    print("\nFinal category distribution:")
    for cat, n in Counter(r["category"] for r in final).most_common():
        print(f"  {cat}: {n}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writeheader()
        writer.writerows(final)

    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
