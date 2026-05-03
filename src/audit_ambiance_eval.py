"""
Audit the val + test ambiance samples for label noise.

For each sample, count category-keyword hits and classify into:
  - "clean_ambiance": only/mostly ambiance keywords
  - "multi_aspect": ambiance + ≥1 other strong category
  - "probably_mislabeled": ambiance keywords absent/weak, other category dominates
  - "ambiguous": needs human eyeball

Outputs CSV at data/processed/ambiance_audit.csv with:
  - source_split (val/test)
  - text
  - ambiance_hits, food_hits, price_hits, etc.
  - dominant_category
  - my_classification
  - human_decision (empty — fill in: keep/relabel:X/multi/drop)
  - notes (empty — for human notes)

Open in Excel, fill `human_decision`, save, then run integrate_audited_ambiance.py.

Estimated review time: ~30-40 min for ~170 samples.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = ROOT / "data" / "processed" / "val.csv"
TEST_CSV = ROOT / "data" / "processed" / "test.csv"
OUT_CSV = ROOT / "data" / "processed" / "ambiance_audit.csv"
TARGET_CATEGORY = "الجو والمكان"

# Keyword sets per category (extended from build_dataset.py + generate_synthetic.py)
KW = {
    "الجو والمكان": {
        "ديكور", "تصميم", "اضاءه", "كراسي", "طاولات", "جلسه", "جلسات",
        "موقف", "تكييف", "ضيق", "مساحه", "ريحه", "زحمه", "مزدحم",
        "موسيقى", "حر", "بارد", "كنبات", "كنب", "حمامات", "حمام",
        "جلسه", "اجواء", "مكتظ", "ضوضاء", "ضجيج", "خانق", "مظلم",
        "تهويه", "هواء", "interior", "decor", "AC",
    },
    "جودة الطعام": {
        "طعم", "نكهه", "ذوق", "بايخ", "لذيذ", "محروق", "ني",
        "اكل بارد", "اكل ساخن", "ملح", "حلو", "حامض", "متبل",
        "دسم", "مالح", "بيتزا", "برجر", "شاورما", "دجاج", "لحم", "سمك",
        "وجبه", "صحن", "طبق", "البطاطس", "الرز", "كباب",
    },
    "السعر والقيمة": {
        "سعر", "غالي", "اسعار", "قيمه", "ثمن", "مبالغ", "مكلف", "رخيص",
        "ريال", "فلوس", "تكلفه", "حسابي", "فاتوره",
    },
    "النظافة": {
        "نظافه", "نظيف", "متسخ", "وسخ", "حشرات", "وصخ", "خايس",
        "اوساخ", "غير نظيف", "بقع",
    },
    "خدمة الموظفين": {
        "موظف", "جرسون", "نادل", "كاشير", "اسلوب", "ادب", "اخلاق",
        "تعامل", "خدمه", "العمال", "الجرسونات", "الويتر",
    },
    "وقت الانتظار": {
        "انتظار", "انتظرت", "ساعه", "نص ساعه", "نصف ساعه", "بطيء",
        "تاخر الطلب", "وقت طويل", "مده طويله",
    },
    "التوصيل": {
        "توصيل", "مندوب", "سائق", "ديليفري", "delivery", "وصل بارد",
        "ما وصل", "تتبع", "الديليفري",
    },
    "دقة الطلب": {
        "ناقص", "غلط", "بدلوا", "نسوا", "ما طلبت", "خلطوا", "غلطوا",
        "وصل غير", "بدون بصل", "اضافات غير",
    },
    "عامة": set(),  # no specific keywords
}


def count_hits(text, keyword_set):
    return sum(1 for kw in keyword_set if kw in text)


def classify(text, hits):
    """Returns ('classification', 'notes') tuple."""
    amb = hits.get("الجو والمكان", 0)
    others = {k: v for k, v in hits.items() if k != "الجو والمكان" and v > 0}

    if amb == 0 and not others:
        return "ambiguous", "no clear keywords"
    if amb == 0:
        dom = max(others.items(), key=lambda x: x[1])
        return "probably_mislabeled", f"no ambiance keywords; dominant: {dom[0]} ({dom[1]} hits)"
    if amb >= 2 and not others:
        return "clean_ambiance", f"{amb} ambiance keywords, no competing category"
    if amb >= 2 and others:
        max_other = max(others.values())
        if max_other >= amb:
            dom = max(others.items(), key=lambda x: x[1])
            return "multi_aspect", f"ambiance={amb}, competing {dom[0]}={dom[1]}"
        return "multi_aspect", f"ambiance dominant ({amb}) but also: {others}"
    if amb == 1 and not others:
        return "clean_ambiance", "1 ambiance keyword, no others"
    if amb == 1 and others:
        max_other = max(others.values())
        if max_other > amb:
            dom = max(others.items(), key=lambda x: x[1])
            return "probably_mislabeled", f"weak ambiance (1), stronger: {dom[0]} ({dom[1]})"
        return "ambiguous", f"weak ambiance (1), comparable: {others}"
    return "ambiguous", f"hits: {hits}"


def main():
    rows = []
    for split, path in [("val", VAL_CSV), ("test", TEST_CSV)]:
        df = pd.read_csv(path)
        amb = df[df.category == TARGET_CATEGORY].copy()
        print(f"{split}: {len(amb)} ambiance samples")
        for _, r in amb.iterrows():
            text = str(r["text"])
            hits = {cat: count_hits(text, kws) for cat, kws in KW.items()}
            classification, notes = classify(text, hits)
            dom = max(((c, h) for c, h in hits.items() if h > 0), key=lambda x: x[1], default=("none", 0))
            rows.append({
                "source_split": split,
                "text": text,
                **{f"hits_{c}": h for c, h in hits.items()},
                "dominant_category": dom[0],
                "dominant_hits": dom[1],
                "my_classification": classification,
                "notes_auto": notes,
                "human_decision": "",   # for user to fill: keep / relabel:CATEGORY / multi / drop
                "human_notes": "",
            })

    df_out = pd.DataFrame(rows)
    # Sort: clean_ambiance first (likely keeps), then multi_aspect, then probably_mislabeled
    sort_order = {"clean_ambiance": 0, "ambiguous": 1, "multi_aspect": 2, "probably_mislabeled": 3}
    df_out["_sort"] = df_out["my_classification"].map(sort_order)
    df_out = df_out.sort_values(["_sort", "source_split"]).drop(columns=["_sort"])

    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")  # BOM for Excel UTF-8
    print(f"\nSaved {len(df_out)} rows -> {OUT_CSV}")
    print()
    print("Auto-classification summary:")
    for k, v in df_out["my_classification"].value_counts().items():
        print(f"  {k}: {v}")
    print()
    print("Next steps:")
    print(f"  1. Open {OUT_CSV} in Excel")
    print("  2. For each row, fill `human_decision` column with one of:")
    print("       - keep                    (it's truly ambiance)")
    print("       - relabel:<category>      (e.g. relabel:جودة الطعام)")
    print("       - multi                   (multi-aspect, drop from single-label eval)")
    print("       - drop                    (mislabeled or unusable)")
    print("  3. Save the CSV (keep BOM/UTF-8 encoding)")
    print("  4. Run: py src/integrate_audited_ambiance.py")


if __name__ == "__main__":
    main()
