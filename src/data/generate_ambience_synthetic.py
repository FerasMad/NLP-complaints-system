"""Generate contrastive-pair synthetic ambience training data.

Produces a CSV of hand-crafted Saudi/Gulf complaint pairs where each pair
shares a risky word (بارد, حار, قديم, ريحة, زيت, etc.) but one row is
ambience and the other is a non-ambience hard-negative. The model learns
the boundary from the contrast.

Why contrastive pairs and not free-form generation: the v3 ambience class
died from over-broad synthetic templates that taught the model to fire on
any "place + adjective" pattern. Contrastive pairs explicitly anchor the
boundary by giving the model both sides of every confusable case.

Output schema (matches docs/DATA_SCHEMA_AMBIENCE.md):

  text                 — the synthetic complaint (Arabic, 5-18 words)
  category             — الجو والمكان or one of the 8 production classes
  ambience_subtype     — one of the 12 subtypes (only on ambience rows)
  source               — always 'ambience_synthetic' (train-only via leakage gate)
  is_hard_negative     — true on the contrastive non-ambience rows
  risk_keyword         — the boundary-breaking word the pair shares
  contrast_group_id    — g001, g002, ... — links each positive to its negative
  quality_score        — 1-5; default 5 for hand-curated templates
  notes                — short rationale for the pair

CLI:

    python src/data/generate_ambience_synthetic.py \\
        --output data/text/ambience_synthetic_v1.csv

The output is train-only by virtue of source='ambience_synthetic' — the
existing leakage gate in tests/test_data_pipeline.py blocks this source
from val/test splits.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Contrastive pair definitions — each entry produces TWO rows.
#
# Format: (ambience_text, ambience_subtype, hard_neg_text, hard_neg_category, risk_keyword, notes)
# ---------------------------------------------------------------------------

PAIRS: list[tuple[str, str, str, str, str, str]] = [
    # --- Temperature: place vs food ----------------------------------------
    (
        "المكان بارد والمكيف قوي ما نقدر نقعد",
        "temperature_ac",
        "الاكل بارد لما وصل ما عجبني الطعم",
        "جودة الطعام",
        "بارد",
        "place temp vs food temp",
    ),
    (
        "المطعم حار والمكيف ما يبرد جلسنا نتعرق",
        "temperature_ac",
        "الشاورما حارة بزياده ما اقدر اكلها",
        "جودة الطعام",
        "حار",
        "place temp vs food spice",
    ),
    (
        "التكييف ضعيف والجو خانق",
        "temperature_ac",
        "القهوة باردة ما لها طعم ابدا",
        "جودة الطعام",
        "بارد",
        "AC vs cold drink",
    ),
    (
        "المكيف بارد بزياده يثلج",
        "temperature_ac",
        "الايس كريم ذايب وحامض الطعم",
        "جودة الطعام",
        "بارد",
        "AC over-cold vs melted dessert",
    ),
    (
        "الفرع حار جدا الجو ما يطاق",
        "temperature_ac",
        "البرجر مقلي زياده ومحروق من برا",
        "جودة الطعام",
        "حار",
        "place hot vs food over-fried",
    ),
    # --- Smell: environment vs food ---------------------------------------
    (
        "ريحة الزيت ماسكة في المكان طلعت ريحتي",
        "smell",
        "البطاطس مليانه زيت وحامضه",
        "جودة الطعام",
        "زيت",
        "env oil smell vs greasy food",
    ),
    (
        "ريحة الدخان تخنق ما فيه تهوية",
        "smell",
        "الاكل ريحته غريبة شككتني فيه",
        "جودة الطعام",
        "ريحة",
        "smoke in air vs food smell",
    ),
    (
        "ريحة المجاري طالعة من الحمام",
        "smell",
        "ريحة الزيت في الصلصة بايخه",
        "جودة الطعام",
        "ريحة",
        "sewage smell vs food smell",
    ),
    (
        "الريحة تلصق في الملابس بعد ما اطلع",
        "smell",
        "ريحة اللحمه طلعت غريبة ورديتها",
        "جودة الطعام",
        "ريحة",
        "lingering env smell vs food smell",
    ),
    # --- Furniture/seating: condition vs cleanliness ----------------------
    (
        "الطاولة تهتز كل ما تحط شي عليها",
        "decor_furniture",
        "الطاولة وسخه فيها بقع طعام من قبل",
        "النظافة",
        "الطاولة",
        "wobbly table vs dirty table",
    ),
    (
        "الكنب مهلوك والاسفنج طالع",
        "decor_furniture",
        "الكنب وسخ عليه بقع وما تنظف",
        "النظافة",
        "الكنب",
        "worn sofa vs dirty sofa",
    ),
    (
        "الكراسي قاسية على الظهر ما اقدر اقعد",
        "seating_comfort",
        "الكراسي عليها بقايا طعام من اللي قبلنا",
        "النظافة",
        "الكراسي",
        "uncomfortable chairs vs dirty chairs",
    ),
    (
        "الطاولات صغيرة ما تكفي الطلب",
        "seating_comfort",
        "الطاولة وصخه ومالها بصمة تنظيف",
        "النظافة",
        "الطاولة",
        "small tables vs uncleaned tables",
    ),
    # --- Bathroom: facility vs cleanliness --------------------------------
    (
        "السيفون خربان وما فيه صابون اصلا",
        "bathroom_facilities",
        "الحمام وصخ ومقرف ما يستحمل",
        "النظافة",
        "الحمام",
        "broken bathroom vs dirty bathroom",
    ),
    (
        "باب الحمام ما يقفل والمغسلة معطلة",
        "bathroom_facilities",
        "الحمام مليان مجاري وريحه تطلع",
        "النظافة",
        "الحمام",
        "broken bathroom hardware vs filthy bathroom",
    ),
    (
        "ما فيه مناديل في الحمام ولا صابون",
        "bathroom_facilities",
        "الارضيه في الحمام مبلولة ووسخه",
        "النظافة",
        "الحمام",
        "missing supplies vs floor hygiene",
    ),
    # --- Crowding/space ---------------------------------------------------
    (
        "المكان زحمه موت ما لقينا مكان نقعد",
        "space_crowding",
        "الموظفين ما يعرفون يخدمون من الزحمه",
        "خدمة الموظفين",
        "زحمه",
        "crowding vs slow-staff-blamed-on-crowd",
    ),
    (
        "الجلسات متلاصقه فينا ما فيه مساحه",
        "space_crowding",
        "الفاتوره غاليه على الجلسه القصيرة",
        "السعر والقيمة",
        "الجلسات",
        "tight seating vs price",
    ),
    (
        "ضيق وما تقدر تتنفس من الزحمة",
        "space_crowding",
        "السعر مبالغ فيه على وجبه ضيقه",
        "السعر والقيمة",
        "ضيق",
        "tight space vs portion-size complaint",
    ),
    # --- Noise/music ------------------------------------------------------
    (
        "الموسيقى صوتها عالي ما نقدر نسولف",
        "noise_music",
        "الموظف صوته عالي ومزعج وقت الطلب",
        "خدمة الموظفين",
        "عالي",
        "loud music vs loud staff",
    ),
    (
        "الاطفال يصرخون في كل المكان جو ازعاج",
        "noise_music",
        "الموظف ازعجني بالاسئله الكثيرة",
        "خدمة الموظفين",
        "ازعاج",
        "child noise vs annoying staff",
    ),
    (
        "الدي جي يفتح الصوت لين تطق ودانك",
        "noise_music",
        "النادل صوته عالي وما يحترم",
        "خدمة الموظفين",
        "عالي",
        "DJ noise vs rude loud staff",
    ),
    # --- Lighting ---------------------------------------------------------
    (
        "الاضاءه قويه على العين بزياده",
        "lighting",
        "السعر قوي على وجبة بسيطه",
        "السعر والقيمة",
        "قوي",
        "bright lights vs steep price",
    ),
    (
        "المكان مظلم وما اقدر اشوف المنيو",
        "lighting",
        "المنيو قديم وفيه اطباق مش موجوده",
        "خدمة الموظفين",
        "قديم",
        "dim lights vs outdated menu",
    ),
    (
        "النور خافت بزياده تحس انك في كهف",
        "lighting",
        "الطلب وصل خافت الطعم وبدون نكهه",
        "جودة الطعام",
        "خافت",
        "weak light vs weak-flavored food",
    ),
    # --- Decor / general place vibe ---------------------------------------
    (
        "الديكور قديم والفرش مهلوك يبي تجديد",
        "decor_furniture",
        "الاكل قديم محسوس عليه ومش طازه",
        "جودة الطعام",
        "قديم",
        "old decor vs old food",
    ),
    (
        "المكان كئيب وما يفتح النفس",
        "general_place_vibe",
        "الموظف كان بارد في التعامل",
        "خدمة الموظفين",
        "كئيب",
        "depressing place vs cold staff",
    ),
    (
        "الجو ثقيل في المطعم ما يستحمل",
        "general_place_vibe",
        "الفاتوره ثقيله على الجيب",
        "السعر والقيمة",
        "ثقيل",
        "heavy vibe vs heavy bill",
    ),
    (
        "الشكل قديم وكأنه من زمان والحمد لله",
        "general_place_vibe",
        "الاكل قديم وما يستاهل ابدا",
        "جودة الطعام",
        "قديم",
        "old-style place vs old food",
    ),
    # --- Parking ----------------------------------------------------------
    (
        "ما فيه مواقف اضطرينا نمشي من بعيد",
        "parking",
        "الطلب تأخر والمندوب تاه",
        "التوصيل",
        "بعيد",
        "no parking vs delivery-far",
    ),
    (
        "المواقف بعيدة جدا والمدخل صعب",
        "parking",
        "المندوب وصل بعيد عن البيت ومانتبه",
        "التوصيل",
        "بعيد",
        "distant parking vs distant delivery",
    ),
    # --- Outdoor / view ---------------------------------------------------
    (
        "الجلسة الخارجية مكشوفة على الشارع وجو مزعج",
        "outdoor_view",
        "الطلب وصل برا الكيس واتلطخ",
        "التوصيل",
        "برا",
        "outdoor seating vs delivery-outside-bag",
    ),
    (
        "التراس ما فيه ظل والشمس قوية",
        "outdoor_view",
        "الشمس على الباب طلعت الايس كريم خربان",
        "جودة الطعام",
        "الشمس",
        "no outdoor shade vs sun-spoiled food",
    ),
    # --- Privacy ----------------------------------------------------------
    (
        "ما فيه خصوصية كل الناس تشوفك تاكل",
        "privacy",
        "الكاشير ما يحترم خصوصية اللي قبلك",
        "خدمة الموظفين",
        "خصوصية",
        "no privacy in seating vs no customer privacy",
    ),
    (
        "قسم العوائل ضيق ولاصقين فينا",
        "privacy",
        "الموظف لاصق فينا ما تحس بالراحه",
        "خدمة الموظفين",
        "لاصقين",
        "crowded family section vs hovering staff",
    ),
]


# ---------------------------------------------------------------------------
# Schema + writer
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "text",
    "category",
    "ambience_subtype",
    "source",
    "is_hard_negative",
    "risk_keyword",
    "contrast_group_id",
    "quality_score",
    "notes",
]


def expand_pairs(pairs: list[tuple]) -> list[dict]:
    """Each pair becomes two rows: ambience positive + non-ambience negative."""
    rows: list[dict] = []
    for i, (amb_text, subtype, neg_text, neg_cat, risk, note) in enumerate(pairs, start=1):
        gid = f"g{i:03d}"
        rows.append({
            "text": amb_text,
            "category": "الجو والمكان",
            "ambience_subtype": subtype,
            "source": "ambience_synthetic",
            "is_hard_negative": False,
            "risk_keyword": risk,
            "contrast_group_id": gid,
            "quality_score": 5,
            "notes": f"ambience positive — {note}",
        })
        rows.append({
            "text": neg_text,
            "category": neg_cat,
            "ambience_subtype": "",
            "source": "ambience_synthetic",
            "is_hard_negative": True,
            "risk_keyword": risk,
            "contrast_group_id": gid,
            "quality_score": 5,
            "notes": f"contrastive hard-negative — {note}",
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Emit contrastive ambience synthetic CSV.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/text/ambience_synthetic_v1.csv"),
        help="Output CSV (default: data/text/ambience_synthetic_v1.csv)",
    )
    p.add_argument(
        "--min-quality",
        type=int,
        default=4,
        help="Drop rows with quality_score below this threshold (default: 4)",
    )
    args = p.parse_args()

    rows = expand_pairs(PAIRS)
    rows = [r for r in rows if r["quality_score"] >= args.min_quality]

    n_amb = sum(1 for r in rows if r["category"] == "الجو والمكان")
    n_neg = sum(1 for r in rows if r["is_hard_negative"])
    n_groups = len({r["contrast_group_id"] for r in rows})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"  total rows:           {len(rows)}")
    print(f"  ambience positives:   {n_amb}")
    print(f"  contrastive negatives: {n_neg}")
    print(f"  contrast groups:      {n_groups}")
    print()
    print(f"All rows have source='ambience_synthetic' — train-only via the existing leakage gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
