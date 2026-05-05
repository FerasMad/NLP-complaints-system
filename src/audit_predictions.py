"""Behavioral audit of the production ensemble.

Runs a curated set of realistic Arabic complaints (single-aspect and
multi-aspect, including the three failure cases the user flagged) and
reports per-input predictions plus systematic failure metrics.

Outputs:
  models/audit_predictions.txt — full per-input report
  models/audit_summary.json — machine-readable summary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ensemble_inference import EnsembleClassifier  # noqa: E402

CONFIG = ROOT / "models" / "ensemble_final" / "config.json"
OUT_TXT = ROOT / "models" / "audit_predictions.txt"
OUT_JSON = ROOT / "models" / "audit_summary.json"

# (text, expected_set, label)
# expected_set is the set of categories that SHOULD appear in the top-2 (or top-1 alone for clear cases)
TESTS: list[tuple[str, set[str], str]] = [
    # ---- Pure single-aspect: food quality ----
    ("الاكل بايخ ومالح", {"جودة الطعام"}, "food: bland+salty"),
    ("اللحم محروق وبدون طعم", {"جودة الطعام"}, "food: burnt"),
    ("البرجر مالح جدا", {"جودة الطعام"}, "food: salty burger"),
    ("الطبخ مو متقن", {"جودة الطعام"}, "food: poorly cooked"),

    # ---- Pure single-aspect: delivery ----
    ("المندوب تاخر ساعتين", {"التوصيل"}, "delivery: late driver"),
    ("ضاع الطلب ولا احد رد علي", {"التوصيل"}, "delivery: lost order"),
    ("السائق وصل لمكان غلط", {"التوصيل"}, "delivery: wrong address"),

    # ---- Pure single-aspect: staff service ----
    ("الموظف غير محترم وصاح علي", {"خدمة الموظفين"}, "service: rude"),
    ("الكاشير اسلوبه سيء", {"خدمة الموظفين"}, "service: bad attitude"),
    ("النادل ما يبتسم ابدا", {"خدمة الموظفين"}, "service: unfriendly waiter"),

    # ---- Pure single-aspect: wait time ----
    ("انتظرت ساعه كامله في المطعم قبل ان ياتي طلبي", {"وقت الانتظار"}, "wait: 1hr in-restaurant (USER FAILURE #2)"),
    ("الويتر ما جا الا بعد ٤٥ دقيقة", {"وقت الانتظار"}, "wait: 45 min for waiter"),
    ("الانتظار طويل جدا في المطعم", {"وقت الانتظار"}, "wait: long wait"),
    ("جلسنا ساعه ما حد جا", {"وقت الانتظار"}, "wait: hour, no one came"),

    # ---- Pure single-aspect: cleanliness ----
    ("المكان متسخ جدا", {"النظافة"}, "clean: dirty"),
    ("الطاولات مليانه ذباب", {"النظافة"}, "clean: flies on tables"),
    ("الحمام ما ينظف", {"النظافة"}, "clean: dirty bathroom"),

    # ---- Pure single-aspect: price ----
    ("الاسعار غاليه جدا", {"السعر والقيمة"}, "price: expensive"),
    ("الفاتوره مبالغ فيها", {"السعر والقيمة"}, "price: inflated bill"),
    ("ما يستاهل الفلوس", {"السعر والقيمة"}, "price: not worth it"),

    # ---- Pure single-aspect: order accuracy ----
    ("طلبت برجر بدون بصل لكنهم وضعوه", {"دقة الطلب"}, "accuracy: wrong order"),
    ("نسوا الكاتشب والصلصه", {"دقة الطلب"}, "accuracy: missing items"),
    ("ناقص في الطلب صنف", {"دقة الطلب"}, "accuracy: missing item"),

    # ---- Pure single-aspect: general ----
    ("تجربه سيئه عموما لن اعود لهذا المكان", {"عامة"}, "general: vague (USER FAILURE #3)"),
    ("ما عجبني المكان", {"عامة"}, "general: didn't like it"),

    # ---- Multi-aspect: food + delivery ----
    ("وصل الطلب بارد جدا والمندوب تاخر اكثر من ساعتين", {"جودة الطعام", "التوصيل"}, "multi: cold+late (USER FAILURE #1)"),
    ("الاكل بارد لما وصل والمندوب ضاع", {"جودة الطعام", "التوصيل"}, "multi: cold+lost"),

    # ---- Multi-aspect: service + price ----
    ("الموظف وقح والاسعار غاليه", {"خدمة الموظفين", "السعر والقيمة"}, "multi: rude+expensive"),

    # ---- Multi-aspect: cleanliness + food ----
    ("المكان متسخ والاكل سيء", {"النظافة", "جودة الطعام"}, "multi: dirty+bad food"),

    # ---- Multi-aspect: wait + accuracy ----
    ("انتظرت ساعتين وجاني الطلب غلط", {"وقت الانتظار", "دقة الطلب"}, "multi: wait+wrong"),

    # ---- Multi-aspect: wait + service ----
    ("انتظرت طويل والنادل ما يهتم", {"وقت الانتظار", "خدمة الموظفين"}, "multi: wait+unhelpful waiter"),

    # ---- Multi-aspect: food + service ----
    ("الاكل سيء والموظف غير محترم", {"جودة الطعام", "خدمة الموظفين"}, "multi: bad food+rude staff"),

    # ---- Tricky / edge ----
    ("الطلب وصل بارد", {"جودة الطعام", "التوصيل"}, "ambiguous: cold-on-arrival"),
    ("اكل ممتاز بس الانتظار طويل", {"وقت الانتظار"}, "positive food + wait complaint"),
]


def main() -> int:
    print(f"Loading ensemble from {CONFIG} ...")
    clf = EnsembleClassifier(CONFIG)
    print(f"  models loaded: {len(clf.models)}")
    print()

    lines: list[str] = []
    summary = {
        "total": len(TESTS),
        "top1_correct": 0,
        "top2_covers_expected": 0,
        "multi_aspect_caught": 0,
        "multi_aspect_total": 0,
        "per_category_correct": {},
        "per_category_total": {},
        "failures": [],
    }

    for cat in [
        "التوصيل", "السعر والقيمة", "النظافة", "جودة الطعام",
        "خدمة الموظفين", "دقة الطلب", "عامة", "وقت الانتظار",
    ]:
        summary["per_category_correct"][cat] = 0
        summary["per_category_total"][cat] = 0

    for text, expected, label in TESTS:
        result = clf.predict(text, top_k=3)
        top = result.top_3
        if not top:
            lines.append(f"\nABSTAIN: {label}\n  text: {text}\n  reason: {result.abstain_reason}")
            continue

        top1_cat, top1_score = top[0]
        top2_cat, top2_score = (top[1] if len(top) > 1 else (None, 0.0))
        top3_cat, top3_score = (top[2] if len(top) > 2 else (None, 0.0))

        is_multi_expected = len(expected) > 1
        if is_multi_expected:
            summary["multi_aspect_total"] += 1
            top2_set = {top1_cat, top2_cat}
            if expected.issubset(top2_set):
                summary["multi_aspect_caught"] += 1
                multi_status = "✓ multi caught"
            else:
                multi_status = f"✗ multi MISSED (expected {expected}, got {top2_set})"
        else:
            multi_status = ""

        # Single-aspect: top-1 must match the single expected cat
        # Multi-aspect: top-1 must be IN the expected set
        if top1_cat in expected:
            summary["top1_correct"] += 1
            top1_status = "✓"
        else:
            top1_status = f"✗ (expected {expected})"
            summary["failures"].append({
                "text": text,
                "label": label,
                "expected": list(expected),
                "got_top1": top1_cat,
                "got_top1_score": round(top1_score, 3),
                "got_top2": top2_cat,
                "got_top2_score": round(top2_score, 3),
            })

        top2_set = {top1_cat, top2_cat} if top2_cat else {top1_cat}
        if expected.issubset(top2_set):
            summary["top2_covers_expected"] += 1

        primary_expected = list(expected)[0]
        summary["per_category_total"][primary_expected] += 1
        if top1_cat in expected:
            summary["per_category_correct"][primary_expected] += 1

        lines.append(f"\n[{label}]")
        lines.append(f"  text:     {text}")
        lines.append(f"  expected: {expected}")
        lines.append(f"  top1:     {top1_cat:<18} {top1_score:.0%}  {top1_status}")
        if top2_cat:
            lines.append(f"  top2:     {top2_cat:<18} {top2_score:.0%}")
        if top3_cat:
            lines.append(f"  top3:     {top3_cat:<18} {top3_score:.0%}")
        if multi_status:
            lines.append(f"  multi:    {multi_status}")

    # Aggregate
    lines.insert(0, "=" * 70)
    lines.insert(1, "AUDIT REPORT — production ensemble (current state)")
    lines.insert(2, "=" * 70)
    lines.insert(3, "")
    lines.insert(4, f"Total tests: {summary['total']}")
    lines.insert(5, f"Top-1 correct: {summary['top1_correct']}/{summary['total']} ({summary['top1_correct']/summary['total']:.0%})")
    lines.insert(6, f"Top-2 covers expected: {summary['top2_covers_expected']}/{summary['total']} ({summary['top2_covers_expected']/summary['total']:.0%})")
    lines.insert(7, f"Multi-aspect cases caught: {summary['multi_aspect_caught']}/{summary['multi_aspect_total']}")
    lines.insert(8, "")
    lines.insert(9, "Per-category top-1 accuracy (this audit set, not the held-out test):")
    for cat in summary["per_category_correct"]:
        c = summary["per_category_correct"][cat]
        t = summary["per_category_total"][cat]
        if t:
            lines.insert(10, f"  {cat:<18} {c}/{t}")
    lines.insert(10, "")
    lines.insert(11, "=" * 70)
    lines.insert(12, "PER-INPUT DETAILS")
    lines.insert(13, "=" * 70)

    txt = "\n".join(lines)
    OUT_TXT.write_text(txt + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(txt)
    print(f"\nSaved -> {OUT_TXT}")
    print(f"Saved -> {OUT_JSON}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
