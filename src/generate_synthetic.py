"""
Generate synthetic Arabic complaints for small categories.
Brings every small category up to TARGET_PER_CATEGORY.

Outputs to data/text/synthetic_complaints.csv (kept separate so it can
be inspected / removed if needed).
"""
import csv
import os
import random
from collections import Counter

random.seed(42)

INPUT = "data/text/complaints_labeled.csv"
OUTPUT = "data/text/synthetic_complaints.csv"
TARGET_PER_CATEGORY = 2500


# ---- vocabulary ----
ITEMS = [
    "برجر", "شاورما", "بيتزا", "شيش طاووق", "مندي", "كبسة", "فلافل",
    "حمص", "سلطة سيزر", "وجبة دجاج", "عصير برتقال", "كولا", "سندوتش",
    "وجبة عائلية", "بروستد", "نقانق", "مشاوي", "باستا", "سوشي", "كنافة",
    "بطاطا مقلية", "وجبة الاطفال", "طلب الفطور", "وجبة العشاء", "سمك مشوي",
    "كباب", "كفتة", "تكا", "كرسبي", "ناجتس", "ساندويش لحم", "تشيكن رول",
    "وجبة الغداء", "سلطة فتوش", "متبل", "ورق عنب", "سمبوسه", "بقلاوه",
    "ايس كريم", "ميلك شيك", "قهوه عربيه", "شاي اخضر", "شوربه عدس",
    "فاهيتا", "بانكيك", "كروسان", "مشكل خضار", "اوزي", "بخاري", "صياديه",
    "مكرونه", "نودلز", "فطيره", "معجنات", "زبدية حمص", "مقبلات",
]

MODIFIERS = [
    "بصل", "طماطم", "خس", "صلصة حارة", "مايونيز", "جبنة", "مخلل",
    "خبز", "ثوم", "بطاطس", "صوص", "كاتشب", "خردل", "زبدة",
    "زيتون", "فلفل", "ليمون", "طحينه", "خل", "زنجبيل", "نعناع",
    "كزبرة", "بقدونس", "ملح", "سكر", "صلصة بشاميل", "كريمة",
    "صلصة طماطم", "صوص ثوم", "صلصة باربكيو", "حلاوه", "مكسرات",
]

WRONG_ITEMS = [
    "وجبة شخص ثاني", "صنف مختلف تماما", "نوع اخر من الطعام",
    "ما لم اطلبه", "وجبة بارده", "حجم اصغر", "وجبة منتهية الصلاحيه",
    "صنف بسعر اعلى", "وجبة قديمة", "اكل مغاير", "طلب لشخص اخر",
    "وجبة محشية بمكونات لم اطلبها", "نسخة غير صحيحه من الطلب",
    "وجبة بدون مكونات اساسيه", "صنف ليس على القائمه", "اكل غير ناضج",
]

QUANTITIES = [
    "اثنين", "ثلاث", "اربع", "خمس", "وحده", "نصف الكميه", "ربع الكميه",
]

APPS = [
    "هنقرستيشن", "جاهز", "مرسول", "طلبات", "التطبيق", "ابلكيشن التوصيل",
    "تطبيق الطلب", "البرنامج", "خدمة التوصيل", "الموقع",
]
TIMES = [
    "ساعة", "ساعتين", "ثلاث ساعات", "نصف ساعة", "وقت طويل", "اكثر من ساعة",
    "نصف ساعة اضافية", "ساعه ونصف", "اربع ساعات", "وقت يفوق المعقول",
    "اكثر مما هو مكتوب", "وقت طويل جدا", "مدة اطول من المتوقع",
]
TEMPS = [
    "بارده جدا", "ساخنه بشكل مفرط", "متوسطة الحراره", "غير مناسبة الحراره",
    "بارده غير صالحه للاكل", "ساخنه لدرجة الاحتراق", "بدرجه حراره غير عاديه",
]
DRIVER_ISSUES = [
    "لم يجد العنوان", "لم يتصل قبل الوصول", "كان متعجل", "لم يحترم الوقت",
    "اخطا في رقم المنزل", "وصل لمكان اخر", "تجاهل تعليمات التوصيل",
    "كان يتصل اكثر من مرة", "تكلم بطريقة غير مهذبه", "تاخر في تسليم الطلب",
    "ضاع منه العنوان", "لم يستلم تنبيه الطلب",
]


# ---------- TEMPLATES ----------

def t_order_accuracy():
    item = random.choice(ITEMS)
    mod = random.choice(MODIFIERS)
    mod2 = random.choice(MODIFIERS)
    wrong = random.choice(WRONG_ITEMS)
    qty = random.choice(QUANTITIES)
    item2 = random.choice(ITEMS)
    p = random.randint(0, 19)
    if p == 0: return f"طلبت {item} لكن وصلني {wrong} بدلا منه"
    if p == 1: return f"الطلب ناقص نسوا {mod} والوجبه غير مكتمله"
    if p == 2: return f"طلبت {item} بدون {mod} لكنهم وضعوه رغم تنبيهي"
    if p == 3: return f"الكميه كانت غلط طلبت {qty} {item} واستلمت اقل"
    if p == 4: return f"وصلني طلب شخص ثاني بالخطا {wrong} وليس ما طلبته"
    if p == 5: return f"طلبت {item} لكنهم بدلوه ب{wrong} دون اخباري"
    if p == 6: return f"لم يكن هذا ما طلبت انا طلبت {item} ووصلني {item2}"
    if p == 7: return f"الطلب مختلف عن الفاتوره اضافوا {item2} ولم اطلبه"
    if p == 8: return f"طلبي ناقص صنف {item} ولم يتم تعويضي عنه"
    if p == 9: return f"اخطاوا في تحضير {item} وضعوا {mod} رغم اني طلبت بدونه"
    if p == 10: return f"وجبة {item} وصلت ناقصه المكونات الاضافات غير صحيحه"
    if p == 11: return f"طلبت {item} مع {mod} لكن وصل بدون {mod} ابدا"
    if p == 12: return f"الطلب فيه اخطاء كثيره الكميه غلط و{item} مفقود"
    if p == 13: return f"نسوا اضافة {mod} و{mod2} في الطلب رغم اني دفعت ثمنها"
    if p == 14: return f"طلبت {item} ووصل صنف اخر تماما هذا ليس ما طلبت ابدا"
    if p == 15: return f"اضافوا {mod} الى {item} رغم انني حذفته من الطلب"
    if p == 16: return f"طلبي يحتوي على {qty} {item} وصلني فقط {wrong}"
    if p == 17: return f"ارسلوا {wrong} بدلا من {item} الذي طلبته"
    if p == 18: return f"خلطوا الطلبات وصلني {item2} بدلا من {item}"
    return f"المطعم نسي {item} من الطلب وارسل {item2} بدلا عنه"


def t_delivery():
    app = random.choice(APPS)
    time = random.choice(TIMES)
    item = random.choice(ITEMS)
    temp = random.choice(TEMPS)
    issue = random.choice(DRIVER_ISSUES)
    p = random.randint(0, 19)
    if p == 0: return f"طلبت من {app} منذ {time} ولم يصل الطلب حتى الان"
    if p == 1: return f"المندوب تاخر علي اكثر من {time} والطلب وصل {temp}"
    if p == 2: return f"السائق {issue} والطلب لم يصل في الوقت المحدد"
    if p == 3: return f"الطلب ضاع لم يصل ولم يرد علي خدمة العملاء في {app}"
    if p == 4: return f"تم خصم المبلغ من حسابي ولم اتلق الطلب من {app}"
    if p == 5: return f"التوصيل بطيء جدا في {app} انتظرت {time} ثم الغيت"
    if p == 6: return f"رسوم التوصيل مرتفعه جدا والوجبه وصلت {temp} لا تستحق"
    if p == 7: return f"المندوب {issue} ووصل لمكان اخر تماما"
    if p == 8: return f"وعدوني بالتوصيل خلال نصف ساعه لكن مر {time} ولم يصل"
    if p == 9: return f"تتبع الطلب لا يعمل في {app} ولا اعرف اين المندوب"
    if p == 10: return f"وصل الطلب {temp} بسبب طول مدة التوصيل {time}"
    if p == 11: return f"المندوب {issue} ولم يحاول التواصل لتسليم الطلب"
    if p == 12: return f"الطلب من {app} لم يصل ولم اتلق اي اشعار بالغاء"
    if p == 13: return f"التوصيل في وقت الذروة سيء جدا انتظرت {time} في {app}"
    if p == 14: return f"وقت التوصيل المتوقع كان نصف ساعه واستمر {time} كاملة"
    if p == 15: return f"السائق {issue} و{item} وصل {temp}"
    if p == 16: return f"خدمة التوصيل في {app} مزريه السائق {issue}"
    if p == 17: return f"طلبت {item} منذ {time} والمندوب {issue}"
    if p == 18: return f"السائق وصل بعد {time} و{item} كان {temp}"
    return f"تجربه سيئه مع {app} الانتظار {time} والمندوب {issue}"


AMBIANCE_PARTS = [
    "الكراسي", "الديكور", "الاضاءه", "التكييف", "المقاعد", "الطاولات",
    "الموسيقى", "المساحه", "التهويه", "الجلسه", "موقف السيارات", "الازدحام",
    "الالوان", "الحرارة", "تصميم المكان", "الحمامات", "الواجهة",
]
AMBIANCE_ADJ = [
    "سيئه", "قديمه", "مزعجه", "ضعيفه", "غير مريحه", "متهالكه", "متسخه",
    "مكتظه", "صغيره", "مزدحمه", "مهتزه", "خانقه", "غير مناسبه",
]
AMBIANCE_ISSUE = [
    "لا تناسب نوع المطعم", "تحتاج تجديد", "تتعب الزبون", "لا توحي بالراحه",
    "غير لائقه بالسعر", "محبطه", "تشعرني بالضيق", "لا تستحق الزيارة",
    "تفسد التجربه كاملة", "غير مقبوله", "لا تليق بمطعم بهذا الاسم",
]


def t_ambiance():
    part = random.choice(AMBIANCE_PARTS)
    adj = random.choice(AMBIANCE_ADJ)
    issue = random.choice(AMBIANCE_ISSUE)
    part2 = random.choice(AMBIANCE_PARTS)
    adj2 = random.choice(AMBIANCE_ADJ)
    p = random.randint(0, 11)
    if p == 0: return f"{part} {adj} و{part2} {adj2} المكان {issue}"
    if p == 1: return f"{part} في المطعم {adj} جدا و{issue}"
    if p == 2: return f"المكان فيه مشاكل كثيره {part} {adj} و{part2} {adj2}"
    if p == 3: return f"الديكور قديم و{part} {adj} الجو {issue}"
    if p == 4: return f"{part} كانت {adj} طوال الوقت لا توجد راحه ابدا"
    if p == 5: return f"عند الدخول لاحظت ان {part} {adj} و{part2} ايضا"
    if p == 6: return f"المطعم يحتاج تطوير {part} {adj} والاجواء {issue}"
    if p == 7: return f"تجربه سيئه من ناحية الديكور {part} {adj} {issue}"
    if p == 8: return f"الجو العام للمطعم سيء {part} {adj} والصاله ضيقة"
    if p == 9: return f"{part} {adj} مما جعل الجلسه غير ممتعه ابدا"
    if p == 10: return f"الاجواء {issue} بسبب ان {part} {adj}"
    return f"المكان غير لائق {part} {adj} و{part2} {adj2}"


GENERAL_NEG = [
    "تجربه سيئه", "خيبة امل", "تجربه محبطه", "غير راضي عن الزياره",
    "لم تعجبني التجربه", "تجربه دون المستوى", "زياره فاشله",
    "تجربه لا تستحق", "تجربه عاديه جدا", "تجربه باهته",
]
GENERAL_RESTO = [
    "هذا المطعم", "المكان", "هذا الفرع", "هذه السلسله", "المطعم بشكل عام",
    "هذا النوع من المطاعم", "المحل", "هذا المقهى",
]
GENERAL_REASON = [
    "لاسباب كثيره", "لعدم اهتمامهم بالعميل", "بسبب تراجع المستوى",
    "لانهم لا يحترمون وقت الزبون", "لان الجوده متدنيه عموما",
    "لان المستوى لا يستحق", "لان السعر لا يتناسب مع الخدمه",
    "بسبب سوء التنظيم العام", "لان الادارة غير محترفه",
]
GENERAL_VERDICT = [
    "لن اعود مره اخرى", "لا انصح به", "لن اكرر التجربه",
    "ابحثوا عن مكان افضل", "خياراتكم خارجه افضل بكثير",
    "لم يعجبني ابدا", "تجربتي كانت اقل من المتوقع", "افضل الذهاب لمكان اخر",
]


def t_general():
    neg = random.choice(GENERAL_NEG)
    resto = random.choice(GENERAL_RESTO)
    reason = random.choice(GENERAL_REASON)
    verdict = random.choice(GENERAL_VERDICT)
    p = random.randint(0, 9)
    if p == 0: return f"{neg} مع {resto} {reason}"
    if p == 1: return f"{neg} {verdict}"
    if p == 2: return f"{resto} {reason} و{verdict}"
    if p == 3: return f"السمعه افضل من الواقع {neg} و{verdict}"
    if p == 4: return f"{resto} لم يقنعني {reason}"
    if p == 5: return f"بعد عدة زيارات {neg} {verdict}"
    if p == 6: return f"تواقعاتي كانت اعلى {resto} {reason}"
    if p == 7: return f"بشكل عام {neg} مع {resto} {verdict}"
    if p == 8: return f"{neg} كل شيء كان متوسط {reason}"
    return f"{verdict} {resto} {reason}"


GENERATORS = {
    "دقة الطلب": t_order_accuracy,
    "التوصيل": t_delivery,
    "الجو والمكان": t_ambiance,
    "عامة": t_general,
}


def existing_counts():
    counts = Counter()
    seen = set()
    with open(INPUT, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            counts[r["category"]] += 1
            seen.add(r["text"])
    return counts, seen


def main():
    counts, seen = existing_counts()
    print("Current per-category counts:")
    for cat in GENERATORS:
        print(f"  {cat}: {counts.get(cat, 0)}")

    rows = []
    for cat, gen in GENERATORS.items():
        need = max(0, TARGET_PER_CATEGORY - counts.get(cat, 0))
        if need == 0:
            print(f"\n{cat}: already at target, skipping")
            continue
        print(f"\n{cat}: generating {need} synthetic rows...")
        added = 0
        attempts = 0
        while added < need and attempts < need * 50:
            attempts += 1
            t = gen()
            if t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "category": cat, "priority": "متوسطة", "source": "synthetic"})
            added += 1
        print(f"  added {added} rows")

    print(f"\nTotal new synthetic rows: {len(rows)}")

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved to {OUTPUT}")

    # also append to main labeled file
    with open(INPUT, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "category", "priority", "source"])
        writer.writerows(rows)
    print(f"Appended to {INPUT}")


if __name__ == "__main__":
    main()
