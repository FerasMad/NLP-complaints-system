"""
Generate synthetic Arabic complaints for the four small categories.

v2: rewritten to mimic real Saudi/Gulf food-app reviews. Real reviews are:
  - dialectal (وش، ياليت، بس، يا اخي، والله) not formal MSA
  - fragmented (often 2-5 words: "كل شيئ فيه سيء", "الطلب ناقص")
  - emotionally repetitive (جدا جدا، مرة مرة، ابدا ابدا)
  - punctuation-heavy or punctuation-free
  - mixed register (MSA + dialect in same sentence)
  - sometimes code-switched (delivery, WiFi, AC, online)

Strategy: 60+ templates per generator, with explicit short/medium/long mix,
larger vocabularies, dialect markers throughout. Output written to
data/text/synthetic_complaints.csv (kept separate so it can be inspected /
removed if needed).
"""
import csv
import os
import random
import sys
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

random.seed(42)

INPUT = "data/text/complaints_labeled.csv"
OUTPUT = "data/text/synthetic_complaints.csv"
TARGET_PER_CATEGORY = 2500


# ============================================================
# SHARED VOCABULARIES
# ============================================================

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
    "هريسه", "مفطح", "جريش", "مرقوق", "مطبق", "سليق", "سمبوسك",
    "ميني برجر", "تشيكن سلايدر", "وجبة الاطفال هابي ميل", "بوكس دجاج",
]

MODIFIERS = [
    "بصل", "طماطم", "خس", "صلصة حارة", "مايونيز", "جبنة", "مخلل",
    "خبز", "ثوم", "بطاطس", "صوص", "كاتشب", "خردل", "زبدة",
    "زيتون", "فلفل", "ليمون", "طحينه", "خل", "زنجبيل", "نعناع",
    "كزبرة", "بقدونس", "ملح", "سكر", "صلصة بشاميل", "كريمة",
    "صلصة طماطم", "صوص ثوم", "صلصة باربكيو", "حلاوه", "مكسرات",
    "مشروم", "ذره", "فلفل حار", "صلصه حلوه", "شطه", "صلصه شيدر",
    "بهارات", "زعتر", "عسل", "تمر", "رمان",
]

WRONG_ITEMS = [
    "وجبة شخص ثاني", "صنف مختلف تماما", "نوع اخر من الطعام",
    "ما لم اطلبه", "وجبة بارده", "حجم اصغر", "وجبة منتهية الصلاحيه",
    "صنف بسعر اعلى", "وجبة قديمه", "اكل مغاير", "طلب لشخص اخر",
    "وجبه محشيه بمكونات لم اطلبها", "نسخه غير صحيحه من الطلب",
    "وجبه بدون مكونات اساسيه", "صنف ليس على القائمه", "اكل غير ناضج",
    "طلب نص مفقود", "وجبه ناقصه نصها", "صنف اخر مو طلبي",
    "اكل لشخص ثاني تماما",
]

QUANTITIES = [
    "اثنين", "ثلاث", "اربع", "خمس", "وحده", "نصف الكميه", "ربع الكميه",
    "اكثر من ثلاث", "ست قطع", "اربع قطع",
]

APPS = [
    "هنقرستيشن", "جاهز", "مرسول", "طلبات", "التطبيق", "ابلكيشن التوصيل",
    "تطبيق الطلب", "البرنامج", "خدمه التوصيل", "الموقع", "التطبيق هذا",
    "هذا التطبيق", "تطبيقكم", "ابكم",
]
TIMES = [
    "ساعه", "ساعتين", "ثلاث ساعات", "نصف ساعه", "وقت طويل", "اكثر من ساعه",
    "نصف ساعه اضافيه", "ساعه ونص", "اربع ساعات", "وقت يفوق المعقول",
    "اكثر مما هو مكتوب", "وقت طويل جدا", "مده اطول من المتوقع",
    "اكثر من ساعتين", "ثلث ساعه زياده", "ساعه كامله", "تقريبا ساعه",
    "ساعتين ونص", "وقت طويل مرره",
]
TEMPS = [
    "بارد جدا", "بارده", "ساخن بشكل مفرط", "متوسط الحراره", "غير مناسب الحراره",
    "بارد غير صالح للاكل", "ساخن لدرجه الاحتراق", "بدرجه حراره غير عاديه",
    "بارد ميت", "فاتر", "بارد كانه طلع من الثلاجه",
]
DRIVER_ISSUES = [
    "ما لقى العنوان", "ما اتصل قبل الوصول", "كان متعجل", "ما احترم الوقت",
    "غلط في رقم المنزل", "وصل لمكان ثاني", "تجاهل تعليمات التوصيل",
    "كان يتصل اكثر من مره", "تكلم بطريقه غير مهذبه", "تاخر في تسليم الطلب",
    "ضاع منه العنوان", "ما استلم تنبيه الطلب", "كلامه غير لائق",
    "اسلوبه سيء", "ما يفهم العنوان", "ضايقني", "اساء الادب", "تطفل",
]


# ============================================================
# DIALECT / STYLE HELPERS
# ============================================================

# Common Saudi/Gulf intensifiers and frustration markers
INTENSIFIERS = [
    "جدا", "جدا جدا", "مره", "مره مره", "مرره", "كثير", "كثير كثير",
    "بشكل مبالغ فيه", "نهائي", "ابدا ابدا", "للاسف", "والله", "يا اخي",
]

NEGATIVE_OPENERS = [
    "والله", "يا اخي", "يا جماعه", "صراحه", "بصراحه", "الصراحه",
    "للاسف", "قسما", "اقسم بالله", "تخيل", "شوفوا",
]

CLOSING_VERDICTS = [
    "لن اعود", "ما اطلب منهم تاني", "ابحثوا عن غيره", "نهائي ما اعود",
    "خلاص ما اطلب", "ما يستاهل", "زفت", "خايس", "تجربه فاشله",
    "تجربه سيئه جدا", "ولا انصح", "لا انصح ابدا", "ابدا ما انصح",
    "تخيب الامل", "محبط", "احباط", "تعبني", "ما احب اشكي بس",
]

# Code-switch English words common in Saudi reviews
ENGLISH_TERMS = [
    "delivery", "online", "cash", "wifi", "AC", "menu", "offer",
    "promo", "voucher", "service", "support", "rating",
]


def maybe_dialect(text):
    """20% chance to add a dialect intensifier somewhere."""
    if random.random() < 0.2:
        return f"{text} {random.choice(INTENSIFIERS)}"
    return text


def maybe_punct(text):
    """30% chance to add expressive punctuation."""
    r = random.random()
    if r < 0.05:
        return text + "!!!"
    if r < 0.10:
        return text + "..."
    if r < 0.15:
        return text + " ،"
    if r < 0.30:
        return text + "."
    return text


def maybe_opener(text):
    """15% chance to prepend an emotional opener."""
    if random.random() < 0.15:
        return f"{random.choice(NEGATIVE_OPENERS)} {text}"
    return text


def stylize(text):
    """Apply dialect/punctuation/opener noise. Use sparingly so meaning is preserved."""
    return maybe_punct(maybe_opener(text))


# ============================================================
# ORDER ACCURACY  (دقة الطلب) — 60+ templates
# ============================================================

def t_order_accuracy():
    item = random.choice(ITEMS)
    item2 = random.choice(ITEMS)
    mod = random.choice(MODIFIERS)
    mod2 = random.choice(MODIFIERS)
    wrong = random.choice(WRONG_ITEMS)
    qty = random.choice(QUANTITIES)
    intens = random.choice(INTENSIFIERS)

    p = random.randint(0, 64)

    # ---- short fragments (mimics real choppy reviews) ----
    if p == 0: return stylize(f"الطلب جا ناقص")
    if p == 1: return stylize(f"كل طلباتهم غلط")
    if p == 2: return stylize(f"الطلب غلط دائما")
    if p == 3: return stylize(f"دايم ناقص")
    if p == 4: return stylize(f"كل ما اطلب يجي شي غلط")
    if p == 5: return stylize(f"الطلب مو طلبي")
    if p == 6: return stylize(f"وجبه ناقصه")
    if p == 7: return stylize(f"غلط في غلط")
    if p == 8: return stylize(f"دايما ناقص دايما غلط")
    if p == 9: return stylize(f"يجيب طلب ثاني مو طلبي")
    if p == 10: return stylize(f"اخطاء في كل طلب")

    # ---- medium-length, dialectal ----
    if p == 11: return stylize(f"طلبت {item} وجاني {wrong} وش هذا")
    if p == 12: return stylize(f"كل طلباتهم غلط في غلط ما اطلب منهم تاني")
    if p == 13: return stylize(f"الطلب جا ناقص كم مره ما يتعلمون")
    if p == 14: return stylize(f"يا اخي طلبت {item} ووصلني {item2} مو معقول")
    if p == 15: return stylize(f"دايم الطلبات تجي ناقصه ودايم نفس المشكله")
    if p == 16: return stylize(f"طلبت بدون {mod} لكن وضعوه {intens}")
    if p == 17: return stylize(f"وش هذا التطبيق طلبي ناقص نص الاشياء")
    if p == 18: return stylize(f"كل ما اطلب يوصلني شي غلط او ناقص")
    if p == 19: return stylize(f"الطلب فيه {qty} {item} ناقصه")
    if p == 20: return stylize(f"اضافات غير صحيحه ولا تشبه طلبي")
    if p == 21: return stylize(f"خلطوا الطلبات استلمت طلب جاري ما طلبي")
    if p == 22: return stylize(f"نسوا {mod} و{mod2} وانا دفعت ثمنها")
    if p == 23: return stylize(f"بدلوا {item} ب{wrong} بدون ما يخبروني")
    if p == 24: return stylize(f"الفاتوره فيها {item} لكن وصلني {item2} فقط")
    if p == 25: return stylize(f"طلب الكمبو ناقص الشراب ولا البطاطس")

    # ---- longer, multi-clause, frustrated ----
    if p == 26: return stylize(f"تجربه سيئه جدا الطلب جاني ناقص كلمت خدمه العملاء يقولون ارسل ايميل ارسل ايميل بس ما يردون")
    if p == 27: return stylize(f"المطعم نسي {item} من الطلب وارسل {item2} بدلا منه ما اعرف وش يصير في المطبخ")
    if p == 28: return stylize(f"للاسف اصبح تطبيق فاشل لا يوضح طلبات العميل اما يوصل ناقص او غير المطلوب")
    if p == 29: return stylize(f"كل مره اطلب كل مره ينقص شي دايم نفس القصه ما يتعلمون من الاخطاء")
    if p == 30: return stylize(f"طلبت {item} مع تعديلات معينه يا ليتهم قروها كلها وصلت غلط بدون {mod}")
    if p == 31: return stylize(f"وصلني الطلب وفتحته القيت ان فيه {item2} مو {item} اللي طلبته كلمت الدعم بدون فايده")
    if p == 32: return stylize(f"الطلب جاني ودفعت كامل المبلغ لقيت ان الكميه ناقصه واضافات غير صحيحه")

    # ---- with refund/customer service angle ----
    if p == 33: return stylize(f"دفعت ولم اتلق الطلب كامل تواصلت مع الدعم بدون فايده")
    if p == 34: return stylize(f"الطلب ناقص وما تم تعويضي عنه")
    if p == 35: return stylize(f"رفعت تذكره دعم بسبب نقص الطلب ما رد علي احد")
    if p == 36: return stylize(f"المبلغ خصم كامل لكن الطلب فيه نقص واضح")
    if p == 37: return stylize(f"حذفوا اضافات وما خصموا قيمتها من الفاتوره")
    if p == 38: return stylize(f"نقص الطلب صار عاده دائمه عندهم وفي كل مره يعتذرون فقط")

    # ---- with item names + specific numbers ----
    if p == 39: return stylize(f"طلبت {qty} {item} استلمت اقل بكثير")
    if p == 40: return stylize(f"الكمبو فيه {qty} اشياء وصلني فقط واحد")
    if p == 41: return stylize(f"ينقص دائما {item} او {mod} او كلاهما")
    if p == 42: return stylize(f"طلبي يحتوي على ثلاث وجبات وصلني وجبتين")

    # ---- code-switched ----
    if p == 43: return stylize(f"طلبي جاء غلط online ولا فيه help من الدعم")
    if p == 44: return stylize(f"order كله غلط وما فيه refund")
    if p == 45: return stylize(f"كل order يجي ناقص نفس الشي")

    # ---- direct dialect with strong opinion ----
    if p == 46: return stylize(f"يا اخي الطلب ناقص من زمان نفس المشكله ما تنحل")
    if p == 47: return stylize(f"والله ما اطلب منهم بعد طلبي يجي غلط دايما")
    if p == 48: return stylize(f"بصراحه الطلب جاني غلط تماما لا انصح ابدا")
    if p == 49: return stylize(f"شوفوا يا جماعه الطلب فيه نقص واضح")
    if p == 50: return stylize(f"للاسف اخطاء كل طلب نسوا {mod} ضافوا {mod2}")

    # ---- vague but pointed ----
    if p == 51: return stylize(f"طلب فاشل ناقص في كل مره")
    if p == 52: return stylize(f"دقه الطلب صفر")
    if p == 53: return stylize(f"دقتهم في الطلب سيئه جدا")
    if p == 54: return stylize(f"اخطاء كثيره في كل طلب")
    if p == 55: return stylize(f"يخلطون الطلبات دايما")
    if p == 56: return stylize(f"المطعم ما يعرف يحضر الطلب صح")
    if p == 57: return stylize(f"الكاشير او المطبخ كلهم غلط")

    # ---- specific complaint patterns from real text ----
    if p == 58: return stylize(f"اول الطلب مكتوب عليه بدون {mod} لكن وصل مع {mod}")
    if p == 59: return stylize(f"كتبت ملاحظه واضحه بدون {mod} لكن ما قروها")
    if p == 60: return stylize(f"بدون {mod} مكتوبه بالعربي وبالانجليزي وما اعتمدوها")
    if p == 61: return stylize(f"الباركود يوضح الطلب بس المطعم تجاهله")
    if p == 62: return stylize(f"احيانا كثير تحصل غلطات احنا اللي ندفع ثمنها")
    if p == 63: return stylize(f"كل مره يجيب {item2} بدل {item} حتى اللي ما طلبته يجيبه")
    return stylize(f"وجبه {item} وصلت ناقصه المكونات الاضافات غير صحيحه ابدا")


# ============================================================
# DELIVERY  (التوصيل) — 60+ templates
# ============================================================

def t_delivery():
    app = random.choice(APPS)
    time = random.choice(TIMES)
    item = random.choice(ITEMS)
    temp = random.choice(TEMPS)
    issue = random.choice(DRIVER_ISSUES)
    intens = random.choice(INTENSIFIERS)

    p = random.randint(0, 64)

    # ---- short fragments ----
    if p == 0: return stylize(f"التوصيل بطيء")
    if p == 1: return stylize(f"رسوم توصيل غاليه")
    if p == 2: return stylize(f"التوصيل غالي مره")
    if p == 3: return stylize(f"المندوب تاخر")
    if p == 4: return stylize(f"الطلب وصل بارد")
    if p == 5: return stylize(f"ما وصل الطلب")
    if p == 6: return stylize(f"وين الطلب")
    if p == 7: return stylize(f"الطلب ضايع")
    if p == 8: return stylize(f"التوصيل فاشل")
    if p == 9: return stylize(f"التوصيل زفت")
    if p == 10: return stylize(f"المندوب ما رد")

    # ---- medium with dialect ----
    if p == 11: return stylize(f"طلبت من {app} من {time} وما وصل الطلب لين الحين")
    if p == 12: return stylize(f"المندوب تاخر علي اكثر من {time} والطلب وصل {temp}")
    if p == 13: return stylize(f"السائق {issue} والطلب ما وصل في الوقت المحدد")
    if p == 14: return stylize(f"الطلب ضاع ما وصل وما رد علي خدمه العملاء في {app}")
    if p == 15: return stylize(f"تم خصم المبلغ من حسابي ولم اتلق الطلب من {app}")
    if p == 16: return stylize(f"التوصيل بطيء جدا في {app} انتظرت {time} ثم الغيت")
    if p == 17: return stylize(f"رسوم التوصيل مرتفعه جدا والوجبه وصلت {temp} لا تستحق")
    if p == 18: return stylize(f"المندوب {issue} ووصل لمكان ثاني تماما")
    if p == 19: return stylize(f"وعدوني بالتوصيل خلال نص ساعه لكن مر {time} وما وصل")
    if p == 20: return stylize(f"تتبع الطلب لا يعمل في {app} ولا اعرف وين المندوب")
    if p == 21: return stylize(f"وصل الطلب {temp} بسبب طول مده التوصيل {time}")
    if p == 22: return stylize(f"المندوب {issue} وما حاول التواصل لتسليم الطلب")
    if p == 23: return stylize(f"الطلب من {app} ما وصل وما اتلقيت اي اشعار بالغاء")
    if p == 24: return stylize(f"التوصيل في وقت الذروه سيء جدا انتظرت {time} في {app}")
    if p == 25: return stylize(f"وقت التوصيل المتوقع كان نص ساعه واستمر {time} كامله")
    if p == 26: return stylize(f"السائق {issue} و{item} وصل {temp}")
    if p == 27: return stylize(f"خدمه التوصيل في {app} مزريه السائق {issue}")
    if p == 28: return stylize(f"طلبت {item} من {time} والمندوب {issue}")
    if p == 29: return stylize(f"السائق وصل بعد {time} و{item} كان {temp}")

    # ---- longer, frustrated ----
    if p == 30: return stylize(f"تجربه سيئه مع {app} الانتظار {time} والمندوب {issue} والاكل وصل {temp}")
    if p == 31: return stylize(f"للاسف التوصيل فاشل طالب الساعه واحد وصلني الساعه ثلاث واكل بارد فوق هذا السعر غالي")
    if p == 32: return stylize(f"يا جماعه التوصيل في {app} اصبح كارثه انتظر {time} لكل طلب")
    if p == 33: return stylize(f"رسوم التوصيل في الزياده وقت الذروه مبالغه جدا تكلفه التوصيل اكثر من الطلب نفسه")
    if p == 34: return stylize(f"المطاعم اللي طلبت منها من ارجع اطلب منها تختفي تعامل بعض المندوبين مو حلو")
    if p == 35: return stylize(f"التطبيق رسوم الالغاء عاليه التوصيل سيء السعر غالي خدمه عملاء فاشله")

    # ---- payment / app issues ----
    if p == 36: return stylize(f"التطبيق ياخذ الفلوس قبل التوصيل لكن لا يضمن وصول الطلب")
    if p == 37: return stylize(f"الدفع تم لكن ما وصل اي طلب لين الحين")
    if p == 38: return stylize(f"رفضوا استرداد الفلوس بعد فشل التوصيل")
    if p == 39: return stylize(f"الكوبونات ما تشتغل ورسوم التوصيل غاليه")
    if p == 40: return stylize(f"تطبيق ياخذ المبلغ ولا يوصل الطلب وما يرد على العميل")

    # ---- driver behavior focused ----
    if p == 41: return stylize(f"المندوب اساء الادب")
    if p == 42: return stylize(f"السائق يكلم بطريقه غير لائقه")
    if p == 43: return stylize(f"المندوب ضايقني وانا متضايق من الطلب")
    if p == 44: return stylize(f"السائق رفض يجيب الطلب لين باب البيت")
    if p == 45: return stylize(f"تواصل المندوب سيء جدا اكثر من اتصال للعنوان")

    # ---- general delivery complaints ----
    if p == 46: return stylize(f"التوصيل غالي مره وغير كذا مندوب توصيل ضايقني")
    if p == 47: return stylize(f"اسعار التوصيل مرتفعه يا ليت تنظرون فيها")
    if p == 48: return stylize(f"تجربه فاشله مع التوصيل لن تتكرر")
    if p == 49: return stylize(f"التوصيل عندهم اسوا من اي تطبيق ثاني")
    if p == 50: return stylize(f"كل تطبيقات التوصيل افضل من {app}")

    # ---- code-switched ----
    if p == 51: return stylize(f"delivery كله سيء و{app} ما يهتم بالعميل")
    if p == 52: return stylize(f"التوصيل online ضعيف service مزري")
    if p == 53: return stylize(f"delivery time غلط دايما")
    if p == 54: return stylize(f"رسوم delivery غاليه ما تستاهل")

    # ---- specific complaint patterns from real text ----
    if p == 55: return stylize(f"يطولون في التوصيل والتوصيل غالي مره يوصل ثلاثين ريال طلب وجبه")
    if p == 56: return stylize(f"المندوب ضاع منه العنوان واتصل عشر مرات")
    if p == 57: return stylize(f"تطبيق رسوم الغاء عاليه التوصيل سيء السعر غالي")
    if p == 58: return stylize(f"خدمه عملاء سيئه ما تفيد دفعت رسوم التوصيل مرتين")
    if p == 59: return stylize(f"يضللون الموقع وياخذون رسوم توصيل زياده")
    if p == 60: return stylize(f"خدمه التوصيل اصبحت سيئه جدا مقارنه بالبدايه")
    if p == 61: return stylize(f"المندوب وصل العنوان لكن ما يبي يطلع لين الباب")
    if p == 62: return stylize(f"وصل الطلب ناقص لان المندوب فتحه بنفسه")
    if p == 63: return stylize(f"مندوب التوصيل يلغي الطلب بدون سبب بعد قبوله")
    return stylize(f"التوصيل اخذ {time} والمندوب {issue} ووصل الطلب {temp}")


# ============================================================
# AMBIANCE  (الجو والمكان) — 80+ templates
# ============================================================
# This is the biggest priority — currently weakest at 42% F1, only 12 templates.

AMBIANCE_PARTS = [
    "الكراسي", "الديكور", "الاضاءه", "التكييف", "المقاعد", "الطاولات",
    "الموسيقى", "المساحه", "التهويه", "الجلسه", "موقف السيارات", "الازدحام",
    "الالوان", "الحراره", "تصميم المكان", "الحمامات", "الواجهه",
    "الصاله", "المدخل", "الطاوله", "الكنبات", "الديكورات", "الانارة",
    "السقف", "الارضيات", "الزوايا", "الجلسات الخارجيه", "التراس",
    "الحديقه", "الدرج", "الباب", "الفناء", "الشرفه",
]
AMBIANCE_ADJ = [
    "سيئه", "قديمه", "مزعجه", "ضعيفه", "غير مريحه", "متهالكه", "متسخه",
    "مكتظه", "صغيره", "مزدحمه", "مهتزه", "خانقه", "غير مناسبه",
    "ضيقه", "سيئه التهويه", "غير نظيفه", "بايخه", "كئيبه", "مظلمه",
    "ضعيفه الانارة", "حاره", "بارده", "خايسه", "زفت",
]
AMBIANCE_ISSUE = [
    "لا تناسب نوع المطعم", "تحتاج تجديد", "تتعب الزبون", "لا توحي بالراحه",
    "غير لائقه بالسعر", "محبطه", "تشعرني بالضيق", "لا تستحق الزياره",
    "تفسد التجربه كامله", "غير مقبوله", "لا تليق بمطعم بهذا الاسم",
    "تخلي الزبون يستعجل ويطلع", "تخلي الجلسه مش مريحه", "تذكرني ما اعود",
    "تشعرني بالغثيان", "تخنقني",
]

NOISE_TYPES = [
    "موسيقى عاليه", "موسيقى مزعجه", "صوت اطفال", "ضجيج", "صوت طناجر من المطبخ",
    "زحمه عند المدخل", "صوت تليفزيون عالي", "محادثات بصوت عالي",
]

PLACE_PROBLEMS = [
    "ريحه طبخ خانقه", "ريحه دخان", "ريحه زيت قلي", "حر شديد", "برد شديد",
    "الزحمه ما تتحمل", "كراسي ما تتحرك", "طاولات صغيره ومتلاصقه",
    "ما فيه موقف سيارات", "ما فيه قسم عائلات", "اسعار عاليه على هذا الديكور",
]


def t_ambiance():
    part = random.choice(AMBIANCE_PARTS)
    part2 = random.choice(AMBIANCE_PARTS)
    adj = random.choice(AMBIANCE_ADJ)
    adj2 = random.choice(AMBIANCE_ADJ)
    issue = random.choice(AMBIANCE_ISSUE)
    noise = random.choice(NOISE_TYPES)
    problem = random.choice(PLACE_PROBLEMS)
    intens = random.choice(INTENSIFIERS)

    p = random.randint(0, 79)

    # ---- short fragments (real reviews are often very short) ----
    if p == 0: return stylize(f"المكان زفت")
    if p == 1: return stylize(f"ديكور قديم")
    if p == 2: return stylize(f"الجو مو حلو")
    if p == 3: return stylize(f"المكان ضيق")
    if p == 4: return stylize(f"موقف سيارات صفر")
    if p == 5: return stylize(f"كراسي مو مريحه")
    if p == 6: return stylize(f"المطعم مزدحم")
    if p == 7: return stylize(f"اضاءه ضعيفه")
    if p == 8: return stylize(f"الجلسه مو مريحه")
    if p == 9: return stylize(f"حر داخل المطعم")
    if p == 10: return stylize(f"المكان متسخ")
    if p == 11: return stylize(f"تكييف ضعيف")
    if p == 12: return stylize(f"موسيقى عاليه")
    if p == 13: return stylize(f"ريحه دخان")
    if p == 14: return stylize(f"الديكور قديم جدا")

    # ---- medium dialect ----
    if p == 15: return stylize(f"{part} {adj} و{part2} {adj2} المكان {issue}")
    if p == 16: return stylize(f"{part} في المطعم {adj} جدا و{issue}")
    if p == 17: return stylize(f"المكان فيه مشاكل كثيره {part} {adj} و{part2} {adj2}")
    if p == 18: return stylize(f"الديكور قديم و{part} {adj} الجو {issue}")
    if p == 19: return stylize(f"{part} كانت {adj} طوال الوقت لا توجد راحه ابدا")
    if p == 20: return stylize(f"عند الدخول لاحظت ان {part} {adj} و{part2} كذلك")
    if p == 21: return stylize(f"المطعم يحتاج تطوير {part} {adj} والاجواء {issue}")
    if p == 22: return stylize(f"تجربه سيئه من ناحيه الديكور {part} {adj} {issue}")
    if p == 23: return stylize(f"الجو العام للمطعم سيء {part} {adj} والصاله ضيقه")
    if p == 24: return stylize(f"{part} {adj} مما جعل الجلسه غير ممتعه ابدا")
    if p == 25: return stylize(f"الاجواء {issue} بسبب ان {part} {adj}")

    # ---- noise / smell focused ----
    if p == 26: return stylize(f"{noise} طوال الوقت ما تقدر تجلس")
    if p == 27: return stylize(f"{noise} كانت الكارثه الكبرى")
    if p == 28: return stylize(f"المكان فيه {noise} ما يخليك تستمتع")
    if p == 29: return stylize(f"{problem} داخل المطعم لا تطاق")
    if p == 30: return stylize(f"ريحه الزيت في المطعم خانقه ما اقدر اتنفس")
    if p == 31: return stylize(f"كل ما تطلع من المطعم ملابسك ريحتها زفت")
    if p == 32: return stylize(f"الدخان في المطعم يخنق ما فيه شفط هواء")
    if p == 33: return stylize(f"{noise} و{problem} تجربه فاشله")

    # ---- temperature / AC ----
    if p == 34: return stylize(f"التكييف ضعيف ما يبرد المكان حر مرره")
    if p == 35: return stylize(f"المطعم بارد جدا التكييف على اخره")
    if p == 36: return stylize(f"الحراره داخل المطعم ما تتحمل")
    if p == 37: return stylize(f"تكييف معطل في يوم حار جدا")

    # ---- comparison with expectations / value ----
    if p == 38: return stylize(f"اسعارهم عاليه و{part} {adj} ما يستاهلون")
    if p == 39: return stylize(f"الاسعار تطلب ديكور افضل {part} {adj}")
    if p == 40: return stylize(f"المطعم غالي والديكور رخيص")
    if p == 41: return stylize(f"تدفع كثير لكن المكان {adj}")
    if p == 42: return stylize(f"المطعم باسمه فقط الديكور قديم والاجواء سيئه")

    # ---- crowding / parking / family ----
    if p == 43: return stylize(f"الزحمه عند الدخول ما تتحمل ولا فيه قسم عائلات")
    if p == 44: return stylize(f"ما فيه موقف للسيارات اضطر اوقف بعيد")
    if p == 45: return stylize(f"المكان دايم مزدحم ما تلقى طاوله فاضيه")
    if p == 46: return stylize(f"المطعم صغير وما يكفي العدد")
    if p == 47: return stylize(f"قسم العائلات صغير مرره وكراسي قليله")

    # ---- bathrooms / cleanliness of place ----
    if p == 48: return stylize(f"الحمامات نظافتها سيئه جدا")
    if p == 49: return stylize(f"الارضيات متسخه وما فيه احد ينظف")
    if p == 50: return stylize(f"الطاولات لزجه ما مسحوها")
    if p == 51: return stylize(f"المكان شكله مو نظيف من بره وداخل")

    # ---- dialect emotional ----
    if p == 52: return stylize(f"يا اخي المكان زفت ما اعود")
    if p == 53: return stylize(f"والله ديكور قديم وما يستاهل")
    if p == 54: return stylize(f"بصراحه المكان مو على المستوى المتوقع")
    if p == 55: return stylize(f"شوفوا يا جماعه الجو في هذا المطعم سيء جدا")
    if p == 56: return stylize(f"للاسف المكان كئيب الديكور قديم والاضاءه ضعيفه")
    if p == 57: return stylize(f"تجربه احباط الديكور سيء والمكان مزدحم")

    # ---- specific real-style complaints ----
    if p == 58: return stylize(f"الديكور عادي جدا لا يميزه شي")
    if p == 59: return stylize(f"الصراحه ديكور المطعم قديم يحتاج تجديد")
    if p == 60: return stylize(f"كنبات قديمه ومتسخه طاولات مهتزه")
    if p == 61: return stylize(f"الجلسات الخارجيه مكشوفه على الشارع مزعج")
    if p == 62: return stylize(f"المطعم يفتقر للخصوصيه ما فيه فواصل")
    if p == 63: return stylize(f"الديكور تقليدي مكرر في كل المطاعم")
    if p == 64: return stylize(f"اضاءه كافيه لكن الالوان كئيبه")
    if p == 65: return stylize(f"تصميم سيء ما يلائم نوع الطعام")

    # ---- multi-aspect complaints ----
    if p == 66: return stylize(f"الديكور قديم {part} {adj} و{noise}")
    if p == 67: return stylize(f"المكان كله مشاكل {part} {adj} {part2} {adj2} و{problem}")
    if p == 68: return stylize(f"المطعم سيء الجو {issue} والديكور قديم")
    if p == 69: return stylize(f"بدخول المطعم تشعر بالضيق {part} {adj} وريحه خانقه")

    # ---- code-switched ----
    if p == 70: return stylize(f"الديكور interior سيء مرره ما يستاهل")
    if p == 71: return stylize(f"AC ضعيف والمكان حار")
    if p == 72: return stylize(f"WiFi ضعيف والديكور سيء")
    if p == 73: return stylize(f"المطعم decor قديم ويحتاج تجديد كامل")

    # ---- emotional verdict ----
    if p == 74: return stylize(f"المكان غير لائق ما اعود")
    if p == 75: return stylize(f"تجربه سيئه بسبب الديكور والاجواء")
    if p == 76: return stylize(f"ابحثوا عن مكان غيره الديكور سيء")
    if p == 77: return stylize(f"ما انصح بالمكان الجو غير مريح ابدا")
    if p == 78: return stylize(f"الديكور والاجواء يخلونك تطلع بسرعه")
    return stylize(f"المكان غير لائق {part} {adj} و{part2} {adj2}")


# ============================================================
# GENERAL  (عامة) — 50+ templates
# ============================================================

GENERAL_NEG = [
    "تجربه سيئه", "خيبه امل", "تجربه محبطه", "غير راضي عن الزياره",
    "ما اعجبتني التجربه", "تجربه دون المستوى", "زياره فاشله",
    "تجربه ما تستاهل", "تجربه عاديه جدا", "تجربه باهته",
    "تجربه احباط", "زياره مخيبه", "تجربه ضعيفه",
]
GENERAL_RESTO = [
    "هذا المطعم", "المكان", "هذا الفرع", "هذه السلسله", "المطعم بشكل عام",
    "هذا النوع من المطاعم", "المحل", "هذا المقهى", "المطعم", "الفرع هذا",
    "هذي السلسله",
]
GENERAL_REASON = [
    "لاسباب كثيره", "لعدم اهتمامهم بالعميل", "بسبب تراجع المستوى",
    "لانهم ما يحترمون وقت الزبون", "لان الجوده متدنيه عموما",
    "لان المستوى ما يستاهل", "لان السعر ما يتناسب مع الخدمه",
    "بسبب سوء التنظيم العام", "لان الاداره غير محترفه",
    "بسبب اللامبالاه", "لانهم تغيروا للاسوا",
]
GENERAL_VERDICT = [
    "لن اعود مره ثانيه", "لا انصح به", "لن اكرر التجربه",
    "ابحثوا عن مكان افضل", "خياراتكم خارجه افضل بكثير",
    "ما عجبني ابدا", "تجربتي كانت اقل من المتوقع", "الافضل الذهاب لمكان ثاني",
    "ما يستاهل الزياره", "ما اعود حتى لو دفعوا لي",
]


def t_general():
    neg = random.choice(GENERAL_NEG)
    resto = random.choice(GENERAL_RESTO)
    reason = random.choice(GENERAL_REASON)
    verdict = random.choice(GENERAL_VERDICT)
    intens = random.choice(INTENSIFIERS)

    p = random.randint(0, 49)

    # ---- short fragments ----
    if p == 0: return stylize(f"تجربه سيئه")
    if p == 1: return stylize(f"ما اعود")
    if p == 2: return stylize(f"ما يستاهل")
    if p == 3: return stylize(f"خيبه امل")
    if p == 4: return stylize(f"تجربه فاشله")
    if p == 5: return stylize(f"للاسف")
    if p == 6: return stylize(f"محبط")
    if p == 7: return stylize(f"ما انصح به")
    if p == 8: return stylize(f"تعبني المطعم")
    if p == 9: return stylize(f"الصراحه ما يستاهل")

    # ---- medium ----
    if p == 10: return stylize(f"{neg} مع {resto} {reason}")
    if p == 11: return stylize(f"{neg} {verdict}")
    if p == 12: return stylize(f"{resto} {reason} و{verdict}")
    if p == 13: return stylize(f"السمعه افضل من الواقع {neg} و{verdict}")
    if p == 14: return stylize(f"{resto} ما اقنعني {reason}")
    if p == 15: return stylize(f"بعد عده زيارات {neg} {verdict}")
    if p == 16: return stylize(f"توقعاتي كانت اعلى {resto} {reason}")
    if p == 17: return stylize(f"بشكل عام {neg} مع {resto} {verdict}")
    if p == 18: return stylize(f"{neg} كل شي كان متوسط {reason}")
    if p == 19: return stylize(f"{verdict} {resto} {reason}")

    # ---- specific real-style ----
    if p == 20: return stylize(f"كل شي فيه سيء")
    if p == 21: return stylize(f"المطعم نزل مستواه كثير")
    if p == 22: return stylize(f"تراجع كبير في الجوده والخدمه")
    if p == 23: return stylize(f"المطعم ما عاد كما كان")
    if p == 24: return stylize(f"كانت بدايته قويه لكن انحدر مستواه")
    if p == 25: return stylize(f"اول مره تجربه ولا اخر مره")
    if p == 26: return stylize(f"ما توقعت يكون بهذا المستوى")
    if p == 27: return stylize(f"كل ما زرت المطعم اخرج محبط")
    if p == 28: return stylize(f"بعد كم زياره قررت ما اعود")
    if p == 29: return stylize(f"الادراه ما تهتم بالعميل ابدا")

    # ---- dialect emotional ----
    if p == 30: return stylize(f"يا اخي تجربه فاشله ما تستاهل")
    if p == 31: return stylize(f"والله ما يستحق الزياره")
    if p == 32: return stylize(f"بصراحه تجربه ما اعود لها")
    if p == 33: return stylize(f"للاسف خيبه امل كبيره")
    if p == 34: return stylize(f"شوفوا يا جماعه ما يستاهل")
    if p == 35: return stylize(f"يا جماعه الخير ما تروحون")

    # ---- with verdict ----
    if p == 36: return stylize(f"ما اعود حتى لو دفعوا لي")
    if p == 37: return stylize(f"ابحثوا عن مكان غيره")
    if p == 38: return stylize(f"خلاص شطبت المطعم من قائمتي")
    if p == 39: return stylize(f"اخر زياره لي للمطعم وما اعود")
    if p == 40: return stylize(f"تركت المطعم خلاص لين يحسنون")

    # ---- code-switched ----
    if p == 41: return stylize(f"المطعم overall سيء")
    if p == 42: return stylize(f"experience مخيبه للامل")
    if p == 43: return stylize(f"value مو مناسب")

    # ---- vague but real-sounding ----
    if p == 44: return stylize(f"عادي جدا ما فيه شي مميز")
    if p == 45: return stylize(f"المطعم ولا الاوسط")
    if p == 46: return stylize(f"كل شي عادي تجربه عاديه")
    if p == 47: return stylize(f"ما عجبني شي بصراحه")
    if p == 48: return stylize(f"تجربه باهته ما تتذكر")
    return stylize(f"{neg} {resto} {verdict}")


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
        print(f"  added {added} rows (in {attempts} attempts)")

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
