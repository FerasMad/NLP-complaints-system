"""Generate contrastive-pair synthetic ambience training data.

Produces a CSV of hand-crafted Saudi/Gulf complaint pairs where each pair
shares a risky word (بارد, حار, قديم, ريحة, زيت, etc.) but one row is
ambience and the other is a non-ambience hard-negative. The model learns
the boundary from the contrast.

Why contrastive pairs and not free-form generation: the v3 ambience class
died from over-broad synthetic templates that taught the model to fire on
any "place + adjective" pattern. Contrastive pairs explicitly anchor the
boundary by giving the model both sides of every confusable case.

Pairs are organized by ambience subtype so coverage gaps are visible.
Targets ≥10 pairs per subtype and ≥120 pairs total.

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
from pathlib import Path


# Format: (ambience_text, ambience_subtype, hard_neg_text, hard_neg_category, risk_keyword, notes)
PairTuple = tuple[str, str, str, str, str, str]


# ---------------------------------------------------------------------------
# 1. temperature_ac — heat / cold / AC vs food temperature
# ---------------------------------------------------------------------------
TEMPERATURE_AC: list[PairTuple] = [
    ("المكان بارد والمكيف قوي ما نقدر نقعد", "temperature_ac",
     "الاكل بارد لما وصل ما عجبني الطعم", "جودة الطعام", "بارد", "place temp vs food temp"),
    ("المطعم حار والمكيف ما يبرد جلسنا نتعرق", "temperature_ac",
     "الشاورما حارة بزياده ما اقدر اكلها", "جودة الطعام", "حار", "place temp vs food spice"),
    ("التكييف ضعيف والجو خانق", "temperature_ac",
     "القهوة باردة ما لها طعم ابدا", "جودة الطعام", "بارد", "AC vs cold drink"),
    ("المكيف بارد بزياده يثلج", "temperature_ac",
     "الايس كريم ذايب وحامض الطعم", "جودة الطعام", "بارد", "AC over-cold vs melted dessert"),
    ("الفرع حار جدا الجو ما يطاق", "temperature_ac",
     "البرجر مقلي زياده ومحروق من برا", "جودة الطعام", "حار", "place hot vs food over-fried"),
    ("المكيف خربان ودخلنا في حر فظيع", "temperature_ac",
     "الشوربه طلعت بارده وما لها طعم", "جودة الطعام", "بارد", "broken AC vs cold soup"),
    ("الجو في المطعم خانق ومالقينا تهويه", "temperature_ac",
     "الطلب جاء خانق بالبهارات وما اقدر اكله", "جودة الطعام", "خانق", "stuffy place vs over-spiced food"),
    ("المكيف يطلع هوا ساخن وحار", "temperature_ac",
     "الكبسه طلعت حاره بزياده والفلفل قوي", "جودة الطعام", "حار", "broken AC blowing hot vs spicy rice"),
    ("التهوية سيئه والمكان رطوبه", "temperature_ac",
     "الخبز فيه رطوبه وكأنه قديم", "جودة الطعام", "رطوبه", "humid place vs stale bread"),
    ("المكان مثل الثلاجه من برد المكيف", "temperature_ac",
     "الكوكا طلعت ثلج ما اقدر اشربها", "جودة الطعام", "ثلج", "freezing room vs frozen drink"),
    ("جلسنا في كنبات حر والمكيف بعيد", "temperature_ac",
     "البطاطس وصلت حر زياده يحرق اللسان", "جودة الطعام", "حر", "hot seats vs scalding fries"),
    ("المكيف يصقع والاطفال صار يبردون", "temperature_ac",
     "الشاي صقعان ما اقدر اشربه", "جودة الطعام", "يصقع", "freezing AC vs cold tea"),
]


# ---------------------------------------------------------------------------
# 2. seating_comfort — chairs/sofas/tables comfort vs hygiene/staff
# ---------------------------------------------------------------------------
SEATING_COMFORT: list[PairTuple] = [
    ("الكراسي قاسية على الظهر ما اقدر اقعد", "seating_comfort",
     "الكراسي عليها بقايا طعام من اللي قبلنا", "النظافة", "الكراسي", "uncomfortable chairs vs dirty chairs"),
    ("الطاولات صغيرة ما تكفي الطلب", "seating_comfort",
     "الطاولة وسخه ومالها بصمة تنظيف", "النظافة", "الطاولة", "small tables vs uncleaned tables"),
    ("الكنب قاسي وما تشعر بأي راحه", "seating_comfort",
     "الكنب وسخ عليه بقع وما تنظف", "النظافة", "الكنب", "hard sofa vs dirty sofa"),
    ("ما فيه مكان نقعد كل الجلسات محجوزه", "seating_comfort",
     "الجلسه طويله جدا ما حد جا يخدمنا", "خدمة الموظفين", "الجلسه", "no available seats vs slow service"),
    ("الكنب صغير وثلاثه ما يقعدون فيه", "seating_comfort",
     "الكاشير ما رحب فينا والجلسه طويله", "خدمة الموظفين", "الجلسه", "tight sofa vs unwelcoming staff"),
    ("الجلسات الارضيه ما فيها وسائد كافيه", "seating_comfort",
     "الجلسه استمرت ساعتين انتظار", "وقت الانتظار", "الجلسه", "uncushioned floor seats vs long wait"),
    ("القعدات ضيقه ولا اقدر امد رجلي", "seating_comfort",
     "الموظفين قعدوا يسولفون وما يخدمون", "خدمة الموظفين", "قعدوا", "tight seating vs idle staff"),
    ("الكرسي يهتز كل ما اتحرك", "seating_comfort",
     "الموظف يهتز كل ما تكلمه ما عنده تركيز", "خدمة الموظفين", "يهتز", "wobbly chair vs nervous staff"),
    ("الطاوله كبيره بزياده على المجموعه الصغيره", "seating_comfort",
     "الطاوله المحجوزه ما تطابق العدد المسجل", "خدمة الموظفين", "الطاوله", "wrong-size table vs reservation issue"),
    ("جلسات العوائل قليلة وكلها مشغوله", "seating_comfort",
     "الموظفين ما يفهمون نحجز للعوائل", "خدمة الموظفين", "العوائل", "few family seats vs staff confusion"),
    ("البارتشن مكسور وما يفصل بينا", "seating_comfort",
     "البارتشن المعدني وقع لما لمسه الموظف بالخطأ", "خدمة الموظفين", "البارتشن", "broken partition vs staff accident"),
]


# ---------------------------------------------------------------------------
# 3. space_crowding — crowded place vs related crowds
# ---------------------------------------------------------------------------
SPACE_CROWDING: list[PairTuple] = [
    ("المكان زحمه موت ما لقينا مكان نقعد", "space_crowding",
     "الموظفين ما يعرفون يخدمون من الزحمه", "خدمة الموظفين", "زحمه", "crowding vs slow-staff-blamed-on-crowd"),
    ("الجلسات متلاصقه فينا ما فيه مساحه", "space_crowding",
     "الفاتوره غاليه على الجلسه القصيرة", "السعر والقيمة", "الجلسات", "tight seating vs price"),
    ("ضيق وما تقدر تتنفس من الزحمة", "space_crowding",
     "السعر مبالغ فيه على وجبه ضيقه", "السعر والقيمة", "ضيق", "tight space vs portion-size complaint"),
    ("المكان مزدحم ما تقدر تمشي بين الطاولات", "space_crowding",
     "الطلب مزدحم بالاضافات اللي ما طلبتها", "دقة الطلب", "مزدحم", "packed place vs over-stuffed order"),
    ("ما فيه مساحه شخصيه كأنك في حافله", "space_crowding",
     "الفاتوره ما فيها مساحه لتفصيل الاصناف", "السعر والقيمة", "مساحه", "no personal space vs unclear bill"),
    ("كتمه فظيعه من الزحام داخل المطعم", "space_crowding",
     "الكاشير حسبها كتمه ما رد علي", "خدمة الموظفين", "كتمه", "stuffy crowd vs unresponsive cashier"),
    ("لاصقين فينا وما اقدر اتنفس", "space_crowding",
     "الموظف لاصق فينا ما يفارقنا", "خدمة الموظفين", "لاصقين", "crowded patrons vs intrusive staff"),
    ("مصفوطين مع بعض ما تقدر تتحرك", "space_crowding",
     "الطلبات مصفوطه فوق بعض في الكيس", "التوصيل", "مصفوطين", "packed in vs packed delivery bag"),
    ("ما فيه مجال نمشي حتى لدورة المياه", "space_crowding",
     "ما فيه مجال للتعديل بالطلب بعد ارساله", "خدمة الموظفين", "مجال", "no walking room vs no order edits"),
    ("زحمة بزياده وما حسبوا حساب العدد", "space_crowding",
     "الكاشير حسب طلب اللي قبلي معاي بالزحمه", "دقة الطلب", "حسبوا", "crowding vs miscounted bill"),
]


# ---------------------------------------------------------------------------
# 4. noise_music — sound problems vs noisy staff
# ---------------------------------------------------------------------------
NOISE_MUSIC: list[PairTuple] = [
    ("الموسيقى صوتها عالي ما نقدر نسولف", "noise_music",
     "الموظف صوته عالي ومزعج وقت الطلب", "خدمة الموظفين", "عالي", "loud music vs loud staff"),
    ("الاطفال يصرخون في كل المكان جو ازعاج", "noise_music",
     "الموظف ازعجني بالاسئله الكثيرة", "خدمة الموظفين", "ازعاج", "child noise vs annoying staff"),
    ("الدي جي يفتح الصوت لين تطق ودانك", "noise_music",
     "النادل صوته عالي وما يحترم", "خدمة الموظفين", "عالي", "DJ noise vs rude loud staff"),
    ("الصوت في المطعم مزعج ما نسمع بعض", "noise_music",
     "الطلب وصل مزعج بالسبام والاضافات الغلط", "دقة الطلب", "مزعج", "loud place vs spam-stuffed order"),
    ("الاغاني العربيه عاليه بزياده", "noise_music",
     "الاغاني الموسيقيه بالخلفيه طبيعيه بس الموظف عالي", "خدمة الموظفين", "الاغاني", "loud songs vs loud staff over music"),
    ("صوت المعدات في المطبخ يسمع لين عندنا", "noise_music",
     "الموظفين في المطبخ يصيحون على بعض", "خدمة الموظفين", "يصيحون", "kitchen equipment noise vs kitchen-staff yelling"),
    ("الصراخ من الاطفال خرب الجو علينا", "noise_music",
     "الموظف صرخ علي لما طلبت تعديل الطلب", "خدمة الموظفين", "صراخ", "child screaming vs staff yelling"),
    ("الصوت العالي خلانا ما نسمع طلباتنا للنادل", "noise_music",
     "النادل ما يسمع لما نكلمه يقولك ها كل مره", "خدمة الموظفين", "نسمع", "noise made staff inaudible vs deaf staff"),
    ("بزارين لاعبين بالكوره داخل المطعم وصوتهم عالي", "noise_music",
     "البزارين بالطلب احتاجوا ميني برجر مفقود", "دقة الطلب", "بزارين", "noisy kids vs missing kids meal"),
    ("الصوت العالي خلى راسي يدور", "noise_music",
     "الفاتوره العاليه خلت راسي يدور", "السعر والقيمة", "العالي", "loud noise dizzying vs high bill dizzying"),
]


# ---------------------------------------------------------------------------
# 5. lighting — lights too bright/dim vs other "bright/dim"
# ---------------------------------------------------------------------------
LIGHTING: list[PairTuple] = [
    ("الاضاءه قويه على العين بزياده", "lighting",
     "السعر قوي على وجبة بسيطه", "السعر والقيمة", "قوي", "bright lights vs steep price"),
    ("المكان مظلم وما اقدر اشوف المنيو", "lighting",
     "المنيو قديم وفيه اطباق مش موجوده", "خدمة الموظفين", "قديم", "dim lights vs outdated menu"),
    ("النور خافت بزياده تحس انك في كهف", "lighting",
     "الطلب وصل خافت الطعم وبدون نكهه", "جودة الطعام", "خافت", "weak light vs weak-flavored food"),
    ("الكشافات في عيوننا مباشره مزعجه", "lighting",
     "الكاشير لمعت عيونه لما شاف الفاتوره", "خدمة الموظفين", "لمعت", "spotlights in eyes vs surprised cashier"),
    ("اللمبات نص شغاله نص مطفيه", "lighting",
     "الويتر طفي اللمبه على الكيكه قبل ما يجي", "خدمة الموظفين", "اللمبه", "broken room bulbs vs cake-candle blown out by waiter"),
    ("النور قوي مره يحرق العين لما تجلس طويل", "lighting",
     "البرجر طلع نور لون من القلي القوي", "جودة الطعام", "نور", "harsh light vs over-fried food"),
    ("ظلام دامس ولا تشوف ايش تاكل", "lighting",
     "الطعام ظلام بسبب التتبيل المعتم", "جودة الطعام", "ظلام", "dark room vs dark-colored food"),
    ("الاضاءه ملونه وملحوظه فيها صداع", "lighting",
     "الصلصه ملونه بشكل غريب يخوف", "جودة الطعام", "ملونه", "weird-colored lighting vs weird-colored sauce"),
    ("الشمعه على الطاوله ضعيفه مكنش نقدر نقرا", "lighting",
     "الشمعه طلعت من الكيكه قبل ما يجي الويتر", "خدمة الموظفين", "الشمعه", "weak candle vs unlit cake-candle"),
]


# ---------------------------------------------------------------------------
# 6. smell — environmental odors vs food smell
# ---------------------------------------------------------------------------
SMELL: list[PairTuple] = [
    ("ريحة الزيت ماسكة في المكان طلعت ريحتي", "smell",
     "البطاطس مليانه زيت وحامضه", "جودة الطعام", "زيت", "env oil smell vs greasy food"),
    ("ريحة الدخان تخنق ما فيه تهوية", "smell",
     "الاكل ريحته غريبة شككتني فيه", "جودة الطعام", "ريحة", "smoke in air vs food smell"),
    ("ريحة المجاري طالعة من الحمام", "smell",
     "ريحة الزيت في الصلصة بايخه", "جودة الطعام", "ريحة", "sewage smell vs food smell"),
    ("الريحة تلصق في الملابس بعد ما اطلع", "smell",
     "ريحة اللحمه طلعت غريبة ورديتها", "جودة الطعام", "ريحة", "lingering env smell vs food smell"),
    ("ريحة الشيشه من الجلسات الخارجيه تدخل علينا", "smell",
     "ريحة الشاورما اللي طلبتها كانت غريبه", "جودة الطعام", "ريحة", "shisha smell vs grilled-meat smell"),
    ("ريحة المعسل تخنقنا والقسم الداخلي ما يفصل", "smell",
     "ريحة المعسل في المنيو مغريه بس الطعم سيء", "جودة الطعام", "المعسل", "shisha smell vs flavored food"),
    ("ريحة المطبخ تطلع علينا لين الطاوله", "smell",
     "ريحة المطبخ في الاكل قويه يبي تخفيف", "جودة الطعام", "المطبخ", "kitchen odor leak vs food too kitchen-y"),
    ("الريحة العفنه في الزاويه قبيحه", "smell",
     "الاكل ريحته عفنه طلب جديد لزم", "جودة الطعام", "عفنه", "moldy corner smell vs spoiled food"),
    ("ريحة العرق منتشره في القسم العائلي", "smell",
     "اللحمه فيها ريحة عرق جايه من سوء التحضير", "جودة الطعام", "عرق", "BO smell in place vs off meat"),
    ("طلعت ريحتي مثل المطعم اول ما طلعت", "smell",
     "ريحتي طلعت من الطلب لما فتحت الكيس", "جودة الطعام", "ريحتي", "place smell on me vs food smell on hand"),
    ("ريحة الزيت القديم تطلع من المقلاه", "smell",
     "اللحمه فيها ريحة زيت قديم بصراحه", "جودة الطعام", "زيت", "old oil smell in place vs old oil in food"),
]


# ---------------------------------------------------------------------------
# 7. decor_furniture — physical decor/state vs other "old/broken"
# ---------------------------------------------------------------------------
DECOR_FURNITURE: list[PairTuple] = [
    ("الديكور قديم والفرش مهلوك يبي تجديد", "decor_furniture",
     "الاكل قديم محسوس عليه ومش طازه", "جودة الطعام", "قديم", "old decor vs old food"),
    ("الطاولة تهتز كل ما تحط شي عليها", "decor_furniture",
     "الطاولة وسخه فيها بقع طعام من قبل", "النظافة", "الطاولة", "wobbly table vs dirty table"),
    ("الكنب مهلوك والاسفنج طالع", "decor_furniture",
     "الكنب وسخ عليه بقع وما تنظف", "النظافة", "الكنب", "worn sofa vs dirty sofa"),
    ("الجدران فيها بويه متشققه ومتهالكه", "decor_furniture",
     "الجدران فيها بقايا طعام تطير من المطبخ", "النظافة", "الجدران", "cracked walls vs splattered walls"),
    ("الديكور ما يناسب فخامة الاسعار", "decor_furniture",
     "الاسعار غاليه على ديكور بسيط", "السعر والقيمة", "الاسعار", "decor doesn't match price vs price too high for decor"),
    ("الارضيه فيها بلاط مكسور خطر للوقوع", "decor_furniture",
     "الارضيه دبقه من الزيت اللي مراق", "النظافة", "الارضيه", "broken floor vs greasy floor"),
    ("السيراميك متشقق ومن زمن لازم يتغير", "decor_furniture",
     "السيراميك على الطلب يبي تنظيف", "النظافة", "السيراميك", "old ceramic floor vs unwashed dishes"),
    ("الكوشن الموجود على الكراسي مهترئ ووسخ", "decor_furniture",
     "الكوشن الذي طلبته نسوه في الطلب", "دقة الطلب", "الكوشن", "worn cushion vs forgotten cushion add-on"),
    ("المخدات على الجلسة فاضيه وبدون حشو", "decor_furniture",
     "المخدات اللي طلبتها كاضافه ما وصلت مع الطلب", "دقة الطلب", "المخدات", "empty cushions vs missing cushion add-on"),
    ("الرخام على الطاوله متكسر من الحواف", "decor_furniture",
     "الرخام مذكور بالمنيو لتزيين الكيك بس مالقيناه", "دقة الطلب", "الرخام", "broken marble counter vs missing decoration"),
    ("الفرش قديم ولونه باهت من الاستخدام", "decor_furniture",
     "الفرش بالطلب طلع غير اللي اخترته", "دقة الطلب", "الفرش", "faded furniture vs wrong furniture-style food packaging"),
]


# ---------------------------------------------------------------------------
# 8. parking — parking access vs other "far"
# ---------------------------------------------------------------------------
PARKING: list[PairTuple] = [
    ("ما فيه مواقف اضطرينا نمشي من بعيد", "parking",
     "الطلب تأخر والمندوب تاه", "التوصيل", "بعيد", "no parking vs delivery-far"),
    ("المواقف بعيدة جدا والمدخل صعب", "parking",
     "المندوب وصل بعيد عن البيت ومانتبه", "التوصيل", "بعيد", "distant parking vs distant delivery"),
    ("الباركنج الخاص ممتلئ والشارع ضيق", "parking",
     "البركنج فيه طلب ناقص لما طلعنا", "دقة الطلب", "بركنج", "full parking vs missing item discovered at parking"),
    ("الموقف الوحيد ضيق ما يسع السياره", "parking",
     "الميزانيه ضيقه ما تكفي للوجبه العائليه", "السعر والقيمة", "ضيق", "narrow parking vs tight budget"),
    ("المدخل صعب الوصول مع الازدحام", "parking",
     "المدخل لقائمة الطعام صعب من الموقع الالكتروني", "التوصيل", "المدخل", "physical entrance vs app interface"),
    ("الطلعه من المطعم محصوره وضيقه", "parking",
     "الطلعه من المنيو سعرها مرتفع", "السعر والقيمة", "الطلعه", "exit ramp vs menu price spike"),
    ("الموقع صعب الوصول وما فيه باركنج قريب", "parking",
     "الموقع للطلب الالكتروني فيه مشكله", "التوصيل", "الموقع", "physical location vs app location"),
    ("وقفت بعيد مره عشان كل المواقف زحمه", "parking",
     "وقفت بعيد عن الكاشير عشان ما يسمعني", "خدمة الموظفين", "وقفت", "parked far vs stood far"),
    ("الباركنج مدفوع وما حد قال لنا", "parking",
     "الباركنج المدفوع ما يحسبوه على الفاتوره", "السعر والقيمة", "الباركنج", "paid parking surprise vs parking on bill"),
]


# ---------------------------------------------------------------------------
# 9. bathroom_facilities — broken/missing supplies vs hygiene
# ---------------------------------------------------------------------------
BATHROOM_FACILITIES: list[PairTuple] = [
    ("السيفون خربان وما فيه صابون اصلا", "bathroom_facilities",
     "الحمام وصخ ومقرف ما يستحمل", "النظافة", "الحمام", "broken bathroom vs dirty bathroom"),
    ("باب الحمام ما يقفل والمغسلة معطلة", "bathroom_facilities",
     "الحمام مليان مجاري وريحه تطلع", "النظافة", "الحمام", "broken bathroom hardware vs filthy bathroom"),
    ("ما فيه مناديل في الحمام ولا صابون", "bathroom_facilities",
     "الارضيه في الحمام مبلولة ووسخه", "النظافة", "الحمام", "missing supplies vs floor hygiene"),
    ("المغسلة ما يطلع منها مي", "bathroom_facilities",
     "المغسلة في المطبخ يصبون فيها كل شي", "النظافة", "المغسلة", "broken tap vs dirty kitchen sink"),
    ("الحمام بدون مرايا ولا اضاءه كافيه", "bathroom_facilities",
     "الحمام نتنه ريحه تخنق", "النظافة", "الحمام", "no mirror in bathroom vs stinking bathroom"),
    ("جهاز التجفيف معطل بعد غسل اليدين", "bathroom_facilities",
     "التجفيف للاكل قبل التغليف ما تم", "جودة الطعام", "التجفيف", "broken hand-dryer vs undried food packaging"),
    ("السيفون يطلع كل المي على الارض", "bathroom_facilities",
     "الارض في الحمام مليانه مي ووسخه", "النظافة", "الارض", "broken flush vs dirty wet floor"),
    ("دورات المياه الرجاليه مقفله بدون توضيح", "bathroom_facilities",
     "دورات المياه الرجاليه وسخه بشكل مبالغ", "النظافة", "دورات المياه", "men's room locked vs men's room dirty"),
    ("المراحيض الجوال معطل بعد كل استخدام", "bathroom_facilities",
     "المراحيض كلها وسخه بشكل لا يصدق", "النظافة", "المراحيض", "broken urinal vs all toilets dirty"),
    ("لا يوجد اشاره دخول للحمام للضيوف", "bathroom_facilities",
     "اشاره الموظفين للزبون قليلة الادب", "خدمة الموظفين", "اشاره", "no bathroom signage vs rude staff signaling"),
]


# ---------------------------------------------------------------------------
# 10. outdoor_view — terrace/view vs other "outside"
# ---------------------------------------------------------------------------
OUTDOOR_VIEW: list[PairTuple] = [
    ("الجلسة الخارجية مكشوفة على الشارع وجو مزعج", "outdoor_view",
     "الطلب وصل برا الكيس واتلطخ", "التوصيل", "برا", "outdoor seating vs delivery-outside-bag"),
    ("التراس ما فيه ظل والشمس قوية", "outdoor_view",
     "الشمس على الباب طلعت الايس كريم خربان", "جودة الطعام", "الشمس", "no outdoor shade vs sun-spoiled food"),
    ("الجلسة على الواجهه فيها ضوضاء سيارات", "outdoor_view",
     "الواجهه في المنيو مغريه بس الطعم مخيب", "جودة الطعام", "الواجهه", "noisy facade vs food cover misleading"),
    ("التراس مفتوح وفيه ذباب وحشرات", "outdoor_view",
     "الطلب فيه ذباب اشمأز تماما", "النظافة", "ذباب", "outdoor flies vs flies in food"),
    ("الجلسه برا مزعجه من زحمة الشارع", "outdoor_view",
     "الجلسه برا الموعد ما حد جا", "خدمة الموظفين", "برا", "outdoor noisy vs late seating"),
    ("الاطلاله على المدخل قبيحه ومش ممتعه", "outdoor_view",
     "الاطلاله على المنيو غير واضحه", "خدمة الموظفين", "الاطلاله", "ugly view vs unclear menu"),
    ("البلكونه ضيقه ولا تكفي اربعه افراد", "outdoor_view",
     "البلكونه التي كانت في الفنادق ابعد", "السعر والقيمة", "البلكونه", "small balcony vs balcony as upsell"),
    ("السطح ما فيه تكييف خارجي ولا ميني فان", "outdoor_view",
     "السطح فوق الطلب وقع منه السمسم", "جودة الطعام", "السطح", "rooftop no AC vs sesame on top"),
    ("التراس فيه عمال يدخنون قريب منا", "outdoor_view",
     "التراس بالموقع غير متاح للحجز اونلاين", "التوصيل", "التراس", "smokers on terrace vs terrace not bookable"),
    ("جلسة برا ما فيها واي فاي ولا اضائه", "outdoor_view",
     "جلسة برا تطلب باستيكي عشان التوصيل", "التوصيل", "جلسة برا", "outdoor no wifi vs delivery-outside packaging"),
]


# ---------------------------------------------------------------------------
# 11. privacy — privacy of the seating arrangement vs other "exposed"
# ---------------------------------------------------------------------------
PRIVACY: list[PairTuple] = [
    ("ما فيه خصوصية كل الناس تشوفك تاكل", "privacy",
     "الكاشير ما يحترم خصوصية اللي قبلك", "خدمة الموظفين", "خصوصية", "no privacy in seating vs no customer privacy"),
    ("قسم العوائل ضيق ولاصقين فينا", "privacy",
     "الموظف لاصق فينا ما تحس بالراحه", "خدمة الموظفين", "لاصقين", "crowded family section vs hovering staff"),
    ("ما فيه حواجز بين الجلسات وكل الناس تسمعك", "privacy",
     "ما فيه حواجز للموظفين فيدخلون عليك بدون استئذان", "خدمة الموظفين", "حواجز", "no partitions vs no staff barrier"),
    ("الجلسه مكشوفه على المطبخ ما فيها سرية", "privacy",
     "الجلسه مكشوفه للمندوب لما طلب الفاتوره", "خدمة الموظفين", "مكشوفه", "exposed to kitchen vs cashier sees order"),
    ("جنبنا ناس يتسمعون كل كلمه نقولها", "privacy",
     "جنبنا ناس استلموا طلبنا بالغلط", "دقة الطلب", "جنبنا ناس", "neighbors eavesdropping vs neighbors got our order"),
    ("الافراد ما يقدرون يجلسون بدون عوائل", "privacy",
     "الافراد ما يقدرون يحجزون اونلاين", "التوصيل", "الافراد", "singles section issue vs singles can't book online"),
    ("القسم العائلي يطل على قسم الافراد", "privacy",
     "القسم العائلي حسبوا اسعاره غلط في الفاتوره", "السعر والقيمة", "القسم العائلي", "family section exposed vs miscalculated family-section bill"),
    ("جلسات العوائل فاضيه بس ما يخلونا نقعد فيها", "privacy",
     "جلسات العوائل اسعارها اعلى من الافراد", "السعر والقيمة", "جلسات العوائل", "family seats restricted vs family seats pricier"),
    ("الستاره بين الجلسات شفافه ما تستر", "privacy",
     "الستاره الزخرفيه على الجدار مغبره", "النظافة", "الستاره", "see-through curtain vs dusty curtain"),
    ("ما فيه ساتر بين كل عائله وعائله", "privacy",
     "الساتر للنادل ضيق ما يقدر يخدم", "خدمة الموظفين", "الساتر", "no family partition vs partition blocks waiter"),
]


# ---------------------------------------------------------------------------
# 12. general_place_vibe — overall feel/mood vs related personality words
# ---------------------------------------------------------------------------
GENERAL_PLACE_VIBE: list[PairTuple] = [
    ("المكان كئيب وما يفتح النفس", "general_place_vibe",
     "الموظف كان بارد في التعامل", "خدمة الموظفين", "كئيب", "depressing place vs cold staff"),
    ("الجو ثقيل في المطعم ما يستحمل", "general_place_vibe",
     "الفاتوره ثقيله على الجيب", "السعر والقيمة", "ثقيل", "heavy vibe vs heavy bill"),
    ("الشكل قديم وكأنه من زمان والحمد لله", "general_place_vibe",
     "الاكل قديم وما يستاهل ابدا", "جودة الطعام", "قديم", "old-style place vs old food"),
    ("الفرع مقفل من الداخل ما يفتح النفس", "general_place_vibe",
     "الفرع مقفل اليوم وما حد رد على التلفون", "خدمة الموظفين", "مقفل", "stuffy interior vs branch closed"),
    ("الفايب في المكان غريب ما حسيت بالراحه", "general_place_vibe",
     "الفايب في المنيو ما يقنع لاختيار الطلب", "خدمة الموظفين", "الفايب", "weird vibe vs unconvincing menu"),
    ("المحل متهالك وداخل من زمن ما رمم", "general_place_vibe",
     "المحل متهالك في الفروع التانيه افضل", "السعر والقيمة", "متهالك", "rundown branch vs other branches better-value"),
    ("اللوكيشن سيء واحساس عام مزعج", "general_place_vibe",
     "اللوكيشن في الخريطه غلط وضعنا", "التوصيل", "اللوكيشن", "bad-feel location vs wrong map pin"),
    ("الاجواء داخل المطعم باله ومش جذابه", "general_place_vibe",
     "الاجواء العامه للمنيو ما ترغب الزبون", "خدمة الموظفين", "الاجواء", "stale interior vs unappealing menu vibe"),
    ("ما يفتح النفس ابدا الجلوس هنا طويل", "general_place_vibe",
     "ما يفتح النفس الطعم لما تذوقت اول لقمه", "جودة الطعام", "يفتح النفس", "place doesn't lift mood vs food doesn't"),
    ("منفس ما قاله الموقع ولكن داخل المطعم خانق", "general_place_vibe",
     "المنفس في الكيس مكسور والاكل بايخ", "التوصيل", "منفس", "place not airy vs broken delivery-bag vent"),
    ("شرح الجو فظيع لا تمشي يوم العطل", "general_place_vibe",
     "شرح الموظف للمنيو فظيع ما فهمنا شي", "خدمة الموظفين", "شرح", "place atmosphere bad vs bad menu explanation"),
    ("المكروف القديم في الجو ما تجاوبنا معه", "general_place_vibe",
     "المكروف في الكيك قديم وما متماسك", "جودة الطعام", "المكروف", "stale microwave-feel vs stale microwaved food"),
]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

ALL_PAIRS: list[PairTuple] = (
    TEMPERATURE_AC + SEATING_COMFORT + SPACE_CROWDING + NOISE_MUSIC +
    LIGHTING + SMELL + DECOR_FURNITURE + PARKING + BATHROOM_FACILITIES +
    OUTDOOR_VIEW + PRIVACY + GENERAL_PLACE_VIBE
)


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


def expand_pairs(pairs: list[PairTuple]) -> list[dict]:
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

    rows = expand_pairs(ALL_PAIRS)
    rows = [r for r in rows if r["quality_score"] >= args.min_quality]

    n_amb = sum(1 for r in rows if r["category"] == "الجو والمكان")
    n_neg = sum(1 for r in rows if r["is_hard_negative"])
    n_groups = len({r["contrast_group_id"] for r in rows})

    # Per-subtype coverage report — surfaces gaps before training
    per_subtype: dict[str, int] = {}
    for r in rows:
        if r["ambience_subtype"]:
            per_subtype[r["ambience_subtype"]] = per_subtype.get(r["ambience_subtype"], 0) + 1

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
    print("  per-subtype coverage:")
    for subtype in sorted(per_subtype):
        bar = "#" * per_subtype[subtype]
        print(f"    {subtype:<22} {per_subtype[subtype]:>3}  {bar}")
    print()
    print(f"All rows have source='ambience_synthetic' — train-only via the existing leakage gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
