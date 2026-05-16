"""Generate the 500+ row production test fixture for the 8-class Arabic complaints classifier.

Output: tests/fixtures/production_test_500.csv (UTF-8 with BOM, standard CSV).

Columns: id, text, expected_label, difficulty, attack_type, notes.

All rows are hand-written Saudi/Gulf-dialect realistic complaints that do NOT appear
in the existing 95K-row training dataset (verified via membership check after generation).

Categories (production 8-class):
  التوصيل, السعر والقيمة, النظافة, جودة الطعام,
  خدمة الموظفين, دقة الطلب, عامة, وقت الانتظار

Plus special expected_labels:
  multi          — multi-aspect rows where the dominant class is in notes
  abstain        — very-short, system should say "too short"
  no_complaint   — OOD / off-topic, system should not classify
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

# --- canonical category names (must match production model exactly) ---
DELIVERY = "التوصيل"
PRICE = "السعر والقيمة"
HYGIENE = "النظافة"
FOOD = "جودة الطعام"
STAFF = "خدمة الموظفين"
ACCURACY = "دقة الطلب"
GENERAL = "عامة"
WAIT = "وقت الانتظار"

# ---------------------------------------------------------------------------
# 1) التوصيل (Delivery) — 50 sentences
# Focus: courier, app, late drop-off, address issues, bag damage,
# bike crash, called wrong number, returned with food etc.
# ---------------------------------------------------------------------------
DELIVERY_ROWS: list[tuple[str, str, str]] = [
    # (text, difficulty, notes)
    ("المندوب رمى الكيس عند الباب وطنش", "easy", "courier dumped bag"),
    ("الاوردر تاخر بزياده والاكل وصل سايح", "easy", "late delivery + soggy food"),
    ("جاهز اعطاني وقت ١٥ دقيقه واستمر ٧٠ دقيقه", "easy", "estimated vs actual"),
    ("ما وصل لي اشعار وصول المندوب طلع بدون ما يدق", "medium", "no arrival notification"),
    ("الديليفري ابلغني ان طلبي راح لعنوان غير عنواني", "medium", "wrong-address delivery"),
    ("المندوب طلب مني انه اطلع له من الحوش", "medium", "courier refusing to come to door"),
    ("التطبيق سحب الفلوس وما اكد الطلب", "easy", "app charged without confirmation"),
    ("سحب فلوسي من البطاقه والطلب اتلغى لحاله", "medium", "auto-cancel after charge"),
    ("مندوب هنقرستيشن قعد يدور البيت نص ساعه", "easy", "courier got lost"),
    ("الديليفري وصل بدون كيس الزبده والمايونيز", "medium", "missing sides bag on delivery"),
    ("التغليف منفتح كله انسكب الصوص في الكيس", "easy", "spilled in transit"),
    ("المندوب باي قبل ما يسلمني الطلب", "easy", "courier left without handover"),
    ("الاكل وصلني بعد ساعه ونص وانا ساكن جنب الفرع", "easy", "very late from nearby branch"),
    ("ضايع المندوب في الحي وكل شويه يتصل يسالني", "medium", "lost driver calling repeatedly"),
    ("التطبيق يقول تم التوصيل وما وصلني شي", "easy", "false-completed delivery"),
    ("المندوب اعطى طلبي للجار وراح", "medium", "wrong recipient"),
    ("جا مندوب الكريم وطلب بقشيش بالعافيه", "medium", "courier demanding tip"),
    ("الديلفري قال ما اطلع للدور الثاني انزل خذ", "medium", "courier refusing stairs"),
    ("التطبيق رفض دفعي بالبطاقه وخلاني ادفع كاش", "medium", "card declined twice"),
    ("الكيس وصل ممزق من ركوب الموتر", "easy", "torn bag"),
    ("طلبي تأخر ساعتين بسبب المطر وما عوضوني", "medium", "weather delay + no comp"),
    ("جربت اطلب من مرسول وما لقيت اي مندوب متاح", "easy", "no couriers available"),
    ("التطبيق علق عند تأكيد الدفع وما طلع ايصال", "medium", "checkout hang"),
    ("الكيس مكتوب عليه اسم زبون ثاني", "medium", "labeled wrong customer"),
    ("المندوب جالس في السياره يكلم اهله وانا انتظر تحت", "medium", "courier idling on call"),
    ("التطبيق يطلب فيزا فقط ما يقبل مدى", "easy", "payment method restriction"),
    ("توصيل من جدة الى الرياض اخذ يومين", "easy", "cross-city excess"),
    ("اتصلت بخدمة العملاء قالوا المندوب فقد الاشاره", "medium", "support blames signal"),
    ("الطلب وصل وانا في مشوار راح للبيت فاضي", "medium", "delivery missed customer"),
    ("ما عجبني ان المندوب طلب مني ابوكسه على القطعه", "medium", "asked for cash for parking"),
    ("الطلب اتقسم على مندوبين والاكل وصل قسمين", "hard", "split delivery"),
    ("التطبيق يحدد العنوان غلط على الخريطه", "medium", "geocoder error"),
    ("جاء بدون كيس البلاستيك صب الاكل في الباب", "medium", "no bag handoff"),
    ("الاوبشن في التطبيق يعطي ٣٠ دقيقه وطلع ٩٠", "easy", "ETA underestimate"),
    ("ما رضي يعطيني الطلب الا ادفع كاش بسبب التطبيق", "medium", "cash-only forced"),
    ("سحب المبلغ مرتين والديليفري ما وصل", "easy", "double-charge no delivery"),
    ("المندوب جا بسياره خاصه بدون اي شعار", "medium", "unmarked vehicle"),
    ("التغليف فتحه احد قبل ما يوصلني", "hard", "tamper-evident concern"),
    ("الطلب ضايع لحد الحين والتطبيق يقول جاري التوصيل", "easy", "stuck in delivery state"),
    ("الديلفري حاول يفتح الكيس قدامي عشان يحسبها", "hard", "courier opening bag"),
    ("جربت احد توا الديلفري كل مره يجي بارد", "medium", "consistent cold delivery"),
    ("المندوب جايب كيس ثاني غير اللي حسبته انا", "hard", "wrong order entirely"),
    ("الطلب وصل من بعد ما اتصلت ٤ مرات بالدعم", "easy", "after support escalation"),
    ("التطبيق يجبرني اضيف بقشيش قبل ما اكمل الطلب", "medium", "forced tip"),
    ("الديلفري وقف في باب البنايه وما طلع لي", "easy", "ground-floor handoff demand"),
    ("التتبع متوقف ولا يتحرك من ربع ساعه", "medium", "frozen tracker"),
    ("ساعه وانا انتظر مندوب من فرع قريب من البيت", "easy", "wait from nearby branch"),
    ("سياسة التطبيق ما ترجع الفلوس الا بعد اسبوع", "medium", "refund policy delay"),
    ("الكيس وصل مبلل ماء كانه طاح في الشارع", "medium", "wet bag"),
    ("الديلفري طلب رقمي الشخصي عشان يكلمني بره التطبيق", "hard", "out-of-app contact request"),
]

# ---------------------------------------------------------------------------
# 2) السعر والقيمة (Price/value) — 50 sentences
# ---------------------------------------------------------------------------
PRICE_ROWS: list[tuple[str, str, str]] = [
    ("الوجبه ب٧٠ ريال وحجمها مايسد جوع طفل", "easy", "expensive vs portion"),
    ("سعر الزياده على الصوص ١٠ ريال شي ما يتصدق", "easy", "expensive addons"),
    ("الفاتوره طلعت ضعفين اللي توقعته", "easy", "bill doubled"),
    ("اضافة الجبنه وحدها ب١٥ ريال", "easy", "addon surcharge"),
    ("القيمه ما تستاهل المبلغ المدفوع ابدا", "easy", "value mismatch"),
    ("سعر القهوه ب٣٢ ريال وطعمها عادي", "easy", "overpriced coffee"),
    ("حسبوا علي رسوم خدمه بدون ما يخبروني", "medium", "hidden service charge"),
    ("الكوبون اللي في التطبيق ما اشتغل ودفعت كامل", "medium", "broken coupon"),
    ("مشروب صغير ب١٨ ريال يعني سرقه", "easy", "drink overpriced"),
    ("العرض اللي في الانستقرام مو هو نفسه في الفرع", "medium", "ad-vs-reality price"),
    ("اضافة الايس كريم تنده ب٢٠ ريال", "easy", "ice cream surcharge"),
    ("سعر التوصيل غير معقول ٢٥ ريال لمسافه ٣ كيلو", "medium", "delivery fee high"),
    ("دفعت ٢٠٠ على ٣ ساندوتشات وما شبعنا", "easy", "high total + low value"),
    ("الباقي ما اعطوني اياه قالوا ما عندنا فكه", "medium", "no change given"),
    ("اضافة الفلفل ب٥ ريال شي يضحك", "easy", "absurd small surcharge"),
    ("الفاتوره فيها بنود ما طلبتها", "medium", "phantom line items"),
    ("اللحم وحده ب٨٠ ريال ربع كيلو والباقي بصل", "medium", "expensive light entrée"),
    ("الخصم على التطبيق يطبق بعد الضريبه يعني صفر فايده", "hard", "discount-after-tax trick"),
    ("الاسعار في القائمه غير اللي يحسبوها في الكاشير", "medium", "menu vs cashier mismatch"),
    ("سعر السلطه ب٣٥ ريال على نص صحن خس", "easy", "small salad pricey"),
    ("اخر ميسد كم على الاكل ما تستاهل ابدا", "easy", "general value"),
    ("ضعف السعر بدون اي مبرر مقارنه بالفرع الثاني", "medium", "branch-vs-branch pricing"),
    ("اضافة فطر ٧ ريال على بيتزا اصلا غاليه", "easy", "stacked surcharges"),
    ("وجبة الاطفال ب٥٠ ريال وكل اللي فيها قطعتين", "easy", "kids meal overpriced"),
    ("نفس البرجر في فرع ثاني ب٢٥ هنا ب٤٥", "medium", "inconsistent pricing"),
    ("سعر العصير الطبيعي ب٢٨ يعني انت تستهبل", "easy", "fresh juice overpriced"),
    ("ضريبه القيمه طبقوها على وجبه مخفضه", "hard", "VAT on discounted price"),
    ("دفعت ١٢٠ على فطار سادس وقهوه", "easy", "expensive simple breakfast"),
    ("اعلانهم يقول ٢ ب ٥٠ وعند الكاشير ما يمشي العرض", "medium", "promo not honored"),
    ("القيمه الغذائيه قليله مقابل السعر العالي", "easy", "nutrition vs price"),
    ("سعر الحلى ب٤٥ ريال على قطعه ما تتجاوز اللقمتين", "easy", "dessert size vs price"),
    ("تحسبون التوصيل بسعر البنزين بدون ما تقولون", "hard", "fuel surcharge surprise"),
    ("اضطريت ادفع ضعف السعر عشان الجبن الزياده", "easy", "extra cheese gouge"),
    ("الفاتوره فيها مبلغ بقشيش مضاف بدون اذني", "hard", "auto-tip without consent"),
    ("شفت سعر الصلصه في الكاشير اعلى من المنيو ب٣ ريال", "medium", "menu typo or scam"),
    ("اشتركت في عروض البريميوم وما قدمت لي اي تخفيض", "medium", "premium membership useless"),
    ("القهوه التركيه ب٤٠ ريال شي مبالغ فيه", "easy", "turkish coffee overpriced"),
    ("اضافة الصلصه الاضافيه على البيتزا ب٧ ريال", "easy", "extra sauce price"),
    ("الفاتوره مكتوبه بالانجليزي وما اقدر اقراها", "medium", "bill format"),
    ("ميزانيتنا ما تكفي لاسعارهم اللي زادت بثلث", "easy", "price hike"),
    ("نقرشهم بحسبتهم القهوه ب٣٠ والمنيو ب٢٥", "medium", "explicit overcharge"),
    ("سعر اضافه واحده مايونيز ٤ ريال غير منطقي", "easy", "small absurd surcharge"),
    ("اشتريت كوبون من قروبون وما قبل في الفرع", "medium", "third-party voucher rejected"),
    ("القائمه ما فيها اسعار اصلا حتى توقعوني بمبلغ كبير", "medium", "no prices on menu"),
    ("الجلسه في القسم العائلي تكلف ٢٠٠ ريال اضافيه", "hard", "family section upcharge"),
    ("الفاتوره بدون تفصيل واطلبت تفاصيل ما اعطوني", "medium", "no itemized bill"),
    ("ضربوني بضريبة بقشيش ٢٠٪ بدون ما اطلب", "medium", "20% auto-gratuity"),
    ("اسعارهم تتغير كل اسبوع بدون اشعار", "medium", "fluctuating prices"),
    ("جربت اطبق كوبون والتطبيق رفع السعر بعدها", "hard", "coupon raises price"),
    ("الميمبرشب اللي اشتركت فيه ما يفعل العروض الموعوده", "medium", "membership broken promo"),
]

# ---------------------------------------------------------------------------
# 3) النظافة (Cleanliness) — 50 sentences
# ---------------------------------------------------------------------------
HYGIENE_ROWS: list[tuple[str, str, str]] = [
    ("لقيت شعره في وسط الرز", "easy", "hair in food"),
    ("الصحن فيه بقع من اكل سابق", "easy", "dish residue"),
    ("الكاسه على رفها لها بقع روج لشخص ثاني", "easy", "lipstick on glass"),
    ("الشوكه فيها قطع طعام من زبون قبلي", "easy", "fork residue"),
    ("لاحظت بقعة دم على المنشفه", "hard", "blood on napkin"),
    ("الارضيه دبقه ولزقت احذيتي فيها", "easy", "sticky floor"),
    ("الحمام رائحته كريهه ومافيه صابون", "easy", "bathroom smell + supplies"),
    ("شفت صرصور صغير يمشي على البلاط", "easy", "roach"),
    ("الطاوله ما تنظفت من الزبون اللي قبلنا", "easy", "uncleared table"),
    ("الذباب تطير فوق رفوف العرض", "medium", "flies near display"),
    ("لقيت قطعه قمامه صغيره في طبق السلطه", "medium", "trash in salad"),
    ("الكنب مليان بقع طعام محنطه", "easy", "stained sofa"),
    ("الزجاج فيه ابصام يدين ما تنمسح", "easy", "smudged glass"),
    ("سلة المهملات تطفح في زاوية الصاله", "easy", "overflowing trash"),
    ("نظافة المطبخ سيئه شفتها لما طلبت اروح للحمام", "medium", "kitchen visible dirty"),
    ("الصابون السائل في الحمام ينتهي ولا يعبى", "medium", "soap dispenser empty"),
    ("شفت قطه ماشيه جوا المطعم تتمسح بالكراسي", "easy", "stray cat indoors"),
    ("الكاشير يأخذ النقود ويقدم لي الاكل بنفس اليد", "hard", "no glove change"),
    ("شعرت بحشره صغيره طايره تحوم على راسي", "easy", "flying insect"),
    ("الكوب جا فيه قطعه ليمون من زبون ثاني", "medium", "leftover lemon"),
    ("الحمام ما فيه مناديل ورق", "easy", "no tissue in bathroom"),
    ("الكاشير يحط الفلوس فوق رف الاكل", "hard", "money on food shelf"),
    ("لاحظت طبقه دهون على شاشة الكاونتر", "medium", "grease on counter"),
    ("المنشفه المعطره ما تنغيرت من اول صباح", "medium", "unchanged towel"),
    ("اعمدة التكييف فيها غبار متراكم", "easy", "dusty AC vents"),
    ("الكاسات تنشف على فوطه وسخه", "hard", "dirty drying towel"),
    ("شفت اكل سايب على الطاوله من البارحه", "medium", "leftover food overnight"),
    ("الطعمه على الكنب من قبل ساعتين موجوده", "easy", "uncleared crumbs"),
    ("لقيت اظفر صغير في وسط البرجر", "hard", "nail in burger"),
    ("الموظف ينخم في يده ويرجع يخدم", "hard", "staff hygiene"),
    ("الكاتشب علبتها لزجه من خارج", "easy", "sticky condiment bottle"),
    ("الجداول الزجاج مليانه ابصام لاطفال", "easy", "kid handprints"),
    ("شفت فاره صغيره تطل من تحت الثلاجه", "easy", "mouse sighting"),
    ("القهوه فيها رواسب غريبه", "medium", "residue in coffee"),
    ("صحون المازه مكسره من الاطراف", "medium", "chipped dishes (hygiene risk)"),
    ("الفوط الورقيه على الطاوله مبلله من رائحة العفن", "medium", "moldy napkins"),
    ("الكاشير حط فلوسي في درج وفي نفس الدرج خبز", "hard", "money + food storage"),
    ("صحن الاكل ينقط ماء طلع من غسيل ناقص", "medium", "dripping wet dish"),
    ("لقيت حجر صغير في كوب الرز", "medium", "stone in rice"),
    ("شفت ذبابه ميته في طبق المازات", "easy", "dead fly in mezze"),
    ("الكاشير يتقشر جلد يده وما يلبس قفاز", "hard", "skin condition handling food"),
    ("الميز كله مغطى ببقايا توابل من قبل", "easy", "spice residue"),
    ("بقايا شعرات حيوان على كرسي القعده", "medium", "pet hair on chair"),
    ("ريحة الزبد المتعفن طالعه من زاوية المطبخ", "medium", "rancid butter smell"),
    ("الخباز يلمس العجين بكفه بدون قفاز", "hard", "baker no gloves"),
    ("الفناجين متكدسه على رف بدون غطاء", "medium", "uncovered cup storage"),
    ("شفت لقم اكل تطيح من البفيه وترجع تنحط", "hard", "buffet pick-and-replace"),
    ("ريحة الحمام وصلت الى صالة الاكل", "easy", "bathroom smell in dining"),
    ("الكنب جلد متشقق وفي شقوقه قطع طعام", "medium", "torn sofa with food in cracks"),
    ("لاحظت قطع بلاستيك صغيره في طبق الفول", "hard", "plastic shards in food"),
]

# ---------------------------------------------------------------------------
# 4) جودة الطعام (Food quality) — 50 sentences
# ---------------------------------------------------------------------------
FOOD_ROWS: list[tuple[str, str, str]] = [
    ("الدجاج مالح بحيث ما اقدر اكمل", "easy", "oversalted"),
    ("الرز ناشف وكأنه من ثلاث ايام", "easy", "dry old rice"),
    ("اللحم محروق من برا ونيه من جوا", "easy", "burnt outside raw inside"),
    ("الصلصه طعمها مغير شككتني فيها", "medium", "off-tasting sauce"),
    ("البرجر قطعه اللحم رفيعه مايوكل", "easy", "thin patty"),
    ("القهوه طعمها مر بزياده والمسحوق مكثر", "easy", "bitter overdosed coffee"),
    ("الكنافه باردة وعجينتها يابسه", "easy", "cold dry kunafa"),
    ("الطعام طعمه مكرر وكان لقاء جايز", "medium", "reheated leftover"),
    ("الصلصه الخاصه ما لها اي مذاق", "easy", "bland signature sauce"),
    ("اللحمه فيها عروق ما يتأكل منها", "easy", "tough meat tendons"),
    ("الشاورمه عجينتها مكسره ومحروقه", "easy", "shawarma bread burnt"),
    ("الفطار بايخ والبيض مايصير لون اصفر", "medium", "bland eggs"),
    ("الايس كريم ذايب من نص ساعه قبل التقديم", "easy", "melted ice cream"),
    ("الحساء بارد كأنه طلع من البراد", "easy", "cold soup"),
    ("الكفته جافه وما فيها بصل ولا صلصه", "easy", "dry kebabs"),
    ("الدجاج المشوي طعمه فاتح بدون بهارات", "easy", "underseasoned"),
    ("الخبز ابيض شاحب ما تشوف عليه شي", "easy", "pale bread"),
    ("البطاطس مقرمشه بزياده تكسرت بسنوني", "easy", "overfried fries"),
    ("السلطه الخس فيها لون اصفر ذابل", "easy", "wilted lettuce"),
    ("اللازانيا جافه وعدمت الصلصه عليها", "easy", "dry lasagna"),
    ("التمر المحشي ما هو طازج", "easy", "stale stuffed dates"),
    ("القشطه فيها طعم حامض غريب", "medium", "sour cream off"),
    ("اللبن الرايب طعمه فاسد", "easy", "spoiled yogurt"),
    ("الحلوى الباردة وكأنها من السنه الفايته", "hard", "very stale dessert"),
    ("الفول طعمه نيه بدون تطييب", "easy", "underseasoned fava"),
    ("الفلافل قاسيه من برا وغصت في حلقي", "easy", "hard falafel"),
    ("المعكرونه مطبوخه بزياده تحولت لعجينه", "easy", "overcooked pasta"),
    ("الجبن طازه بس بدون نكهه ابدا", "medium", "bland fresh cheese"),
    ("الكيك جاف بدون اي رطوبه", "easy", "dry cake"),
    ("التمر هندي طعمه مر بشكل غير معتاد", "medium", "bitter tamarind"),
    ("القهوه السعوديه باهته بدون هيل", "easy", "weak saudi coffee"),
    ("الشاي الاحمر بطعم البلاستك", "medium", "plastic-tasting tea"),
    ("الشاورما الزيت كثير ينقط منها", "easy", "greasy shawarma"),
    ("البطاطا الحلوه قاسيه وما طبخت كويس", "easy", "underdone sweet potato"),
    ("التبوله طعمها فيه بصل بايخ", "medium", "off onion in tabbouleh"),
    ("المطبق الدجاج اللي فيه ميته", "hard", "dead chicken taste"),
    ("الفطيره فيها اثار حرق من الصاج", "easy", "burnt pastry"),
    ("الكبسه فيها قرفه بزياده", "easy", "too much cinnamon"),
    ("الكبده مرّه ومحروقه بنفس الوقت", "easy", "burnt bitter liver"),
    ("الزبادي اليوناني سايل غريب", "medium", "thin greek yogurt"),
    ("القمح الكامل خبزه يابسه كانها بسكوت", "easy", "dry whole wheat bread"),
    ("الاكل ما يحمل اي بهار مجرد ماء وملح", "easy", "no seasoning"),
    ("السمك فيه طعم مر يدل على عدم النظافه", "hard", "bitter fish"),
    ("الزيتون فيه طعم مكسره", "medium", "broken-tasting olives"),
    ("الميلك شيك سايح من تاخر التقديم", "easy", "watery milkshake"),
    ("البيتزا العجينه نيه من النص", "easy", "doughy pizza"),
    ("الحمص فيه طعم تين ما هو مكانه", "medium", "off hummus"),
    ("الزعتر طعمه طيني وكانه طازج خبيث", "hard", "earthy zaatar"),
    ("المهلبيه فيها كتل دقيق ما ذابت", "easy", "lumpy muhalabia"),
    ("الفلافل وقت قليتها بداخل القلب لزجه", "medium", "sticky inside falafel"),
]

# ---------------------------------------------------------------------------
# 5) خدمة الموظفين (Staff service) — 50 sentences
# ---------------------------------------------------------------------------
STAFF_ROWS: list[tuple[str, str, str]] = [
    ("الكاشير اشار لي بيده ان اروح ولم يكلمني", "easy", "rude gesture"),
    ("الموظفه ما رحبت فينا وقت ما دخلنا", "easy", "no welcome"),
    ("النادل مشغول بالتلفون ولا ينتبه الينا", "easy", "staff on phone"),
    ("طلبت الفاتوره وانتظرت اكثر من نص ساعه", "medium", "slow bill request"),
    ("الموظف بصق علي شبه ضحكه ما عجبتني", "hard", "rude staff behavior"),
    ("المدير ما رد على شكوتي ووقف بعيد", "medium", "manager ignored"),
    ("النادل خلط طلبات الطاولات وكلهم زعلوا", "medium", "mixed up tables"),
    ("الكاشير دق على الزر بعصبيه قدامي", "easy", "agitated cashier"),
    ("الموظفه ضحكت في وجهي وقت قلت لها زبون مش راضي", "hard", "mocking smile"),
    ("استقبلونا بقعدتهم على الكنب ولم يقوم احد", "easy", "staff seated, no welcome"),
    ("النادل تجاهل اشارتي ثلاث مرات", "easy", "ignored hand signal"),
    ("الموظف صرخ على زميله قدام الزباين", "medium", "yelling at colleague"),
    ("استلم الطلب باستخفاف وكأنه يضحك", "medium", "disrespectful"),
    ("الكاشير ما اعطاني الفاتوره ورفض اعطائي", "medium", "refused receipt"),
    ("النادل ما اعطاني المنيو الا بعد ما طلبت ٣ مرات", "easy", "delayed menu"),
    ("الموظف رد علي بصوت عالي ومتعجرف", "easy", "loud arrogant reply"),
    ("استخدم الموظف الاسئله الشخصيه قدامي", "hard", "intrusive questions"),
    ("الكاشير ما رد على سؤالي عن المكونات", "easy", "no answer on ingredients"),
    ("النادل اخذ منيو والداتا والذهب عيوني وما رجع", "easy", "took menu and disappeared"),
    ("الموظفه بصوت منخفض اهانتني سراً", "hard", "covert insult"),
    ("الطباخ طلع من المطبخ يدخن قدام الزباين", "hard", "cook smoking openly"),
    ("الكاشير اجبرني ادفع بقشيش وانا ما اريد", "hard", "forced tip"),
    ("النادل سبني بدل ما يعتذر عن الخطا", "hard", "cursed me out"),
    ("ما حد سالنا ادا الاكل عجبنا او لا", "easy", "no check-in on food"),
    ("الكاشير ضحك مع زميله وقت ما طلبت", "easy", "laughed during order"),
    ("النادل وقف فوق راسي اثناء الاكل", "easy", "hovering"),
    ("الموظف صد وجهه وقت ما اشتكيت", "medium", "turned face on complaint"),
    ("النادل تلقى مكالمه شخصيه واهملنا", "easy", "personal call mid-service"),
    ("الكاشير قال خلصت الكاسه ما عندي ماء", "easy", "denied service"),
    ("الموظف نهرني عن الجلوس في طاوله محجوزه بدون اعتذار", "medium", "harsh reservation enforcement"),
    ("النادل قال ما عنده وقت يشرح المنيو", "medium", "refused menu explanation"),
    ("المدير قاطعني وانا اشتكي", "medium", "manager interrupted complaint"),
    ("الكاشير اعطاني الباقي وراح يحاسب الثاني", "easy", "abrupt change handoff"),
    ("النادل اقصى مزحه غير لايقه قدام عائلتي", "hard", "inappropriate joke"),
    ("النادل صب الماء على طاولتي بالخطا وما اعتذر", "easy", "spilled water no apology"),
    ("الكاشير اجبرنا نعطيه التقييم قبل ما نخلص", "hard", "demanding review"),
    ("الموظف اعطاني الحساب بسرعه وكأنه يطردنا", "medium", "rushed bill = pushing out"),
    ("النادل ما يعرف اي شي عن مكونات الاكل", "easy", "no menu knowledge"),
    ("الموظف لمس وجهي بكفه اثناء صب الماء", "hard", "physical contact"),
    ("الكاشير صبني بنظره مستفزه قدام الجميع", "medium", "provocative look"),
    ("الموظف لبسه غير لائق ومش نظيف", "medium", "unprofessional appearance"),
    ("النادل سمع اني اشتكيت ورفع طلباتي بعنف", "medium", "vengeful service"),
    ("الكاشير اعطاني نظره عابسه طوال الوقت", "easy", "scowling cashier"),
    ("الموظف اعطاني صحن اقل من المطلوب وقال خلاص", "medium", "shorted portion + dismissive"),
    ("النادل اجبر ابنتي تجلس وقالها اسكتي", "hard", "shushing my child"),
    ("الكاشير قاتلني على الكوبون بدون اي تعاون", "medium", "fought over coupon"),
    ("النادل قال انتم زباين قليلين ما لكم اولوية", "hard", "openly prioritizing big orders"),
    ("الكاشير ما اعطاني فرصه اختار وقاطعني", "easy", "interrupted order"),
    ("الموظف وقف ينظر التلفون اثناء اخذ طلبي", "easy", "phone while taking order"),
    ("النادل ما عرف اللغه الانجليزيه وتجاهل ضيفنا", "medium", "no English service"),
]

# ---------------------------------------------------------------------------
# 6) دقة الطلب (Order accuracy) — 50 sentences
# ---------------------------------------------------------------------------
ACCURACY_ROWS: list[tuple[str, str, str]] = [
    ("طلبت برجر بدون مخلل واجاني مليان مخلل", "easy", "no pickle ignored"),
    ("نسوا حصة البطاطس مع الوجبه", "easy", "missing side"),
    ("طلبت قهوه حليب وجاتني سوداء", "easy", "wrong coffee type"),
    ("ضاع نص الطلب في الكيس", "easy", "half order missing"),
    ("طلبت لحم ضاني وجاء دجاج", "easy", "wrong protein"),
    ("نسوا اعطاني الصوصات المطلوبه", "easy", "no sauces"),
    ("طلبت بدون جبنه ولقيتها مليانه جبن", "easy", "no cheese ignored"),
    ("الطلب بدون بصل واتى عليه بصل مفروم", "easy", "no onion ignored"),
    ("بدلوا حجم الكوكا من كبير الى صغير", "medium", "wrong size drink"),
    ("اضاف الموظف صلصه ما طلبتها", "easy", "extra unwanted sauce"),
    ("طلبت بيتزا وسط جاءت صغيره", "easy", "wrong pizza size"),
    ("بدلوا السلطه بمقبلات ثانيه", "easy", "wrong side substitution"),
    ("نسوا حصة الايس كريم", "easy", "missing dessert"),
    ("طلبت دجاج مشوي وجاء مقلي", "easy", "wrong cooking method"),
    ("اعطاني سلطة سيزر بدل ما طلبت يونانيه", "easy", "wrong salad"),
    ("الطلب فيه عبوه مياه ناقصه", "easy", "missing water bottle"),
    ("الكاشير سجل الطلب على غير اللي قلته", "medium", "cashier mis-entered"),
    ("التطبيق غير الطلب لما اضفت ملاحظه", "medium", "app overrode modifier"),
    ("اضاف لحم اضافي بدون ما اطلب وحسبه علي", "hard", "extra item + charged"),
    ("الطلب جاء بكامله من فرع غير اللي طلبت منه", "hard", "wrong branch fulfilled"),
    ("الموظف نسي يضيف الحلى المجاني للعرض", "medium", "missed promo item"),
    ("نسوا الخبز الاضافي اللي طلبته", "easy", "missing extra bread"),
    ("طلبت اضاف زبده وما لقيتها", "easy", "missing add-on"),
    ("جاء البرجر بدون اللحم بالكامل", "easy", "no meat in burger"),
    ("جاء الطلب فيه قطعه فلافل واحده مكان خمس", "easy", "wrong quantity"),
    ("اعطاني عصير برتقال بدل عصير ليمون", "easy", "wrong juice flavor"),
    ("طلبت سلطة بدون طماطم واتت مليانه طماطم", "easy", "tomato ignored"),
    ("نسوا اعطاني حصة الشاي الكرك", "easy", "missing karak"),
    ("التطبيق قبل الطلب لكن الفرع ارسل غير المطلوب", "medium", "app vs branch mismatch"),
    ("اعطاني فطر بدل البصل في الفاهيتا", "easy", "fajita topping swap"),
    ("اخطا في عدد الطلبات لما حسب", "medium", "miscount"),
    ("الكاشير اعطاني الاير فرايز بدل القلي العادي", "easy", "wrong fry type"),
    ("جاءت الوجبه بدون مشروب اساسي", "easy", "missing combo drink"),
    ("اعطاني بدل من الدجاج لحم لانه قال ما عنده دجاج", "medium", "silent substitution"),
    ("طلبت بيتزا بدون مارجريتا فضافها", "medium", "added unwanted item"),
    ("التطبيق سجل لي طبقين متشابهين", "easy", "duplicate items"),
    ("اعطاني فلافل بدل من شاورما لحم", "easy", "wrong main"),
    ("نسي اعطاني صحن البقلاوه الاضافي", "easy", "missing dessert add"),
    ("طلبت طعام حار ما اعطاني فلفل اطلاقا", "easy", "no spice ignored"),
    ("اعطاني الزبادي بدل من السلطة الجانبيه", "easy", "wrong side"),
    ("الكاشير سجل عربي بدل اللحم", "medium", "Arabic typo error"),
    ("بدون بصل اخضر بس اتى مليان بصل اخضر", "easy", "green onion ignored"),
    ("اعطاني ليمون بدل من البرتقال للشطه", "medium", "wrong citrus"),
    ("اعطاني عصير بطعم الشمام ما طلبت ذا", "easy", "wrong flavor"),
    ("جاءت الوجبه بدون التتبيله المطلوبه", "easy", "missing marinade"),
    ("الموظف خلط الحلى بين زبائن", "medium", "dessert swap"),
    ("الطلب فيه طبق متكرر مرتين", "easy", "duplicate"),
    ("التطبيق غير عدد القطع لما اضفت تعديل", "medium", "app changed quantity"),
    ("اعطاني تشيز كيك بدل من البراوني", "easy", "wrong dessert"),
    ("نسوا حتى يحطون السكر في القهوه", "easy", "no sugar"),
]

# ---------------------------------------------------------------------------
# 7) عامة (General) — 50 sentences
# Pure vague dissatisfaction without a clear single category.
# ---------------------------------------------------------------------------
GENERAL_ROWS: list[tuple[str, str, str]] = [
    ("بصراحه ما عجبتني التجربه نهائياً", "easy", "general dissatisfaction"),
    ("ما اعتقد اني راح ارجع لهالمكان", "easy", "won't return"),
    ("مكان ما يستحق الزياره ولا التجربه", "easy", "not worth visit"),
    ("شي خايس بشكل عام", "easy", "general bad"),
    ("تجربه باهته بكل المعنى", "easy", "bland experience"),
    ("لا انصح احد يخوض التجربه", "easy", "do not recommend"),
    ("الكل سيء ما اقدر اخص شي", "medium", "everything bad, nothing specific"),
    ("ضايع وقتي وفلوسي في هالمكان", "easy", "wasted time/money"),
    ("ما توقعت يكون بهذي السوء", "easy", "below expectations"),
    ("تجربه محبطه ما لها وصف", "easy", "depressing experience"),
    ("الكل بس على بعضه يضحك", "medium", "everything off"),
    ("ما عجبني فيه شي حلو ابداً", "easy", "nothing good"),
    ("راجع لها مره ثانيه لما اكره نفسي", "medium", "sarcasm-light"),
    ("ندمت اني جربتهم", "easy", "regret trying"),
    ("شي ما يستاهل الذكر", "easy", "not worth mentioning"),
    ("سيء بكل المقاييس وما اقدر اقول اكثر من ذلك", "easy", "general bad"),
    ("ما اعتقد اني حابب ارجعه", "easy", "general bad"),
    ("التجربه عادي ما تستحق التوصيه", "easy", "mediocre"),
    ("مكان لا اعرف وش يعمل صح", "medium", "nothing right"),
    ("تجربه ما ترتقي لاسمهم", "easy", "below brand reputation"),
    ("احط نجمه وحده فقط", "easy", "1-star summary"),
    ("بصراحه قمه في السوء", "easy", "peak bad"),
    ("ما ادري ليش الناس تمدحه", "easy", "puzzled by reviews"),
    ("تجربه ناقصه من كل جانب", "easy", "incomplete experience"),
    ("لا انصح ابد بهذا المطعم", "easy", "do not recommend"),
    ("افضل تطلب من اي مطعم ثاني", "easy", "alternative recommendation"),
    ("اخر مره اطلب من هالمكان", "easy", "last time"),
    ("شي صعب اني اوصفه بكلمات", "medium", "hard to describe"),
    ("الفرع كله محتاج اعادة هيكله", "easy", "branch needs rebuilding"),
    ("تجربه فاشله من كل النواحي", "easy", "failed experience"),
    ("لو في صفر نجوم لاعطيتهم", "easy", "zero stars"),
    ("ما عجبني شي من اول لحظه", "easy", "bad from start"),
    ("جلست خمس دقائق وطلعت ما ترتاح النفس", "easy", "left immediately"),
    ("شي يحبط النفس من اول دقيقه", "easy", "demoralizing"),
    ("سيء من كل وجه", "easy", "bad from all angles"),
    ("الفرع هذا مش متطور مثل غيره", "easy", "branch lagging"),
    ("جربتهم اربع مرات وكل مره تجربه سيئه", "medium", "consistent badness"),
    ("بداية النهايه لهذا المطعم", "easy", "beginning of end"),
    ("ضيعت ساعتين من عمري", "easy", "wasted hours"),
    ("ما اقدر اوصف التعاسه اللي حصلت لي", "easy", "indescribable misery"),
    ("الفرع يحتاج اعادة هيكله من الصفر", "easy", "needs full restructure"),
    ("تجربه ما تتكرر ابداً", "easy", "non-repeatable"),
    ("ما لقيت اي شي مميز فيه", "easy", "nothing special"),
    ("الفرع هذا بحالة سيئه عموماً", "easy", "branch in bad shape"),
    ("اقل تقييم ممكن اعطيه", "easy", "lowest possible rating"),
    ("التجربه دون المتوقع بكثير", "easy", "below expectations"),
    ("تجربه ما تعوض ابداً", "easy", "irreplaceable bad experience"),
    ("ما عندي صبر اكتب تفاصيل سيئه", "easy", "won't detail"),
    ("الفرع كله احتاج تجديد", "easy", "needs renewal"),
    ("عموما تجربه ما تستاهل الذكر", "easy", "not worth mentioning"),
]

# ---------------------------------------------------------------------------
# 8) وقت الانتظار (Wait time) — 50 sentences
# ---------------------------------------------------------------------------
WAIT_ROWS: list[tuple[str, str, str]] = [
    ("قعدنا ساعه قبل ما يجي الطلب", "easy", "1hr wait"),
    ("انتظرنا ٤٠ دقيقه عشان كوب قهوه", "easy", "40min coffee"),
    ("الطلب اتاخر اكثر من المعتاد بكثير", "easy", "longer than usual"),
    ("صرنا نصرخ بصوت عالي ولا حد جا", "medium", "yelling no response"),
    ("ساعتين انتظار وما اعتذر احد", "easy", "2hr no apology"),
    ("الوقت تطول بشكل لا يحتمل", "easy", "unbearable wait"),
    ("اخدنا اكثر من ساعه نص في الكاشير", "easy", "checkout wait"),
    ("طلبنا الفاتوره قبل عشرين دقيقه ولم تاتي", "easy", "bill wait"),
    ("الطلب من المطبخ اخذ ٩٠ دقيقه", "easy", "kitchen ticket"),
    ("سحبنا الطلب من السياره بانتظار طويل", "medium", "carry-out wait"),
    ("الكاشير شغال على ٣ طلبات وحد يفهم", "medium", "cashier multi-tasking"),
    ("بطء الخدمه بسبب نقص الموظفين", "medium", "understaffing"),
    ("الطلب اخذ وقت اكثر من اللي قلت لي", "easy", "longer than quoted"),
    ("جلسنا ساعتين بدون ما يلتفت لنا احد", "easy", "ignored for hours"),
    ("الكوكا اخدت ١٥ دقيقه عشان توصل", "easy", "drink wait"),
    ("جلست في الانتظار اطول من اللي اكلت فيه", "medium", "more waiting than eating"),
    ("بانتظار الزيادات اللي طلبتها على البرجر", "easy", "addon wait"),
    ("اخر شي خلصنا الطلب بعد ساعتين", "easy", "two-hour total"),
    ("الكاشير اخدته دفعتي ١٠ دقايق", "easy", "payment wait"),
    ("جلسة طويله بدون اي شي على الطاوله", "easy", "long sit with nothing"),
    ("النادل جا بعد ٢٥ دقيقه ياخذ الطلب", "easy", "long order taking"),
    ("ساعه قبل ما يجي الحلى", "easy", "dessert wait"),
    ("الانتظار في الزحمه كان خرافي", "medium", "weekend wait"),
    ("اخذت الكثير من الوقت قبل اتسلم الفاتوره", "easy", "bill delay"),
    ("الطلب اللي قلت ربع ساعه قال طلع ساعه", "easy", "underestimated time"),
    ("راحت الشهيه قبل ما يجي الاكل", "easy", "lost appetite"),
    ("نص ساعه عشان يحضر السلطه", "easy", "salad wait"),
    ("جلسنا ساعه ٧٥ دقيقه على الباب نستنى دور", "medium", "lobby queue"),
    ("سحبت الطلب وكنت ابي اطلع", "easy", "left because of wait"),
    ("جلست منتظره الكاشير يحاسب ٢٠ دقيقه", "easy", "register wait"),
    ("استمرت الاوامر بدون اي توصيل من الطباخ", "medium", "kitchen backed up"),
    ("ساعتين قبل ما اخذ كوب قهوه", "easy", "extreme coffee wait"),
    ("اول جلستنا اخر طلب الحلى اخذ ساعه", "easy", "dessert ticket"),
    ("الموظف قال خمس دقايق صار خمس ساعات", "easy", "exaggerated wait"),
    ("جلست منتظره طلبي ولا اي خبر", "easy", "no update"),
    ("لما طلبت اعرف وضع طلبي قال انتظر بدون توضيح", "medium", "no transparency"),
    ("بطء التحضير غير منطقي مقابل عدد الزباين", "medium", "slow vs demand"),
    ("جلسنا في الانتظار اكثر من اللي اكلنا فيه", "easy", "wait>eat"),
    ("الكاشير معطل وايقاف الكل", "medium", "broken POS = wait"),
    ("سحبت الطلب من الموقع وكان متاخر", "easy", "self-pickup delayed"),
    ("سحبنا بعد ساعه من تاكيد الاوردر", "easy", "post-confirm wait"),
    ("الفاتوره اخذت اطول من تحضير الاكل", "medium", "bill > prep"),
    ("اول طلب اخذ ساعه ثاني طلب اخذ ساعتين", "easy", "compounding wait"),
    ("صرت اتصل عليهم عشان اعرف وضعي بالطابور", "easy", "had to call"),
    ("الزحمه يوم الويك اند تخلي الانتظار ٣ ساعات", "medium", "weekend extreme"),
    ("استمتعت بالاكل بعد ما خلصت اعصابي بالانتظار", "medium", "sarcastic enjoy"),
    ("الانتظار للحساب اطول من الانتظار للاكل", "medium", "bill > food wait"),
    ("ساعه ونص ولم اتسلم اي شي", "easy", "1.5hr nothing"),
    ("جلست في الموقف انتظر طلبي يجي", "easy", "parking lot wait"),
    ("ساعة كامله انتظار للسلطه الجانبيه", "easy", "side dish hour"),
]

# ---------------------------------------------------------------------------
# 9) Multi-aspect — 30 sentences
# expected_label = "multi", notes records the dominant + secondary categories.
# ---------------------------------------------------------------------------
MULTI_ROWS: list[tuple[str, str, str, str]] = [
    # (text, difficulty, expected dominant label, notes)
    ("الاكل بايخ والموظف عبوس وما رد علي", "hard", FOOD, "FOOD+STAFF"),
    ("الفاتوره غاليه والمندوب تاخر ساعتين", "hard", DELIVERY, "DELIVERY+PRICE (delivery dominant)"),
    ("نسوا نص الطلب والكاشير تكلم معي بطريقه سيئه", "hard", ACCURACY, "ACCURACY+STAFF"),
    ("الحمام مقرف والاكل مالح بزياده", "hard", HYGIENE, "HYGIENE+FOOD"),
    ("الانتظار طويل واسعار غير معقوله", "hard", WAIT, "WAIT+PRICE"),
    ("الموظف بطيء ووجهه عابس والاكل بارد", "hard", STAFF, "STAFF+FOOD"),
    ("المندوب نسي نص الطلب وضاع الكوبون", "hard", DELIVERY, "DELIVERY+ACCURACY"),
    ("الكاشير اخطا في الحسبه والسعر غالي", "hard", PRICE, "PRICE+STAFF"),
    ("الاكل وصل بعد ساعتين باردا وضايع نص الطلب", "hard", DELIVERY, "DELIVERY+ACCURACY+FOOD"),
    ("ضاع الصوص في الكيس والمندوب ما رد", "hard", DELIVERY, "DELIVERY+ACCURACY"),
    ("جدا غالي ومش بنفس النظافه المتوقعه", "hard", PRICE, "PRICE+HYGIENE"),
    ("الانتظار خرافي والكاشير يصرخ", "hard", WAIT, "WAIT+STAFF"),
    ("نسوا الايس كريم وحسبوه علي بالفاتوره", "hard", ACCURACY, "ACCURACY+PRICE"),
    ("ريحه كريهه والاكل ما لها طعم", "hard", HYGIENE, "HYGIENE+FOOD"),
    ("الموظف اهانني والاسعار مرتفعه", "hard", STAFF, "STAFF+PRICE"),
    ("التطبيق ضرب فلوس مرتين والمندوب جا متاخر", "hard", DELIVERY, "DELIVERY+PRICE"),
    ("الصحون متسخه والاكل قديم بزياده", "hard", HYGIENE, "HYGIENE+FOOD"),
    ("الطلب وصل ناقص بعد ساعتين", "hard", DELIVERY, "DELIVERY+ACCURACY+WAIT"),
    ("اخذنا ساعه ونص ولما جا الاكل اتاكلت ميته", "hard", FOOD, "WAIT+FOOD (food dominant)"),
    ("نسوا التتبيله والوقت طال", "hard", ACCURACY, "ACCURACY+WAIT"),
    ("الكاشير شتمني وحاسبني غلط", "hard", STAFF, "STAFF+PRICE"),
    ("الصحن مقدم بطريقه قذره والطلب ناقص", "hard", HYGIENE, "HYGIENE+ACCURACY"),
    ("التوصيل اطول من المعتاد والاكل ساح", "hard", DELIVERY, "DELIVERY+FOOD"),
    ("سحب من البطاقه مرتين والطلب وصل غلط", "hard", PRICE, "PRICE+ACCURACY"),
    ("الموظف ضحك علي وحاسبني زياده", "hard", STAFF, "STAFF+PRICE"),
    ("شعره في الاكل والاستجابه بطيئه", "hard", HYGIENE, "HYGIENE+STAFF"),
    ("اللحم نيه ووصل بعد ساعتين باردا", "hard", FOOD, "FOOD+DELIVERY+WAIT"),
    ("الكاشير تجاهلني والكوبون ما اشتغل", "hard", STAFF, "STAFF+PRICE"),
    ("الانتظار طويل والاكل ضايع نصه", "hard", WAIT, "WAIT+ACCURACY"),
    ("ريحه دخان من المطبخ والاكل طعمه فاسد", "hard", HYGIENE, "HYGIENE+FOOD"),
]

# ---------------------------------------------------------------------------
# 10) Adversarial — 30 sentences (sarcasm, negation, irony)
# ---------------------------------------------------------------------------
ADVERSARIAL_ROWS: list[tuple[str, str, str, str]] = [
    # (text, difficulty, expected_label, notes)
    ("يا سلام على الانتظار الممتع لمدة ساعتين", "hard", WAIT, "sarcasm wait"),
    ("ما شاء الله موظف يعرف يضحك في وجهك وانت تشتكي", "hard", STAFF, "sarcasm staff"),
    ("جدا متحمس للفاتوره اللي ضاعفت ميزانيتي", "hard", PRICE, "sarcasm price"),
    ("احب الذباب اللي يزين الطعام عندهم", "hard", HYGIENE, "sarcasm hygiene"),
    ("ابدع الطباخ في تحويل اللحم الى فحم", "hard", FOOD, "sarcasm food burnt"),
    ("شكرا للمندوب اللي ضيع طريقي وضيع الاكل", "hard", DELIVERY, "sarcasm delivery"),
    ("شطاره عاليه في نسيان نص الطلب", "hard", ACCURACY, "sarcasm accuracy"),
    ("روعه الانتظار اللي يخليك تتنازل عن الجوع", "hard", WAIT, "sarcasm wait"),
    ("الاكل مش بايخ هو فقط بلا طعم", "hard", FOOD, "negation hides bad"),
    ("ليس الاكل لذيذ ولا الخدمه محترمه", "hard", GENERAL, "double negation"),
    ("ما اقول الكاشير مهذب لكنه قح فظ", "hard", STAFF, "negation + insult"),
    ("ما اقول السعر غالي بس ماحد يدفعه راضي", "hard", PRICE, "negation hides price complaint"),
    ("ما لقيت شي وسخ لكن الصحون فيها بقع", "hard", HYGIENE, "negation contradicts hygiene"),
    ("الطلب ما هو غلط هو فقط مش اللي طلبته", "hard", ACCURACY, "negation hides accuracy"),
    ("ما تأخر التوصيل كثير ٩٠ دقيقه عاديه", "hard", DELIVERY, "sarcastic 'normal'"),
    ("الانتظار ليس طويل هو فقط لا ينتهي", "hard", WAIT, "sarcastic wait"),
    ("شعرت بسعادة الانتظار لساعتين", "hard", WAIT, "sarcasm wait"),
    ("ابداع في تقديم الاكل بدرجة غرفة", "hard", FOOD, "sarcasm food cold"),
    ("الموظف فنان في توزيع التجاهل بالتساوي", "hard", STAFF, "sarcasm staff"),
    ("الكاشير قمة الذكاء في حسبة غلط", "hard", PRICE, "sarcasm price"),
    ("ما عجبني شي بصراحه لكن قالوا الكل احبه", "hard", GENERAL, "negation general"),
    ("الذباب هي اضافه مجانيه ما يجب الشكوى", "hard", HYGIENE, "sarcasm hygiene"),
    ("اخر ابداع في خلط الطلبات بين الزبائن", "hard", ACCURACY, "sarcasm accuracy"),
    ("ابدع المندوب وضاع لمدة ٤٠ دقيقه", "hard", DELIVERY, "sarcasm delivery"),
    ("ماحد شفي مثلهم في تطبيق العروض بشكل خاطئ", "hard", PRICE, "sarcasm promo"),
    ("ربما الاكل ليس سيئا لكنه على وشك الفساد", "hard", FOOD, "soft critique food"),
    ("ليست تجربه فاشله هي فقط لا تستحق العوده", "hard", GENERAL, "negation general"),
    ("الموظف يبتسم بطريقه تخفي اهانه", "hard", STAFF, "implied staff insult"),
    ("ما ادري ادا الاكل بارد عمداً ام صدفه", "hard", FOOD, "rhetorical food complaint"),
    ("اشكرهم على عدم اعطائي السلطه المطلوبه", "hard", ACCURACY, "sarcasm accuracy"),
]

# ---------------------------------------------------------------------------
# 11) Mixed dialect (Arabic + English) — 20 sentences
# ---------------------------------------------------------------------------
MIXED_DIALECT_ROWS: list[tuple[str, str, str, str]] = [
    # (text, difficulty, expected_label, notes)
    ("الديليفري took forever ما يستاهل", "medium", DELIVERY, "english 'took forever'"),
    ("الفاتوره too expensive والاكل قليل", "medium", PRICE, "english adjective"),
    ("Staff ما يعرفون customer service", "medium", STAFF, "english service"),
    ("الكاشير اعطاني wrong order بدون اعتذار", "medium", ACCURACY, "english 'wrong order'"),
    ("الاكل cold لما وصل من المندوب", "medium", FOOD, "english 'cold'"),
    ("Wait time عندهم disaster كامل", "medium", WAIT, "english wait time"),
    ("Hygiene عندهم zero ما اقدر اوصف", "medium", HYGIENE, "english hygiene"),
    ("Overall bad experience ما انصح فيه", "medium", GENERAL, "english general"),
    ("Driver ضاع وما رد على calls", "medium", DELIVERY, "english driver/calls"),
    ("Order was wrong وحسبوا علي extras", "medium", ACCURACY, "english+arabic order"),
    ("Service slow والموظف rude بزياده", "medium", STAFF, "english slow rude"),
    ("Price too high لوجبه بسيطه", "medium", PRICE, "english price"),
    ("Bathroom dirty وما فيه toilet paper", "medium", HYGIENE, "english bathroom"),
    ("Food taste weird وكأنه فاسد", "medium", FOOD, "english taste"),
    ("Long queue والكاشير one only", "medium", WAIT, "english queue"),
    ("Bad experience مش بأول مره", "medium", GENERAL, "english experience"),
    ("App keeps crashing لما اضيف items", "medium", DELIVERY, "english app crashing"),
    ("Coupon code not working وما اقدر استخدمه", "medium", PRICE, "english coupon"),
    ("Order missing items وما اشتغل التطبيق", "medium", ACCURACY, "english missing items"),
    ("My order is incomplete وضاع نص الكيس", "medium", ACCURACY, "english incomplete"),
]

# ---------------------------------------------------------------------------
# 12) Out-of-domain (non-restaurant) — 10 sentences
# expected_label = "no_complaint"
# ---------------------------------------------------------------------------
OOD_ROWS: list[tuple[str, str, str]] = [
    ("الجو في الرياض اليوم حار جدا والشمس قويه", "hard", "weather"),
    ("سيارتي البطاريه فاضيه ومحتاج روبيد", "hard", "car battery"),
    ("التطبيق البنكي ما يفتح من الصباح", "hard", "bank app"),
    ("المستشفى انتظرت ٣ ساعات بدون ما يجيني الدكتور", "hard", "hospital wait"),
    ("شركة الكهرباء قطعت الخدمه بدون اشعار", "hard", "utility"),
    ("الانترنت بطيء جدا اليوم ما اقدر اشتغل", "hard", "internet speed"),
    ("الطيارة تاخرت ساعتين على الرحله", "hard", "airline delay"),
    ("جوالي اللمسه ما تشتغل من امس", "hard", "phone touch"),
    ("الجامعه الدرجات اللي اخذتها ظالمه", "hard", "university grade"),
    ("الجار حقي صوت موسيقاه عالي طول الليل", "hard", "neighbor noise"),
]

# ---------------------------------------------------------------------------
# 13) Very short (1-2 words) — 5 sentences
# expected_label = "abstain"
# ---------------------------------------------------------------------------
VERY_SHORT_ROWS: list[tuple[str, str, str]] = [
    ("بايخ", "hard", "1-word"),
    ("سيء جدا", "hard", "2-word"),
    ("زفت", "hard", "1-word slang"),
    ("ما يستاهل", "hard", "fixed expression"),
    ("للاسف", "hard", "1-word interjection"),
]

# ---------------------------------------------------------------------------
# 14) Long (50+ words) — 5 sentences
# Each tagged with dominant category in 'expected_label'.
# ---------------------------------------------------------------------------
LONG_ROWS: list[tuple[str, str, str, str]] = [
    # (text, difficulty, expected_label, notes)
    (
        "زرت الفرع الجديد اللي افتتح في شمال الرياض بعد ما شفت اعلاناتهم الكثيره في السوشال ميديا وتوقعت تجربه مميزه بناء على تقييمات الزملاء لكن للاسف الزياره كانت كارثيه من اول دقيقه استقبالنا كان فاتر والمضيفه ما رحبت فينا وبعدين النادل ما جا الا بعد ربع ساعه واخذ الطلب باستخفاف وضحك مع زميله بلا سبب وكلما طلبنا اضافه او تعديل يرد بلا اكتراث ولا اعرف ادا كان ذلك بسبب نقص التدريب او ضعف الاداره",
        "hard",
        STAFF,
        "long staff-dominant review",
    ),
    (
        "طلبت من خلال تطبيق التوصيل قبل تقريبا ساعتين وكان من المفترض ان يوصل خلال خمس واربعين دقيقه حسب التقدير الموجود في الطلب لكن الواقع كان مختلف تماما حيث تاخر المندوب اكثر من ساعتين بدون اي تواصل ولا اشعار وحتى لما اتصلت بخدمة العملاء قالوا انه لا توجد معلومات عن طلبي ولا يقدرون يعرفون مكانه الحالي وبعد ساعتين والاكل وصل بارد",
        "hard",
        DELIVERY,
        "long delivery-dominant"
    ),
    (
        "هذا الفرع من اسوء التجارب اللي مريت فيها منذ زمن طويل من ناحية النظافه شفت بقايا طعام على الكنب والطاولات ولاحظت ذبابه تطير حول البيتزا الموجوده في رف العرض وحتى الحمام كان قذرا جدا بدون مناديل ورق ولا صابون ووجدت بقعه دم على المنشفه التي اعطاني الكاشير وذلك خطر يهدد صحة الزبائن",
        "hard",
        HYGIENE,
        "long hygiene-dominant"
    ),
    (
        "الفاتوره طلعت اكثر من الضعف عن المتوقع وعندما طلبت تفسير من الكاشير قال ان هذه هي الاسعار بعد التحديث الجديد ولكن لم يخبروني قبل الطلب وحتى الكوبون اللي حصلت عليه من التطبيق لم يطبق على الفاتوره وقالوا ان شروطه لا تنطبق وعندما حاولت ان اوضح لهم انها كانت ضمن العرض الموجود في الاعلان قاطعوني ورفضوا التعديل بدون تفسير منطقي",
        "hard",
        PRICE,
        "long price-dominant"
    ),
    (
        "طلبت وجبه عائليه تتضمن دجاج مشوي مع رز وسلطه ومشروبات وحلى لاربع اشخاص لكن عندما وصل الطلب اكتشفت ان نصف العناصر ناقصه فلا توجد سلطه ولا اضافات الزبده ولا الحلى المجاني المرفق مع العرض كما ان نوع الدجاج كان مقلي بدلا من مشوي والمشروبات كلها بدون ثلج رغم انني طلبتها مثلجه بشكل واضح في الملاحظات",
        "hard",
        ACCURACY,
        "long accuracy-dominant"
    ),
]


def _norm(s: str) -> str:
    """Mimic the production cleaning pipeline closely enough for dup-checking.

    Production clean(): strip diacritics, normalize alef/ya/ta-marbuta,
    strip punctuation, lowercase ASCII, collapse whitespace.
    For dedup we just lowercase + collapse whitespace + strip the most common
    Arabic normalizations.
    """
    import re
    s = s.lower().strip()
    s = re.sub(r"[ً-ٟ]", "", s)  # tashkeel
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
    s = s.replace("ى", "ي")
    s = s.replace("ة", "ه")
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> int:
    out_path = Path(
        "C:/Users/FSOS/Downloads/complaint classifier-20260501T130400Z-3-001/complaint classifier/tests/fixtures/production_test_500.csv"
    )

    rows: list[dict] = []
    counter = 1

    def add(text: str, expected: str, difficulty: str, attack: str, notes: str) -> None:
        nonlocal counter
        rid = f"T{counter:03d}"
        counter += 1
        rows.append({
            "id": rid,
            "text": text,
            "expected_label": expected,
            "difficulty": difficulty,
            "attack_type": attack,
            "notes": notes,
        })

    # 1) delivery
    for text, diff, notes in DELIVERY_ROWS:
        add(text, DELIVERY, diff, "clean", notes)
    # 2) price
    for text, diff, notes in PRICE_ROWS:
        add(text, PRICE, diff, "clean", notes)
    # 3) hygiene
    for text, diff, notes in HYGIENE_ROWS:
        add(text, HYGIENE, diff, "clean", notes)
    # 4) food
    for text, diff, notes in FOOD_ROWS:
        add(text, FOOD, diff, "clean", notes)
    # 5) staff
    for text, diff, notes in STAFF_ROWS:
        add(text, STAFF, diff, "clean", notes)
    # 6) accuracy
    for text, diff, notes in ACCURACY_ROWS:
        add(text, ACCURACY, diff, "clean", notes)
    # 7) general
    for text, diff, notes in GENERAL_ROWS:
        add(text, GENERAL, diff, "clean", notes)
    # 8) wait
    for text, diff, notes in WAIT_ROWS:
        add(text, WAIT, diff, "clean", notes)
    # 9) multi
    for text, diff, dominant, notes in MULTI_ROWS:
        add(text, "multi", diff, "multi_aspect", f"dominant={dominant}; {notes}")
    # 10) adversarial
    for text, diff, expected, notes in ADVERSARIAL_ROWS:
        attack = "sarcasm" if "sarcasm" in notes.lower() else "negation"
        add(text, expected, diff, attack, notes)
    # 11) mixed dialect
    for text, diff, expected, notes in MIXED_DIALECT_ROWS:
        add(text, expected, diff, "mixed_dialect", notes)
    # 12) OOD
    for text, diff, notes in OOD_ROWS:
        add(text, "no_complaint", diff, "ood", notes)
    # 13) very short
    for text, diff, notes in VERY_SHORT_ROWS:
        add(text, "abstain", diff, "very_short", notes)
    # 14) long
    for text, diff, expected, notes in LONG_ROWS:
        add(text, expected, diff, "long", notes)

    # ---- Dedup-check against training set ----
    print(f"Generated {len(rows)} rows", file=sys.stderr)
    print("Loading training set for membership check...", file=sys.stderr)

    import pandas as pd

    df = pd.read_csv(
        "C:/Users/FSOS/Downloads/complaint classifier-20260501T130400Z-3-001/complaint classifier/data/text/complaints_labeled.csv"
    )
    train_norm = set(df["text"].astype(str).map(_norm).tolist())
    overlap = 0
    overlap_ids: list[str] = []
    for r in rows:
        if _norm(r["text"]) in train_norm:
            overlap += 1
            overlap_ids.append(r["id"])
    print(f"Exact-normalized overlap with training set: {overlap}/{len(rows)} ({overlap*100/len(rows):.2f}%)", file=sys.stderr)
    if overlap_ids:
        print("Overlapping IDs (need rewrite):", overlap_ids, file=sys.stderr)

    # ---- Distribution sanity ----
    from collections import Counter
    cats = Counter(r["expected_label"] for r in rows)
    print("\nLabel distribution:", file=sys.stderr)
    for k, v in cats.most_common():
        print(f"  {k}: {v}", file=sys.stderr)
    attacks = Counter(r["attack_type"] for r in rows)
    print("\nAttack-type distribution:", file=sys.stderr)
    for k, v in attacks.most_common():
        print(f"  {k}: {v}", file=sys.stderr)

    # ---- Write CSV ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "text", "expected_label", "difficulty", "attack_type", "notes"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
