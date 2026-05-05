# Ambience Boundaries — The Exact Rule

The single hardest thing about the v3 ambience class was that labelers couldn't agree on what counted. This file gives the exact rule.

## The rule, in one sentence

**A complaint is `الجو والمكان` (ambience) if and only if the user is complaining about a property of the physical place — what they could feel, smell, see, or hear about the room itself, independent of the food, the staff, or the order.**

If you can imagine the same complaint applying to an empty restaurant with no food and no staff, it's probably ambience.

## What counts

| Subtype | Examples |
|---|---|
| `temperature_ac` | المكان حار، المكيف ما يبرد، التكييف يثلج |
| `seating_comfort` | الكراسي قاسية، الكنب وصخ، طاولة تهتز، ما فيه مكان نقعد |
| `space_crowding` | المكان زحمة، ضيق، متلاصقين، مافي مساحة |
| `noise_music` | الصوت عالي، الموسيقى مزعجة، أطفال يصرخون، ما نقدر نسولف |
| `lighting` | الإضاءة قوية على العين، النور ضعيف، ظلام |
| `smell` | ريحة الزيت ماسكة في المكان، ريحة دخان، ريحة الحمام طالعة |
| `decor_furniture` | الديكور قديم، الفرش مهلوك، الجدران متسخة من الزمن |
| `parking` | ما فيه مواقف، الباركنج بعيد، صعب الوصول للمدخل |
| `bathroom_facilities` | الحمام خربان، السيفون ما يشتغل، ما فيه صابون |
| `outdoor_view` | الجلسة الخارجية مكشوفة، التراس ما فيه ظل |
| `privacy` | ما فيه خصوصية، كل الناس تشوفك، ما فيه قسم عوائل |
| `general_place_vibe` | المطعم كئيب، الجو ثقيل، المحل يكتم |

## What does NOT count

If the complaint is primarily about any of these, **it is not ambience**:

| If the complaint is about... | Label it as |
|---|---|
| Taste, freshness, temperature of the **food itself** | `جودة الطعام` |
| Cleanliness of food/dishes/cutlery, food contamination | `النظافة` |
| Staff behavior, attitude, speed, knowledge | `خدمة الموظفين` |
| Delivery courier, app, lateness of delivery | `التوصيل` |
| Wrong items, missing items, modifications ignored | `دقة الطلب` |
| Price vs. value | `السعر والقيمة` |
| Time waiting at the restaurant for the order | `وقت الانتظار` |

## The boundary examples (memorize these)

These are the exact pairs that broke v3. If you can label all 12 correctly, you can label ambience.

| # | Text | Label | Why |
|---|---|---|---|
| 1 | الأكل بارد | `جودة الطعام` | Food temperature, not room temperature |
| 2 | المكان بارد والمكيف قوي | `الجو والمكان` (temperature_ac) | AC / room temperature |
| 3 | الشاورما حارة | `جودة الطعام` | Food spice level |
| 4 | المكان حار والمكيف ما يبرد | `الجو والمكان` (temperature_ac) | AC failure |
| 5 | البرجر فيه زيت كثير | `جودة الطعام` | Food grease content |
| 6 | ريحة الزيت ماسكة في المطعم | `الجو والمكان` (smell) | Smell of the room |
| 7 | الموظف صوته عالي | `خدمة الموظفين` | Staff behavior |
| 8 | الموسيقى صوتها عالي | `الجو والمكان` (noise_music) | Restaurant noise |
| 9 | الطاولة وسخة فيها بقع طعام | `النظافة` | Hygiene contamination |
| 10 | الطاولة تهتز كل ما تحط شي | `الجو والمكان` (decor_furniture) | Furniture condition |
| 11 | المندوب وصل متأخر | `التوصيل` | Delivery courier |
| 12 | المكان زحمة الانتظار طويل | `وقت الانتظار` (primary) | Wait time is the actionable complaint; crowding is mentioned but not the focus |

## The bathroom decision (must be consistent)

The v3 schema put bathroom complaints under `النظافة` because they were almost always about cleanliness ("الحمام وصخ"). For this revival:

**Rule:** A bathroom complaint is `الجو والمكان` (`bathroom_facilities` subtype) **only if** it's about facilities (broken sink, no soap, no paper towels, broken door). If it's about cleanliness ("الحمام قذر / وصخ / مقرف"), it stays `النظافة`.

| Text | Label |
|---|---|
| الحمام وصخ | `النظافة` |
| الحمام قذر جدا | `النظافة` |
| السيفون خربان | `الجو والمكان` (bathroom_facilities) |
| ما فيه صابون في الحمام | `الجو والمكان` (bathroom_facilities) |
| باب الحمام ما يقفل | `الجو والمكان` (bathroom_facilities) |

## Multi-aspect tie-breakers

When a complaint mentions ambience AND another class, use this priority:

1. If the user took an **action** because of one aspect (left without ordering, asked for table change, complained to manager), that aspect wins.
2. Otherwise, the **first complaint mentioned** wins.
3. If still tied, ambience loses the tie. (We're trying to be conservative — over-labeling killed v3.)

Example: "الجو حار والأكل ما يستاهل" → `جودة الطعام` (action-able complaint about food, ambience is mood-setting).

## When in doubt → review_needed

Send the row to `review_needed.csv` and discuss in the labeling meeting. Better to defer than guess. The v3 ambience class died from labelers guessing.
