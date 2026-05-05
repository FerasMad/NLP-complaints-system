"""Generate the team message PDF in Arabic (RTL).

Uses reportlab + arabic_reshaper + python-bidi for proper Arabic shaping
and bidirectional rendering. Falls back to Windows Arial which has
comprehensive Arabic glyph coverage.

Output: team_message.pdf at the project root.
"""
from __future__ import annotations

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "team_message.pdf"

# Use Arial — Windows ships with full Arabic glyph coverage in Arial
ARIAL_REG = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")

if not ARIAL_REG.exists() or not ARIAL_BOLD.exists():
    raise SystemExit("Arial fonts not found at C:/Windows/Fonts/")

pdfmetrics.registerFont(TTFont("Arial", str(ARIAL_REG)))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(ARIAL_BOLD)))

# ---- Design tokens (warm cream, terracotta accent — same as the Space) ----
CREAM = HexColor("#F5EFE6")
INK = HexColor("#2D211A")
INK_MUTED = HexColor("#5C4F45")
TERRACOTTA = HexColor("#A14828")
BORDER = HexColor("#D9CFC0")


def shape(text: str) -> str:
    """Reshape Arabic text and apply bidi for proper display order."""
    if not text:
        return ""
    return get_display(arabic_reshaper.reshape(text))


# ---- Content (verbatim from the user's brief) ----

TITLE = "تحديث للفريق"
SUBTITLE = "مشروع تصنيف شكاوى المطاعم العربية"

SECTIONS = [
    ("__intro__", [
        "السلام عليكم ورحمة الله،",
        "",
        "أبي أكون شفّاف معكم على ما صار في المشروع. في نقطة مهمّة لازم أوضّحها قبل ما نكمل: غيّرنا التوجّه من الخطّة الأصلية، وفي تفاصيل تستاهل التوضيح.",
    ]),
    ("الخطّة اللي اتفقنا عليها", [
        "كانت الخطّة نبدأ بنموذج TF-IDF ونبنيه من الصفر. كان قرار سليم بالنظر لوقتنا ومواردنا، وكان يعطينا فهم أعمق للأساسيات.",
    ]),
    ("ليش غيّرنا التوجّه", [
        "لمّا وصلنا لـ Phase 3 وبدأنا التدريب، اصطدمنا بجدار:",
        "•  TF-IDF أعطى ٨٩٫٨٪ دقّة إجمالية، بس F1 لكل فئة كان كارثي",
        "•  فئة \"الجو والمكان\": ٢٤٪ F1",
        "•  فئة \"دقّة الطلب\": ٢٢٪ F1",
        "•  قاعدتنا في AI Club إن كل فئة لازم تتجاوز ٧٠٪ F1، وما كنّا قريبين من هالرقم",
        "",
        "التدريب من الصفر على ٩٨ ألف سطر عربي ما كان كافي. اللغة العربية معقّدة من الناحية الصرفية وتحتاج بيانات بحجم أكبر بكثير من اللي عندنا.",
    ]),
    ("القرار اللي أخذته", [
        "انتقلت إلى transfer learning مع نماذج CAMeLBERT-mix و MARBERT و AraBERTv02. هذه نماذج عربية مُدرَّبة مسبقاً على بيانات ضخمة، وفصّلتها على بياناتنا عن طريق fine-tuning.",
        "",
        "أعرف إن هذا يخالف الخطّة الأصلية. اتّخذت القرار لأن البديل كان مشروع ما يصل للجودة المطلوبة، والوقت كان يضغط.",
    ]),
    ("النتيجة النهائية", [
        "•  ٩٥٫٠٥٪ دقّة على مجموعة اختبار مستقلّة من ١٣٬٩٨٦ مراجعة",
        "•  كل الفئات الثماني فوق ٨٠٪ F1",
        "•  خطأ المعايرة (ECE) ٠٫٠١٤ بعد temperature scaling",
        "•  فاصل ثقة ٩٥٪: بين ٩٤٫٧٠٪ و ٩٥٫٤١٪",
    ]),
    ("شغلي في المشروع", [
        "١.  توسيع البيانات: scraping من تطبيقات التوصيل السعودية (HungerStation, Jahez, Mrsool, Talabat) — أضفت ١٤٬٧٨٩ مراجعة حقيقية",
        "٢.  Bake-off بين ٥ معماريات: CAMeLBERT × ٢ seeds + CAMeLBERT-da + MARBERT + AraBERTv02 + XLM-R",
        "٣.  Pseudo-labeling بالاتفاق بين أعلى نموذجين",
        "٤.  Audit لفئة \"الجو والمكان\": اكتشفنا أن ٩٩٪ من تصنيفاتها كانت خاطئة، فحذفنا الفئة",
        "٥.  EDA augmentation للفئات الضعيفة (دقّة الطلب وعامّة)",
        "٦.  تجارب التقييم الكاملة: bootstrap CI، calibration، perturbation robustness، cross-dialect canary",
        "٧.  بناء API بـ FastAPI + واجهة Gradio",
        "٨.  تصميم وبرمجة واجهة المستخدم (مع تكرار وتعديل عدّة مرّات)",
        "٩.  كتابة كل التوثيقات (README، REPORT، MODEL_CARD، DATA_CARD)",
        "١٠.  النشر على HuggingFace Hub و Spaces",
    ]),
    ("فضل الفريق", [
        "ما كان يصير شي من هذا بدون شغلكم:",
        "",
        "•  لانا — pipeline التنظيف اللي بنيتيها (clean) لها مكان مركزي في كل خطوة من خطوات البناء",
        "•  خولة — التصنيفات اللي راجعتيها هي قاعدة كل شي. ٩٨ ألف سطر مصنّف ما يجي من فراغ",
        "•  ريما ومحمد — قرارات schema الأصلية، والمحاولات الأولى في التدريب اللي علّمتنا حدود الـ baseline",
        "•  مشعل — دورك في النشر لسّا مهمّ، الـ Space جاهز ويحتاج فقط ربط حساب HuggingFace",
    ]),
    ("اللي بقي من شغل", [
        "•  مشعل: استلام نشر الـ Space النهائي (الكود جاهز، فقط ربط الحساب)",
        "•  شغل اختياري لمن يبي يأخذه:",
        "    •  تدريب multi-label للشكاوى متعدّدة الجوانب",
        "    •  Knowledge distillation لتقليل حجم النموذج النهائي",
        "    •  مقارنة مع GPT-4o كـ baseline تجاري",
    ]),
    ("ملاحظة عن الأدوات", [
        "استخدمت Claude (مساعد AI) للتسريع في كتابة الـ boilerplate و eval scripts والواجهة. القرارات النموذجية والبحثية كلّها مبنيّة على البيانات والـ pipeline اللي بنيناها كفريق. أحبّ نكون شفّافين عن هذي النقطة.",
    ]),
    ("خاتمة", [
        "أعرف إن قرار تغيير الخطّة كان لازم يصير بنقاش جماعي، وأعتذر إنّي أخذته منفرداً. الوقت كان يضغط وحسّيت إن المشروع راح ينهار لو ما تحرّكنا بسرعة. لو في ملاحظات أو أسئلة، أبي نناقشها في الاجتماع القادم.",
        "",
        "تحيّاتي،",
        "فراس مدخلي",
        "رئيس فريق NLP — AI Club",
    ]),
]


# ---- Page geometry ----
PAGE_W, PAGE_H = A4
MARGIN_X = 22 * mm
MARGIN_TOP = 22 * mm
MARGIN_BOTTOM = 20 * mm
USABLE_W = PAGE_W - 2 * MARGIN_X
RIGHT_EDGE = PAGE_W - MARGIN_X  # text right-edge anchor for RTL


def wrap_to_width(c: canvas.Canvas, text: str, font: str, size: float, max_w: float) -> list[str]:
    """Greedy word-wrap respecting max_w in points. Operates on raw Arabic
    (pre-shaping); shaping is applied per output line later."""
    words = text.split(" ")
    lines: list[str] = []
    current: list[str] = []

    def measure(s: str) -> float:
        return c.stringWidth(shape(s), font, size)

    for w in words:
        candidate = (" ".join(current + [w])).strip()
        if measure(candidate) <= max_w or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


def draw_rtl(c: canvas.Canvas, text: str, x_right: float, y: float, font: str, size: float):
    """Draw shaped Arabic text right-anchored at x_right, baseline y."""
    if not text.strip():
        return
    shaped = shape(text)
    width = c.stringWidth(shaped, font, size)
    c.setFont(font, size)
    c.drawString(x_right - width, y, shaped)


def main() -> int:
    c = canvas.Canvas(str(OUTPUT), pagesize=A4)
    c.setTitle("تحديث للفريق - مشروع تصنيف شكاوى المطاعم")
    c.setAuthor("Feras Madkhali")

    # Background fill across whole page
    c.setFillColor(CREAM)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    y = PAGE_H - MARGIN_TOP

    # ---- Title block ----
    title_size = 22
    subtitle_size = 12

    c.setFillColor(TERRACOTTA)
    draw_rtl(c, TITLE, RIGHT_EDGE, y, "Arial-Bold", title_size)
    y -= title_size + 6

    c.setFillColor(INK_MUTED)
    draw_rtl(c, SUBTITLE, RIGHT_EDGE, y, "Arial", subtitle_size)
    y -= subtitle_size + 16

    # Hairline divider
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.6)
    c.line(MARGIN_X, y, RIGHT_EDGE, y)
    y -= 22

    # ---- Sections ----
    section_title_size = 14
    body_size = 11
    line_height = 1.7  # multiplier
    section_gap = 18
    para_gap = 6

    for header, lines in SECTIONS:
        # Section header (skip rendering for the synthetic intro marker)
        if header != "__intro__":
            # Page break check
            if y < MARGIN_BOTTOM + section_title_size + body_size * 3:
                c.showPage()
                c.setFillColor(CREAM)
                c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
                y = PAGE_H - MARGIN_TOP

            c.setFillColor(TERRACOTTA)
            draw_rtl(c, header, RIGHT_EDGE, y, "Arial-Bold", section_title_size)
            y -= section_title_size * line_height

        # Body lines
        c.setFillColor(INK)
        for raw in lines:
            if not raw.strip():
                y -= body_size * 0.7
                continue

            # Wrap if longer than usable width
            wrapped = wrap_to_width(c, raw, "Arial", body_size, USABLE_W)
            for wline in wrapped:
                # Page break check
                if y < MARGIN_BOTTOM + body_size:
                    c.showPage()
                    c.setFillColor(CREAM)
                    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
                    y = PAGE_H - MARGIN_TOP
                    c.setFillColor(INK)

                draw_rtl(c, wline, RIGHT_EDGE, y, "Arial", body_size)
                y -= body_size * line_height

            y -= para_gap

        y -= section_gap - para_gap

    c.save()
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
