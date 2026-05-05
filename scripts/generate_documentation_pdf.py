"""Generate the comprehensive project documentation PDF in Arabic.

15-25 page technical handbook for the Arabic Restaurant Complaints Classifier.
RTL Arabic prose with English ML terms inline, code snippets in Consolas,
ASCII architecture diagrams, terracotta section headers.

Output: documentation.pdf at project root.
"""
from __future__ import annotations

import io
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
OUTPUT = ROOT / "documentation.pdf"

# ---- Font registration ----

ARIAL_REG = Path("C:/Windows/Fonts/arial.ttf")
ARIAL_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
ARIAL_ITALIC = Path("C:/Windows/Fonts/ariali.ttf")
CONSOLAS_REG = Path("C:/Windows/Fonts/consola.ttf")
CONSOLAS_BOLD = Path("C:/Windows/Fonts/consolab.ttf")

for f in (ARIAL_REG, ARIAL_BOLD, CONSOLAS_REG):
    if not f.exists():
        raise SystemExit(f"Required font not found: {f}")

pdfmetrics.registerFont(TTFont("Arial", str(ARIAL_REG)))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(ARIAL_BOLD)))
if ARIAL_ITALIC.exists():
    pdfmetrics.registerFont(TTFont("Arial-Italic", str(ARIAL_ITALIC)))
pdfmetrics.registerFont(TTFont("Consolas", str(CONSOLAS_REG)))
if CONSOLAS_BOLD.exists():
    pdfmetrics.registerFont(TTFont("Consolas-Bold", str(CONSOLAS_BOLD)))

# ---- Design tokens (matches the Space's palette) ----

CREAM = HexColor("#F5EFE6")
PAPER = HexColor("#EEE6D7")
INK = HexColor("#2D211A")
INK_MUTED = HexColor("#5C4F45")
TERRACOTTA = HexColor("#A14828")
TERRACOTTA_LIGHT = HexColor("#C75D3D")
OLIVE = HexColor("#5F6845")
BORDER = HexColor("#D9CFC0")

# ---- Page geometry ----

PAGE_W, PAGE_H = A4
MARGIN_X = 22 * mm
MARGIN_TOP = 22 * mm
MARGIN_BOTTOM = 22 * mm
USABLE_W = PAGE_W - 2 * MARGIN_X
RIGHT_EDGE = PAGE_W - MARGIN_X
LEFT_EDGE = MARGIN_X


# ---- Arabic shaping ----

def shape(text: str) -> str:
    if not text:
        return ""
    return get_display(arabic_reshaper.reshape(text))


# ---- Drawing primitives ----

def draw_rtl(c: canvas.Canvas, text: str, x_right: float, y: float, font: str, size: float, color=INK):
    if not text.strip():
        return
    shaped = shape(text)
    width = c.stringWidth(shaped, font, size)
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x_right - width, y, shaped)


def draw_ltr(c: canvas.Canvas, text: str, x_left: float, y: float, font: str, size: float, color=INK):
    if not text:
        return
    c.setFont(font, size)
    c.setFillColor(color)
    c.drawString(x_left, y, text)


def fill_page_background(c: canvas.Canvas):
    c.setFillColor(CREAM)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)


def draw_page_number(c: canvas.Canvas, page_num: int):
    c.setFont("Arial", 8)
    c.setFillColor(INK_MUTED)
    c.drawString(MARGIN_X, 12 * mm, f"{page_num}")
    c.drawRightString(PAGE_W - MARGIN_X, 12 * mm, "Arabic Restaurant Complaints Classifier")


def wrap_to_width(c: canvas.Canvas, text: str, font: str, size: float, max_w: float) -> list[str]:
    """Word-wrap based on shaped width."""
    words = text.split(" ")
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        candidate = (" ".join(current + [w])).strip()
        if c.stringWidth(shape(candidate), font, size) <= max_w or not current:
            current.append(w)
        else:
            lines.append(" ".join(current))
            current = [w]
    if current:
        lines.append(" ".join(current))
    return lines


# ---- Page-flow state ----

class PageFlow:
    """Manages cursor position, page breaks, page numbering.

    The title page is page 0 (unnumbered, only shows cover content). Numbered
    pages start at 1 with section 1.
    """

    def __init__(self, c: canvas.Canvas):
        self.c = c
        self.page_num = 0  # title page is unnumbered
        self.y = PAGE_H - MARGIN_TOP
        fill_page_background(c)

    def new_page(self):
        # Only stamp page number on numbered pages (skip title page).
        if self.page_num >= 1:
            draw_page_number(self.c, self.page_num)
        self.c.showPage()
        self.page_num += 1
        fill_page_background(self.c)
        self.y = PAGE_H - MARGIN_TOP

    def need(self, mm_required: float):
        """Page-break if less than mm_required vertical space remains."""
        if self.y - mm_required * mm < MARGIN_BOTTOM + 14 * mm:
            self.new_page()

    def advance(self, points: float):
        self.y -= points

    def finalize(self):
        draw_page_number(self.c, self.page_num)


# ---- Higher-level rendering ----

def render_section_header(flow: PageFlow, number: str, title_ar: str, title_en: str):
    # Start on a fresh page — but only if we aren't already at the top of one.
    # (Otherwise a section that ended near a page boundary would create a blank page.)
    if flow.y < PAGE_H - MARGIN_TOP - 6:
        flow.new_page()

    # Number eyebrow
    flow.c.setFont("Arial-Bold", 9)
    flow.c.setFillColor(TERRACOTTA_LIGHT)
    flow.c.drawRightString(RIGHT_EDGE, flow.y, number)
    flow.advance(14)

    # Arabic title
    draw_rtl(flow.c, title_ar, RIGHT_EDGE, flow.y, "Arial-Bold", 22, TERRACOTTA)
    flow.advance(28)

    # English subtitle
    flow.c.setFont("Arial", 10)
    flow.c.setFillColor(INK_MUTED)
    flow.c.drawRightString(RIGHT_EDGE, flow.y, title_en)
    flow.advance(8)

    # Hairline divider
    flow.c.setStrokeColor(BORDER)
    flow.c.setLineWidth(0.6)
    flow.c.line(LEFT_EDGE, flow.y, RIGHT_EDGE, flow.y)
    flow.advance(20)


def render_paragraph(flow: PageFlow, text: str, *, indent_first: bool = False):
    body_size = 10.5
    line_h = body_size * 1.75
    flow.need(line_h * 2 / mm)

    flow.c.setFont("Arial", body_size)
    flow.c.setFillColor(INK)

    lines = wrap_to_width(flow.c, text, "Arial", body_size, USABLE_W)
    for line in lines:
        if flow.y < MARGIN_BOTTOM + line_h:
            flow.new_page()
        draw_rtl(flow.c, line, RIGHT_EDGE, flow.y, "Arial", body_size, INK)
        flow.advance(line_h)
    flow.advance(6)  # paragraph gap


def render_subheading(flow: PageFlow, text: str):
    flow.need(20)
    flow.advance(8)
    draw_rtl(flow.c, text, RIGHT_EDGE, flow.y, "Arial-Bold", 13, INK)
    flow.advance(20)


def render_bullet(flow: PageFlow, text: str):
    body_size = 10.5
    line_h = body_size * 1.7
    flow.need(line_h * 2 / mm)

    # Render as RTL: text first then bullet on the right (after text in RTL = left visually)
    full = "•  " + text
    lines = wrap_to_width(flow.c, full, "Arial", body_size, USABLE_W - 8)
    for i, line in enumerate(lines):
        if flow.y < MARGIN_BOTTOM + line_h:
            flow.new_page()
        draw_rtl(flow.c, line, RIGHT_EDGE - 8, flow.y, "Arial", body_size, INK)
        flow.advance(line_h)
    flow.advance(2)


def render_code_block(flow: PageFlow, code: str, *, lang_hint: str = "python", caption: str = ""):
    """Render a monospace code block with a tinted background."""
    code_size = 8.2
    line_h = code_size * 1.45
    pad = 10
    code_lines = code.split("\n")
    block_h = line_h * len(code_lines) + pad * 2
    needed_mm = (block_h + 16) / mm
    flow.need(needed_mm)

    if caption:
        flow.c.setFont("Arial", 8.5)
        flow.c.setFillColor(INK_MUTED)
        flow.c.drawRightString(RIGHT_EDGE, flow.y, f"// {caption}")
        flow.advance(12)

    # Background rectangle
    bg_top = flow.y + 4
    bg_bottom = flow.y + 4 - block_h
    flow.c.setFillColor(PAPER)
    flow.c.setStrokeColor(BORDER)
    flow.c.setLineWidth(0.5)
    flow.c.rect(LEFT_EDGE, bg_bottom, USABLE_W, block_h, fill=1, stroke=1)

    # Code lines (LTR, monospace)
    text_y = flow.y - pad
    flow.c.setFont("Consolas", code_size)
    flow.c.setFillColor(INK)
    for line in code_lines:
        flow.c.drawString(LEFT_EDGE + pad, text_y, line[:120])  # truncate over-long lines
        text_y -= line_h

    flow.advance(block_h + 14)


def render_ascii_diagram(flow: PageFlow, diagram: str, caption: str = ""):
    """Render an ASCII-art diagram in monospace, preserving spacing."""
    code_size = 7.2
    line_h = code_size * 1.25
    pad = 12
    lines = diagram.split("\n")
    block_h = line_h * len(lines) + pad * 2
    needed_mm = (block_h + 16) / mm
    flow.need(needed_mm)

    if caption:
        draw_rtl(flow.c, caption, RIGHT_EDGE, flow.y, "Arial-Bold", 10, INK)
        flow.advance(16)

    bg_top = flow.y + 4
    bg_bottom = flow.y + 4 - block_h
    flow.c.setFillColor(PAPER)
    flow.c.setStrokeColor(BORDER)
    flow.c.setLineWidth(0.5)
    flow.c.rect(LEFT_EDGE, bg_bottom, USABLE_W, block_h, fill=1, stroke=1)

    text_y = flow.y - pad
    flow.c.setFont("Consolas", code_size)
    flow.c.setFillColor(INK)
    for line in lines:
        flow.c.drawString(LEFT_EDGE + pad, text_y, line)
        text_y -= line_h

    flow.advance(block_h + 14)


def render_callout(flow: PageFlow, text: str):
    """Italic callout / pull-quote in olive."""
    body_size = 10
    line_h = body_size * 1.7
    pad = 10
    lines = wrap_to_width(flow.c, text, "Arial", body_size, USABLE_W - pad * 2 - 4)
    block_h = line_h * len(lines) + pad * 2
    flow.need((block_h + 12) / mm)

    bg_top = flow.y + 4
    bg_bottom = flow.y + 4 - block_h
    flow.c.setFillColor(HexColor("#F0E9DC"))
    flow.c.setStrokeColor(OLIVE)
    flow.c.setLineWidth(0)
    flow.c.rect(LEFT_EDGE, bg_bottom, USABLE_W, block_h, fill=1, stroke=0)
    # Olive accent strip on the right (RTL — leading edge)
    flow.c.setFillColor(OLIVE)
    flow.c.rect(RIGHT_EDGE - 3, bg_bottom, 3, block_h, fill=1, stroke=0)

    text_y = flow.y - pad - line_h * 0.3
    for line in lines:
        draw_rtl(flow.c, line, RIGHT_EDGE - pad - 6, text_y, "Arial", body_size, INK_MUTED)
        text_y -= line_h
    flow.advance(block_h + 14)


# ---- Title page ----

def render_title_page(flow: PageFlow):
    fill_page_background(flow.c)
    # Vertical center the title block
    cy = PAGE_H * 0.62

    # English eyebrow
    flow.c.setFont("Arial-Bold", 9)
    flow.c.setFillColor(TERRACOTTA_LIGHT)
    flow.c.drawRightString(RIGHT_EDGE, cy + 80, "PROJECT DOCUMENTATION  ·  TECHNICAL HANDBOOK")

    # Arabic title (large)
    draw_rtl(flow.c, "توثيق المشروع", RIGHT_EDGE, cy + 50, "Arial-Bold", 36, INK)

    # Arabic subtitle
    draw_rtl(flow.c, "تصنيف شكاوى المطاعم العربية", RIGHT_EDGE, cy + 14, "Arial-Bold", 18, TERRACOTTA)

    # English subtitle
    flow.c.setFont("Arial", 11)
    flow.c.setFillColor(INK_MUTED)
    flow.c.drawRightString(RIGHT_EDGE, cy - 8, "Arabic Restaurant Complaints Classifier")

    # Hairline
    flow.c.setStrokeColor(BORDER)
    flow.c.setLineWidth(0.6)
    flow.c.line(LEFT_EDGE, cy - 28, RIGHT_EDGE, cy - 28)

    # Description block
    draw_rtl(
        flow.c,
        "كل ما تحتاج معرفته عن المشروع: من البداية إلى النشر",
        RIGHT_EDGE, cy - 50, "Arial", 12, INK,
    )
    draw_rtl(
        flow.c,
        "الأدوات، القرارات، المعمارية، PySarf، والأرقام",
        RIGHT_EDGE, cy - 70, "Arial", 12, INK_MUTED,
    )

    # Footer
    flow.c.setFont("Arial", 9)
    flow.c.setFillColor(INK_MUTED)
    flow.c.drawString(MARGIN_X, MARGIN_BOTTOM, "Made by the NLP team at AI Club")
    flow.c.drawRightString(RIGHT_EDGE, MARGIN_BOTTOM, "Feras Madkhali")

    # Force fresh page after the title — section 1 starts at the top of page 1
    flow.new_page()


# ---- Sections ----

SECTION_1_INTRO = [
    ("h2", "ما هذا المشروع؟"),
    ("p", "هذا المشروع نموذج تعلّم آلي عربي يصنّف شكاوى المطاعم إلى ٨ فئات قابلة للتصرّف "
          "(جودة الطعام، التوصيل، خدمة الموظفين، النظافة، السعر والقيمة، دقة الطلب، "
          "وقت الانتظار، عامة). الهدف ليس مجرّد تصنيف نصوص، بل بناء أداة عملية تساعد "
          "فرق التشغيل في المطاعم السعودية على فرز الشكاوى بسرعة وتوجيهها للقسم المناسب."),
    ("p", "بُني المشروع من قِبل فريق NLP في AI Club كمشروع portfolio. الفئة المستهدفة "
          "من المراجعين: مسؤولو التوظيف، المراجعون التقنيون، وزملاء العمل في مجتمع NLP "
          "العربي. الأداء النهائي: ٩٥٫٠٥٪ دقّة على مجموعة اختبار مستقلّة من ١٣٬٩٨٦ مراجعة "
          "حقيقية، وكل الفئات الثماني فوق ٨٠٪ F1."),
    ("h2", "لماذا اللهجة السعودية والخليجية؟"),
    ("p", "البيانات المستخدمة في التدريب جاءت من تطبيقات التوصيل السعودية الكبرى "
          "(HungerStation، Jahez، Mrsool، Talabat). هذا اختيار مقصود: التخصص في لهجة "
          "محدّدة يعطي دقّة عالية على المستخدمين الفعليين. النموذج لن يعمل بنفس الكفاءة "
          "على اللهجات الأخرى (القياسات: السعودية ٦٧٪، الشامية ٦٠٪، المصرية والفصحى ٥٠٪)."),
    ("h2", "ما الذي يميّز هذا المشروع؟"),
    ("p", "ثلاث طبقات تجعله مختلفاً عن مجرد classifier عادي:"),
    ("bullet", "ensemble من ٤ نماذج BERT عربية (CAMeLBERT-mix بـ seedين، MARBERT، AraBERTv02) "
               "بدلاً من نموذج واحد، ما يعطي تنوّعاً في القرار."),
    ("bullet", "طبقة keyword rescue دفاعية تصحّح حالات فشل النموذج المحدّدة (مثل: شكاوى "
               "الحمام التي يصنّفها النموذج خطأً كجودة طعام)."),
    ("bullet", "طبقة aspect extraction تُظهر للمستخدم أي كلمات في شكواه دلّت على كل فئة، "
               "أي شفافية حقيقية في القرار، وليس مجرّد رقم."),
]

SECTION_2_JOURNEY = [
    ("h2", "البداية: TF-IDF كأساس"),
    ("p", "بدأنا بالخطّة المتفق عليها مع الفريق: نموذج TF-IDF + LinearSVC على ٩ فئات. "
          "كان قراراً سليماً للموارد المتاحة، وأعطانا فهماً للأساسيات. النتيجة: ٨٩٫٨٪ دقّة "
          "إجمالية، لكن F1 لكل فئة كان كارثياً (الجو والمكان ٢٤٪، دقة الطلب ٢٢٪)."),
    ("h2", "الانتقال إلى BERT"),
    ("p", "لمّا اصطدمنا بجدار الفئات الصغيرة، انتقلنا إلى transfer learning مع نماذج "
          "CAMeLBERT-mix العربية المُدرَّبة مسبقاً. النتيجة قفزت إلى ٩٣٫٢٪ دقّة بنموذج واحد، "
          "و F1 الفئات الصغيرة تحسّن بشكل ملحوظ."),
    ("h2", "توسيع البيانات"),
    ("p", "أدركنا أن البيانات هي العنق الزجاجة الحقيقي. بنينا scraper لتطبيقات التوصيل "
          "السعودية، أضفنا ١٤٬٧٨٩ مراجعة حقيقية جديدة، ورفعنا فئة دقة الطلب من ١٣٧ إلى "
          "٧١٤ عيّنة. أضفنا أيضاً synthetic templates و pseudo-labels."),
    ("h2", "Bake-off وتشكيل الـ ensemble"),
    ("p", "اختبرنا ٥ معماريات: CAMeLBERT-mix بـ seedين، CAMeLBERT-da، MARBERT، AraBERTv02، "
          "و XLM-R. اخترنا أفضل ٤ بناءً على macro F1، واستبعدنا CAMeLBERT-da (التنظيف يحذف "
          "المؤشرات اللهجية فيفقد ميزته) و XLM-R (multilingual أضعف من المتخصّص بالعربية)."),
    ("h2", "حذف فئة الجو والمكان (قرار حاسم)"),
    ("p", "فئة الجو والمكان لم تتجاوز ٣٥٪ F1 مهما حاولنا. تدقيق ١٧١ عيّنة validation+test "
          "كشف أن ٢ فقط منها تتعلّق فعلياً بالجو والمكان (أقل من ١٪). البقية أخطاء تصنيف. "
          "حذفنا الفئة، وانتقلنا إلى ٨ فئات. النتيجة: قفزة ٧٫٧٪ في macro F1."),
    ("h2", "تعزيز EDA + التقييم الصارم"),
    ("p", "EDA augmentation رفع الفئات الصغيرة (دقة الطلب وعامة) إلى أعلى من ٨٥٪ F1. "
          "أضفنا bootstrap CI، calibration، perturbation robustness، و cross-dialect canary "
          "لإثبات الموثوقية وليس مجرّد ادّعائها."),
    ("h2", "النشر"),
    ("p", "ثلاث واجهات للنشر: FastAPI كـ REST endpoint، Gradio محلّي للتطوير، و HuggingFace "
          "Space للديمو العام. النموذج الفردي الأخفّ متاح أيضاً على HF Hub."),
    ("h2", "طبقة الإنقاذ بالكلمات المفتاحية (keyword rescue)"),
    ("p", "بعد النشر اكتشفنا أخطاء حقيقية في حالات معيّنة (شكوى الحمام تُصنَّف كجودة طعام، "
          "انتظار طويل يُصنَّف كجودة طعام). أضفنا طبقة دفاعية تتفعّل فقط عند ظهور عبارات "
          "واضحة. التكلفة: ٠٫٩٪ على الـ test، المقابل: ١٠٠٪ نجاح على audit يدوي."),
    ("h2", "PySarf و طبقة الـ aspect extraction"),
    ("p", "آخر إضافة: استخدمنا مكتبة PySarf (engine عربي مورفولوجي يدعم اللهجة الخليجية) "
          "لمطابقة الـ stems بدل الـ surface forms. الآن كلمة \"المندوبين\" تُطابِق "
          "\"المندوب\" في القاموس. وأضفنا طبقة \"كيف فهمت شكواك\" تُظهر للمستخدم أي كلمات "
          "في النص دلّت على أي فئة."),
]

SECTION_3_TOOLS = [
    ("h2", "PyTorch + Transformers"),
    ("p", "PyTorch هو الإطار الأساسي للتعلّم العميق. اخترناه لأنه السائد في NLP الأكاديمي "
          "والصناعي، ولأن HuggingFace Transformers مبنيّة عليه. مكتبة Transformers تعطينا "
          "وصولاً مباشراً للنماذج المُدرَّبة مسبقاً مع AutoTokenizer و "
          "AutoModelForSequenceClassification."),
    ("h2", "نماذج BERT العربية"),
    ("p", "اخترنا ثلاث نماذج عربية مُدرَّبة مسبقاً لتنوّع التمثيل:"),
    ("bullet", "CAMeLBERT-mix من CAMeL Lab — مدرَّب على مزيج من الفصحى واللهجات. الأقوى "
               "في اختباراتنا. استخدمناه بـ seedين (٤٢ و ٢٠٢٤) لزيادة التنوّع."),
    ("bullet", "MARBERT من UBC-NLP — مدرَّب على Twitter العربي، لذا قويّ في اللهجات "
               "غير الرسمية والمختصرة، يكمّل CAMeLBERT."),
    ("bullet", "AraBERTv02 من aubmindlab — مختبر آخر، يعطي تنوّعاً معمارياً في الـ ensemble."),
    ("h2", "FastAPI"),
    ("p", "خادم REST سريع مبنيّ على Starlette. اخترناه لـ async support، توليد OpenAPI "
          "تلقائياً، و lifespan manager المناسب لتحميل النموذج مرّة واحدة عند البدء. "
          "Endpointاتنا: /predict، /predict_batch، /healthz، /readyz."),
    ("h2", "Gradio"),
    ("p", "إطار سريع لبناء واجهات ML تفاعلية. استخدمناه في نسختين: محلّي للتطوير، "
          "وعلى HuggingFace Spaces للديمو العام. تحكّمنا في التصميم بـ CSS مخصَّص "
          "بدلاً من الـ themes الافتراضية للحصول على الهوية البصرية الدافئة (cream + "
          "terracotta)."),
    ("h2", "Almarai (الخط)"),
    ("p", "خط عربي إنساني يدعم العربية واللاتينية في عائلة واحدة. اخترناه على Tajawal "
          "(الأكثر شيوعاً) لأن له طابعاً أدفأ يناسب \"الضيافة\" المقصودة في الهوية. "
          "محمَّل من Google Fonts CDN، لا يحتاج تضمين ملفات."),
    ("h2", "scikit-learn و matplotlib"),
    ("p", "scikit-learn لحساب metricات التقييم (accuracy، F1، macro F1، classification "
          "report). matplotlib لتوليد رسوم F1 لكل فئة و evolution chart للنماذج المختلفة. "
          "كلاهما نقطة ارتكاز قياسية في أي pipeline ML."),
    ("h2", "Helsinki-NLP MarianMT (تجربة لم تنجح)"),
    ("p", "جرّبنا back-translation augmentation للفئة العامة باستخدام نماذج Helsinki-NLP "
          "(عربية → إنجليزية → عربية) لإنشاء صياغات بديلة. النتيجة: تحسّن طفيف في validation "
          "لكن regress واضح في test. أوقفنا التجربة. التضمين: scripts/augment_general.py "
          "محفوظ كمرجع."),
    ("h2", "pytest"),
    ("p", "إطار الاختبار. مجموعة الاختبارات تشمل: integrity للبيانات، unit للـ logic "
          "النقي بدون نموذج، Hypothesis property tests، determinism، و regression gate "
          "على ٢٠٠ صف من الـ test set."),
    ("h2", "PySarf"),
    ("p", "engine عربي مورفولوجي من Rashidbm، يدعم اللهجة الخليجية. يقدّم analyze() الذي "
          "يعطي root و stem و prefixes و suffixes لأي كلمة عربية. استخدمناه لمطابقة "
          "الـ stems في طبقة aspect extraction. تفاصيله الكاملة في القسم المخصَّص له."),
]

SECTION_4_ARCH = [
    ("h2", "تدفّق البيانات"),
    ("p", "كل تنبّؤ يمرّ بهذه المراحل:"),
    ("bullet", "النص الخام يدخل من المستخدم"),
    ("bullet", "الـ clean() يطبَّق: حذف tashkeel، توحيد الـ alef/ya/ta-marbuta، حذف الرموز "
               "غير العربية، تطبيع أحرف اللهجة الخليجية (پ، چ، گ، ک، ی)"),
    ("bullet", "الـ tokenizer يحوّل النص إلى tokens (max_length=192)"),
    ("bullet", "كل نموذج من الـ ٤ ينتج softmax probabilities"),
    ("bullet", "نأخذ المتوسّط الموحَّد للاحتمالات (uniform softmax average)"),
    ("bullet", "طبقة keyword rescue تتدخّل إذا ظهرت عبارة لا لبس فيها (مثل: \"الحمام\" → نظافة)"),
    ("bullet", "Temperature scaling يطبَّق عند الحاجة (T=1.523 من المعايرة)"),
    ("bullet", "نأخذ Top-3 categories مع الـ confidence"),
    ("bullet", "طبقة aspect extraction تستخرج الكلمات الدالّة على كل فئة"),
    ("bullet", "render_result يبني HTML المُعرَض للمستخدم"),
    ("h2", "الواجهات الثلاث للنشر"),
    ("p", "نفس منطق الاستدلال موجود في ثلاث طبقات نشر:"),
    ("bullet", "FastAPI (app/api.py) — REST endpoint للاستهلاك البرمجي. lifespan loader، "
               "CORS middleware، PII scrubbing للـ logs، rate limiting ٦٠ req/min."),
    ("bullet", "Gradio محلّي (app/space_app.py) — للتطوير على الجهاز. يستخدم "
               "EnsembleClassifier المحلّي."),
    ("bullet", "HuggingFace Space (hf_space/app.py) — الديمو العام. يحمّل نسخة single "
               "model من HuggingFace Hub لتوفير الموارد على الـ free tier، ويطبّق نفس "
               "rescue + aspect extraction inline."),
]

SECTION_5_DIAGRAM = [
    ("h2", "مخطط معماري للمشروع"),
    ("p", "هذا التدفّق الكامل من المستخدم إلى الواجهات الثلاث:"),
    ("ascii", """
   ~98K Arabic complaints (production + Saudi delivery apps)
                          |
                          v
              +--------------------------+
              |  Data pipeline   src/    |
              |  scrape | clean |        |
              |  augment | split         |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |  Bake-off       src/     |
              |  CAMeLBERT-mix | MARBERT |
              |  AraBERTv02 | XLM-R      |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |  4-model ensemble        |
              |  uniform softmax avg     |
              |  + temperature scaling   |
              |  models/ensemble_final/  |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |  Keyword rescue layer    |
              |  app/ensemble_inference  |
              |  (or inline in hf_space) |
              +------------+-------------+
                           |
                           v
              +--------------------------+
              |  Aspect extraction       |
              |  PySarf stems +          |
              |  phrase matching         |
              +------------+-------------+
                           |
              +------------+------------+
              v            v            v
          FastAPI      Gradio        Colab
          app/api.py   hf_space/     notebooks/
"""),
    ("p", "الفكرة الأساسية: كل واجهة تستهلك نفس الـ pipeline. الفرق الوحيد أن HuggingFace "
          "Space يستخدم single model من HF Hub بدل تحميل ٤ نماذج محلّياً (لقيود الذاكرة في "
          "الـ free tier)، لكنه يطبّق نفس rescue + aspect extraction logic."),
]

SECTION_6_CODE = [
    ("h2", "الـ clean() — التطبيع النصّي"),
    ("p", "نقطة البداية لأي تنبّؤ. تأخذ نصاً عربياً خاماً وترجع نصاً نظيفاً جاهزاً للـ "
          "tokenizer. تشمل توحيد أحرف اللهجة الخليجية المتأثّرة بالفارسية:"),
    ("code", '''def clean(text: str) -> str:
    if not text:
        return ""
    t = TASHKEEL.sub("", text)
    t = t.translate(str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        # Gulf/Persian-influenced chars in Saudi social media
        "پ": "ب", "چ": "ج", "گ": "ك", "ک": "ك", "ی": "ي",
    }))
    t = NON_ARABIC.sub(" ", t)
    return WHITESPACE.sub(" ", t).strip().lower()'''),
    ("h2", "الـ apply_rescue() — طبقة الإنقاذ بالكلمات المفتاحية"),
    ("p", "تتدخّل فقط عند ظهور عبارة واضحة ١٠٠٪ تدلّ على فئة معيّنة. لا تتفعّل في الحالات "
          "العاديّة، فقط للأخطاء المعروفة:"),
    ("code", '''RESCUE_RULES: list[tuple[str, list[str]]] = [
    ("النظافة", ["الحمام", "تواليت", "ذباب", "صراصير"]),
    ("وقت الانتظار", [
        "انتظرت ساعه", "انتظرت ساعتين", "ساعه كامله في المطعم",
        "انتظرنا ساعه", "انتظرنا ساعتين",
    ]),
    ("التوصيل", ["ضاع الطلب", "المندوب تاخر", "المندوب ما رد"]),
    ("جودة الطعام", ["الطبخ", "اللحم محروق", "بدون طعم", "الاكل بايخ"]),
]

def apply_rescue(probs, cleaned_text, rescue_floor=0.55):
    out = probs.copy()
    for cat, phrases in RESCUE_RULES:
        if cat not in CATEGORY_TO_ID:
            continue
        if any(p in cleaned_text for p in phrases):
            cat_idx = CATEGORY_TO_ID[cat]
            current_max = float(out.max())
            out[cat_idx] = max(out[cat_idx], current_max + 0.05, rescue_floor)
    s = out.sum()
    return out / s if s > 0 else out'''),
    ("h2", "الـ extract_aspects() — استخراج الجوانب"),
    ("p", "الطبقة التفسيرية. تأخذ النص النظيف وتُعيد قائمة الـ matches (موضع، فئة، عبارة) "
          "بطريقتين: phrase matching، و stem matching بـ PySarf:"),
    ("code", '''def extract_aspects(cleaned_text):
    raw_matches = []

    # Pass 1: phrase matching (with prefix-tolerant boundary)
    for aspect, phrases in _PHRASES_BY_ASPECT.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            start = 0
            while True:
                idx = cleaned_text.find(phrase, start)
                if idx == -1: break
                end = idx + len(phrase)
                if (_is_word_boundary_start(cleaned_text, idx)
                    and _is_word_boundary_end(cleaned_text, end)):
                    raw_matches.append((idx, end, aspect, phrase))
                start = idx + 1

    # Pass 2: stem matching (PySarf only)
    if _SARF is not None:
        for word in cleaned_text.split():
            stem = _SARF.analyze(word).stem
            if stem in _STEM_TO_ASPECT:
                # ... append match ...

    # Greedy longest-first dedup
    raw_matches.sort(key=lambda m: -(m[1] - m[0]))
    final, covered = [], set()
    for m in raw_matches:
        if not any(i in covered for i in range(m[0], m[1])):
            final.append(m)
            covered.update(range(m[0], m[1]))
    return final'''),
    ("h2", "الـ is_multi_aspect() — كشف تعدّد الجوانب"),
    ("p", "اكتشاف بسيط: إذا كان الـ top-1 أقل من ٨٥٪ والـ top-2 أكثر من ١٥٪، فالشكوى متعدّدة "
          "الجوانب وتُعرَض الفئتان معاً بنفس الوزن البصري:"),
    ("code", '''def is_multi_aspect(top: list[tuple[str, float]]) -> bool:
    """Top-1 < 85% AND top-2 > 15% means competing signals.
    Threshold tuned against the audit set after rescue rebalances scores.
    """
    if len(top) < 2:
        return False
    return top[0][1] < 0.85 and top[1][1] > 0.15'''),
    ("h2", "الـ render_general_fallback() — إعادة صياغة فئة عامة"),
    ("p", "عندما يصبح \"عامة\" هو الـ top-1، نعيد الصياغة كـ \"لم يُحدَّد جانب معيّن\" ونُظهِر "
          "أعلى فئتين محدَّدتين كاحتمالات بديلة، مع تلميح للمستخدم ليكون أكثر تحديداً:"),
    ("code", '''def render_general_fallback(top):
    specific = [(c, s) for c, s in top if c != "عامة"][:2]
    rows = [...]  # render specific categories with low confidence

    return (
        '<div class="general-fallback">'
        '  <strong>لم يحدد جانب معين</strong>'
        '  <span>no specific aspect detected</span>'
        '  <div class="hint">اذكر جانبا محددا...</div>'
        f'  <div class="rail">قد تكون عن: {rows}</div>'
        '</div>'
    )'''),
]

SECTION_7_PYSARF = [
    ("h2", "ما هو PySarf؟"),
    ("p", "PySarf مكتبة Python للتحليل المورفولوجي للعربية، من تطوير Rashidbm. يدعم "
          "اللهجة الخليجية بشكل خاصّ، ويعمل بـ NumPy فقط بدون نماذج تعلّم آلي. يقدّم "
          "تحليلاً صرفياً سريعاً (~٠٫١٥ ms لكل كلمة)."),
    ("h2", "ماذا يقدّم لمشروعنا؟"),
    ("p", "ثلاثة أشياء أساسية:"),
    ("bullet", "extract_root() — يستخرج الجذر العربي (مثلاً: \"المكتبات\" → \"كتب\")"),
    ("bullet", "stem() — يحذف الـ affixes ويُعيد الـ stem (\"والمندوب\" → \"مندوب\")"),
    ("bullet", "تطبيع أحرف اللهجة الخليجية (پ→ب، چ→ج، گ→ك، ی→ي)"),
    ("h2", "كيف دمجناه؟"),
    ("p", "في hf_space/app.py نحاول استيراده، وإذا فشل نسقط بأمان إلى surface matching:"),
    ("code", '''try:
    from pysarf import PySarf
    _SARF = PySarf(dialect="gulf")
except Exception as e:
    print(f"PySarf unavailable, falling back: {e}")
    _SARF = None'''),
    ("p", "عند تحميل القاموس نحسب stem لكل كلمة مفردة في الـ vocabulary مرّة واحدة:"),
    ("code", '''def _build_vocab_indices():
    phrases_by_aspect = {}
    stem_to_aspect = {}
    for aspect, items in ASPECT_VOCAB.items():
        for item in items:
            words = item.split()
            if len(words) > 1:
                phrases_by_aspect.setdefault(aspect, []).append(item)
            else:
                phrases_by_aspect.setdefault(aspect, []).append(item)
                if _SARF is not None:
                    stem = _SARF.analyze(words[0]).stem
                    if stem and stem not in stem_to_aspect:
                        stem_to_aspect[stem] = aspect
    return phrases_by_aspect, stem_to_aspect'''),
    ("h2", "ما المشكلة التي حلّها؟"),
    ("p", "ثلاث مشاكل حقيقية ظهرت قبل دمجه:"),
    ("h3", "١. الجمع والتصريف"),
    ("p", "الكلمة \"المندوبين\" (جمع) لم تكن تُطابِق \"المندوب\" (مفرد) في القاموس، "
          "فكنّا نفقد الإشارة. الآن:"),
    ("ascii", """
Input:    والمندوبين تاخروا
Stem:     مندوب  -> مطابق "المندوب" في القاموس
Aspect:   التوصيل  detected
"""),
    ("h3", "٢. الـ prefixes (و، ف، ب، ل، ك، س)"),
    ("p", "كلمة \"بطعم\" (بـ + طعم) لم تكن تُطابِق \"طعم\" قبل أن أضفنا تساهلاً يدوياً. "
          "PySarf يحسبه أوتوماتيكياً:"),
    ("ascii", """
Input:    بطعم الاكل لذيذ
Analyze:  prefixes=("ب",) stem="طعم" root="طعم"
Match:    "طعم" in vocabulary -> food quality
"""),
    ("h3", "٣. منع الـ false positives"),
    ("p", "كلمة \"المطعم\" (restaurant) كانت تُطابِق \"طعم\" بالخطأ في الـ substring "
          "matching البسيط. PySarf يحلّها بشكل صحيح:"),
    ("ascii", """
المطعم  ->  prefix="ال"  stem="مطعم"  root="طعم"
طعم      ->  prefix=""    stem="طعم"   root="طعم"

Stems are different -> no false match!
"""),
    ("h2", "أداء PySarf"),
    ("p", "اختباراتنا: ٠٫١٥ ms لكل كلمة، أي ~١٫٣ ms للشكوى المتوسّطة (١٠ كلمات). هذا "
          "ضئيل مقارنة بـ inference الـ BERT (~٥٠ ms). إجمالاً: التكلفة لا تُذكَر، "
          "والمكسب في الدقّة كبير."),
    ("h2", "كيف يُحَلّ تعارض المطابقات؟"),
    ("p", "بعد جمع كل الـ matches من الـ phrase pass والـ stem pass، نطبّق greedy "
          "longest-match-first dedup: الترتيب من الأطول إلى الأقصر، نأخذ كل match لا "
          "يتعارض مع ما سبق:"),
    ("code", '''raw_matches.sort(key=lambda m: -(m[1] - m[0]))
final = []
covered = set()
for m in raw_matches:
    s, e = m[0], m[1]
    if any(i in covered for i in range(s, e)):
        continue
    final.append(m)
    covered.update(range(s, e))
final.sort(key=lambda m: m[0])'''),
    ("p", "الفائدة: الـ phrase الطويل \"المندوب تاخر\" يفوز على الـ stem الفرديّ \"مندوب\" "
          "عند التعارض. هذا يعطينا أكثر دلالة في الـ UI."),
    ("callout", "PySarf حوّلنا من keyword matching غبيّ إلى morphology-aware matching. "
                "هذا الفرق بين tool هاوي و tool احترافيّ في NLP العربي."),
]

SECTION_8_DECISIONS = [
    ("h2", "حذف فئة الجو والمكان"),
    ("p", "Audit يدويّ لـ ١٧١ عيّنة كشف أن ٢ منها فقط (١٫٢٪) تتعلّق فعلياً بالجو والمكان. "
          "البقية أخطاء تصنيف (٤٧٪ مصنّفة خطأً، ٤٠٪ متعدّدة الجوانب). القرار: نحذف الفئة "
          "بدلاً من ملاحقتها. النتيجة: قفزة ٧٫٧٪ في macro F1، و ٤٣ نقطة في min-class F1."),
    ("h2", "Trade-off: keyword rescue مقابل الدقّة الإجمالية"),
    ("p", "بعد بناء الـ rescue layer وقياس الأثر:"),
    ("bullet", "Pure model: ٩٥٫٠٥٪ على test set"),
    ("bullet", "With rescue: ٩٤٫١٢٪ (-٠٫٩٣٪)"),
    ("bullet", "Behavioral audit: من ٨٥٪ إلى ١٠٠٪ (+١٥٪)"),
    ("p", "الخسارة ٠٫٩٣٪ على benchmark إجمالي مقابل تصحيح أخطاء فادحة (شكوى الحمام تُصنَّف "
          "كجودة طعام). القرار: نقبل المقايضة. الأخطاء الواضحة على المستخدم أسوأ من ١٪ "
          "على رقم لا يراه."),
    ("h2", "إعادة صياغة فئة \"عامة\""),
    ("p", "عندما يكون \"عامة\" هو الـ top-1، نُعيد العرض كـ \"لم يُحدَّد جانب معيّن\" ونُظهِر "
          "أعلى فئتين محدَّدتين بثقة منخفضة. السبب: \"عامة\" لا تساعد المستخدم في توجيه "
          "الشكوى. الإعادة الصياغة تحوّلها إلى إشارة قابلة للتصرّف: \"كن أكثر تحديداً\"."),
    ("h2", "Temperature scaling"),
    ("p", "النموذج الخام كان confident أكثر من اللازم (over-confident). جرّبنا temperature "
          "scaling بـ T=1.523 (مُحسَّن على validation set بـ NLL minimization). النتيجة: "
          "خفض ECE من ٠٫٠٣٤ إلى ٠٫٠١٤، بدون تغيير في الـ top-1 predictions."),
    ("h2", "تعدّد الجوانب: عرض بصريّ خاصّ"),
    ("p", "بدلاً من عرض top-1 كأنه الإجابة، عند top-1 < ٨٥٪ و top-2 > ١٥٪ نعرض الفئتين "
          "بنفس الـ visual weight (terracotta للاثنين). هذا يعكس عدم يقين النموذج بدلاً "
          "من إخفائه."),
]

SECTION_9_PERFORMANCE = [
    ("h2", "الأرقام النهائية"),
    ("p", "على مجموعة اختبار مستقلّة من ١٣٬٩٨٦ مراجعة حقيقية، لم تُستخدم في التدريب:"),
    ("bullet", "Test accuracy: ٩٥٫٠٥٪ (pure model) / ٩٤٫١٢٪ (with rescue)"),
    ("bullet", "Macro F1: ٩٢٫٠٣٪"),
    ("bullet", "Min class F1: ٨٤٫٨٤٪ (فئة عامة)"),
    ("bullet", "Bootstrap 95% CI: [٩٤٫٧٠٪، ٩٥٫٤١٪]"),
    ("bullet", "Calibration ECE: ٠٫٠١٤ (بعد temperature scaling)"),
    ("h2", "F1 لكل فئة"),
    ("ascii", """
                          F1
  جودة الطعام         96.2%   #####################
  خدمة الموظفين       95.8%   ####################
  النظافة             95.3%   ####################
  السعر والقيمة       94.7%   ####################
  وقت الانتظار        91.9%   ##################
  التوصيل             90.1%   ##################
  دقة الطلب           86.7%   ################
  عامة                84.9%   ###############
"""),
    ("h2", "Behavioral audit"),
    ("p", "بـ ٣٤ حالة اختبار حقيقية (مفردة + متعدّدة الجوانب) كتبناها يدوياً:"),
    ("bullet", "Top-1 accuracy: ١٠٠٪ (٣٤/٣٤)"),
    ("bullet", "Top-2 coverage: ٩٧٪ (٣٣/٣٤)"),
    ("bullet", "Multi-aspect detection: ٧/٨"),
    ("p", "ثلاث حالات فشل أبلغ عنها المستخدم في وقت سابق (شكوى الحمام، انتظار طويل، "
          "شكوى عامّة) كلها تعمل الآن بشكل صحيح بعد طبقة الـ rescue."),
    ("h2", "ميزانية الأداء عند النشر"),
    ("p", "Inference latency على single GPU (RTX 4070): ~٤٫٤ ms. على CPU (HF Spaces): "
          "~٧٧ ms للـ ensemble، ~١٦ ms للنسخة الفردية. PySarf يضيف ~١٫٣ ms لكل شكوى "
          "وهو لا يُذكَر."),
]

SECTION_10_NEXT = [
    ("h2", "ما يستحقّ العمل عليه لاحقاً"),
    ("p", "ثلاث مسارات واضحة بترتيب القيمة:"),
    ("h2", "١. تدريب multi-label حقيقيّ"),
    ("p", "النموذج الحالي single-label: يختار فئة واحدة لكل شكوى. الشكاوى متعدّدة "
          "الجوانب (\"طعام بارد + موظف غير مهذّب\") لا تُعرَض بشكل صحيح إلا عبر heuristic "
          "في الـ UI. الحلّ الحقيقي: تحويل الـ head إلى ٧ binary classifiers مستقلّين "
          "+ confidence threshold. يحتاج ~٣٠٠-٥٠٠ مثال multi-label يدويّ، تكلفة ~$٥٠ "
          "أو يوم عمل من الفريق."),
    ("h2", "٢. Knowledge distillation"),
    ("p", "الـ ensemble (٤ نماذج) أبطأ ٤× وأكبر ٤× من النموذج الفردي. ندرّب CAMeLBERT-mix "
          "كـ student على soft labels من الـ ensemble (KL divergence loss). الهدف: نحتفظ "
          "بـ ٩٩٪ من دقّة الـ ensemble بسرعة الـ single model. مفيد للنشر الإنتاجي."),
    ("h2", "٣. مقارنة مع GPT-4o (commercial baseline)"),
    ("p", "نشغّل GPT-4o على ٢٠٠ عيّنة من الـ test set مع few-shot prompt، ونقارن. "
          "إمّا نتفوّق (قصّة رائعة)، أو نتأخّر بقليل (أيضاً قصّة جيّدة: \"open-source يقترب "
          "من commercial باستخدام أقلّ بكثير\"). تكلفة ~$١."),
    ("h2", "خارج الـ scope الحالي"),
    ("p", "هذه أفكار جيّدة لكن خارج نطاق المشروع portfolio:"),
    ("bullet", "نقل النموذج إلى لهجات أخرى (مصرية، شامية) — يتطلّب جمع بيانات منفصلة"),
    ("bullet", "Sentiment intensity (شدّة المشاعر) كـ dimension إضافيّ"),
    ("bullet", "Active learning loop لاستكشاف بيانات إنتاجية جديدة"),
]

SECTION_11_CLOSING = [
    ("h2", "الفريق"),
    ("p", "هذا المشروع جهد جماعي من فريق NLP في AI Club:"),
    ("bullet", "لانا — pipeline التنظيف clean()، نقطة الارتكاز في كل خطوة"),
    ("bullet", "خولة — التصنيفات اليدوية، قاعدة كل شيء"),
    ("bullet", "ريما ومحمد — قرارات الـ schema الأصلية والـ baseline trials"),
    ("bullet", "مشعل — جانب النشر"),
    ("bullet", "فراس مدخلي — قيادة الفريق، Phase 3 (التدريب)، التقييم، النشر"),
    ("h2", "شكر خاصّ لـ Rashid"),
    ("p", "PySarf من تطوير Rashidbm، ودون هذه المكتبة لما كانت طبقة aspect extraction "
          "بهذه الجودة. شكر خاص له على إتاحتها مفتوحة المصدر، وعلى دعم اللهجة الخليجية "
          "تحديداً."),
    ("h2", "شكر للنماذج المستخدمة"),
    ("p", "المشروع يقف على أكتاف عمالقة:"),
    ("bullet", "CAMeL Lab (NYU Abu Dhabi) — CAMeLBERT-mix"),
    ("bullet", "UBC-NLP — MARBERT"),
    ("bullet", "aubmindlab — AraBERTv02"),
    ("bullet", "Helsinki-NLP — opus-mt models (لتجربة BT الفاشلة)"),
    ("h2", "أين تجد المشروع"),
    ("p", "GitHub: github.com/FerasMad/NLP-complaints-system"),
    ("p", "HuggingFace Model: huggingface.co/FerasMad/arabic-complaints-classifier"),
    ("p", "Live Demo: huggingface.co/spaces/FerasMad/arabic-complaints-classifier"),
    ("callout", "هذا المشروع نُشر بترخيص MIT. خذ الكود، خذ النموذج، استخدمه في مشاريعك. "
                "إذا فادك، مذكور في README يكفي. النيّة: مساهمة في NLP العربي مفتوح المصدر."),
]

SECTIONS = [
    ("١", "مقدّمة", "Introduction", SECTION_1_INTRO),
    ("٢", "الرحلة", "The Journey", SECTION_2_JOURNEY),
    ("٣", "الأدوات", "Tools and Why", SECTION_3_TOOLS),
    ("٤", "المعمارية", "Architecture", SECTION_4_ARCH),
    ("٥", "مخطط معماري", "Architecture Diagram", SECTION_5_DIAGRAM),
    ("٦", "مقاطع من الكود", "Code Snippets", SECTION_6_CODE),
    ("٧", "PySarf", "PySarf Integration", SECTION_7_PYSARF),
    ("٨", "القرارات الكبرى", "Key Decisions", SECTION_8_DECISIONS),
    ("٩", "النتائج", "Performance", SECTION_9_PERFORMANCE),
    ("١٠", "ما بقي", "What's Next", SECTION_10_NEXT),
    ("١١", "خاتمة", "Closing", SECTION_11_CLOSING),
]


def render_section(flow: PageFlow, number: str, title_ar: str, title_en: str, blocks: list):
    render_section_header(flow, number, title_ar, title_en)
    for kind, *args in blocks:
        if kind == "h2":
            render_subheading(flow, args[0])
        elif kind == "h3":
            # Smaller subheading
            flow.advance(4)
            draw_rtl(flow.c, args[0], RIGHT_EDGE, flow.y, "Arial-Bold", 11, INK_MUTED)
            flow.advance(15)
        elif kind == "p":
            render_paragraph(flow, args[0])
        elif kind == "bullet":
            render_bullet(flow, args[0])
        elif kind == "code":
            render_code_block(flow, args[0])
        elif kind == "ascii":
            render_ascii_diagram(flow, args[0])
        elif kind == "callout":
            render_callout(flow, args[0])


def main() -> int:
    c = canvas.Canvas(str(OUTPUT), pagesize=A4)
    c.setTitle("توثيق المشروع - تصنيف شكاوى المطاعم العربية")
    c.setAuthor("Feras Madkhali / AI Club NLP")

    flow = PageFlow(c)

    # Title page
    render_title_page(flow)

    # All sections
    for number, title_ar, title_en, blocks in SECTIONS:
        render_section(flow, number, title_ar, title_en, blocks)

    flow.finalize()
    c.save()

    size_kb = OUTPUT.stat().st_size // 1024
    print(f"Wrote {OUTPUT}")
    print(f"  size: {size_kb} KB")
    print(f"  pages: {flow.page_num}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
