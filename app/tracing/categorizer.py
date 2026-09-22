"""Legal and conversational query categorizer for Law Copilot."""

import re
from typing import Optional


# Keywords mapped to Persian legal categories
CRIMINAL_KEYWORDS = {
    "قتل", "سرقت", "دزدی", "زنا", "لواط", "شرب خمر", "حدود", "حد", "قصاص",
    "دیه", "دیات", "تعزیر", "تعزیرات", "کلاهبرداری", "خیانت در امانت", "جرم",
    "مجازات", "کیفر", "کیفری", "زندان", "حبس", "شلاق", "اعدام", "سارق",
    "حرز", "قذف", "محاربه", "افساد فی الارض", "جنایت", "ضرب و جرح", "توهین",
    "افترا", "جعل", "تبانی", "رشوه", "اختلاس", "ارتشاء", "اخاذی",
}

CIVIL_KEYWORDS = {
    "قرارداد", "عقد", "بیع", "فروش", "خرید", "اجاره", "موجر", "مستاجر",
    "فسخ", "اقاله", "خیار", "خیارات", "خیار غبن", "خیار عیب", "خیار شرط",
    "مهریه", "نفقه", "طلاق", "نکاح", "ازدواج", "حضانت", "ارث", "وصیت",
    "ترکه", "ماترک", "مالکیت", "اموال", "حقوق مدنی", "تعهد", "تعهدات",
    "خسارت", "وجه التزام", "ضمانت", "ضامن", "ودیعه", "عاریه", "وکالت",
    "هبه", "وقف", "رهن", "مرتهن", "صلح", "ماده ۱۰", "ماده ۱۹۰",
}

PROCEDURAL_KEYWORDS = {
    "دادگاه", "دادسرا", "بازپرس", "دادیار", "قاضی", "دادخواست", "شکواییه",
    "لایحه", "تجدیدنظر", "فرجام", "واخواهی", "ابلاغ", "صلاحیت", "صلاحیت دادگاه",
    "شورای حل اختلاف", "دیوان عالی", "دیوان عدالت", "قرار", "حکم قطعی",
    "اجرای احکام", "اعاده دادرسی", "توقیف اموال", "تامین خواسته", "شهادت",
    "کارشناسی", "سوگند", "آیین دادرسی", "مواعد",
}

COMMERCIAL_LABOR_KEYWORDS = {
    "چک", "سفته", "برات", "اسناد تجاری", "ورشکستگی", "شرکت", "سهام",
    "هیئت مدیره", "ثبت شرکت", "قانون کار", "کارگر", "کارفرما", "سنوات",
    "حق بیمه", "تامین اجتماعی", "مرخصی", "اخراج", "وزارت کار", "اداره کار",
}

DRAFTING_KEYWORDS = {
    "تنظیم", "نگارش", "پیش‌نویس", "پیش نویس", "فرمت", "نمونه", "متن قرارداد",
    "بنویس", "طراحی قرارداد", "اصلاح بند", "ماده قرارداد",
}


def categorize_query(
    query: str,
    intent: Optional[str] = None,
    is_sufficient: bool = True,
) -> str:
    """Categorize user query into distinct legal or conversational categories.

    Args:
        query: Raw query text.
        intent: Classified intent (e.g. general, legal_qa, document_generation).
        is_sufficient: Whether grounding evidence was sufficient.

    Returns:
        Persian category label.
    """
    clean_query = query.strip().lower()

    if intent == "general":
        return "گفتگوی عمومی و راهنمایی"

    if intent == "document_generation":
        return "تنظیم و تدوین اسناد حقوقی"

    # Tokenize words/ngrams
    words = set(re.findall(r"[\w\u0600-\u06FF]+", clean_query))

    # Check keyword categories by priority
    if any(k in clean_query or k in words for k in DRAFTING_KEYWORDS):
        return "تنظیم و تدوین اسناد حقوقی"

    if any(k in clean_query or k in words for k in CRIMINAL_KEYWORDS):
        return "حقوق کیفری و مجازات"

    if any(k in clean_query or k in words for k in CIVIL_KEYWORDS):
        return "حقوق مدنی و قراردادها"

    if any(k in clean_query or k in words for k in COMMERCIAL_LABOR_KEYWORDS):
        return "حقوق تجارت و کار"

    if any(k in clean_query or k in words for k in PROCEDURAL_KEYWORDS):
        return "آیین دادرسی و امور قضایی"

    if not is_sufficient:
        return "شواهد ناکافی در پایگاه دانش"

    return "استعلامات عمومی حقوقی"
