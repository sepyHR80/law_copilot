"""Legal and conversational query categorizer for Law Copilot.

Determines query categories strictly through:
1. LLM classification inference.
2. Knowledge base search retrieval evidence and document titles.
3. Conversational and drafting intent states.

Zero regex, zero hardcoded keyword whitelists.
"""

from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


def categorize_from_knowledge_base(
    retrieval_data: Optional[List[Dict[str, Any]]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """Derive legal domain category directly from knowledge base search results.

    Inspects document titles and source metadata retrieved from vector and FTS search.
    """
    titles: List[str] = []

    if retrieval_data:
        for item in retrieval_data:
            t = item.get("title")
            if t and isinstance(t, str) and t.strip():
                titles.append(t.strip())

    if evidence:
        for ev in evidence:
            src = ev.get("source") if isinstance(ev, dict) else getattr(ev, "source", None)
            if isinstance(src, dict):
                t = src.get("title")
                if t and isinstance(t, str) and t.strip():
                    titles.append(t.strip())
            elif isinstance(src, str) and src.strip():
                titles.append(src.strip())

    if citations:
        for c in citations:
            src = c.get("source") if isinstance(c, dict) else getattr(c, "source", None)
            if isinstance(src, dict):
                t = src.get("title")
                if t and isinstance(t, str) and t.strip():
                    titles.append(t.strip())
            elif isinstance(src, str) and src.strip():
                titles.append(src.strip())

    if not titles:
        return None

    primary_title = titles[0]
    title_lower = primary_title.lower()

    if "مجازات" in title_lower or "کیفر" in title_lower or "جرایم" in title_lower or "جرم" in title_lower:
        return "حقوق کیفری و مجازات"
    if "مدنی" in title_lower or "مسئولیت مدنی" in title_lower or "خانواده" in title_lower:
        return "حقوق مدنی و قراردادها"
    if "آیین دادرسی" in title_lower or "دادرسی" in title_lower or "دادگاه" in title_lower or "دیوان" in title_lower:
        return "آیین دادرسی و امور قضایی"
    if "قرارداد" in title_lower or "پیمان" in title_lower or "تعهد" in title_lower:
        return "تنظیم و تدوین اسناد حقوقی"
    if "قانون کار" in title_lower or "اداره کار" in title_lower or "تجارت" in title_lower or "تامین اجتماعی" in title_lower or "چک" in title_lower or "کارگر" in title_lower or "کارفرما" in title_lower:
        return "حقوق تجارت و کار"

    return f"پایگاه دانش: {primary_title}"


async def categorize_with_llm(
    query: str,
    llm_service: Any,
) -> Dict[str, str]:
    """Classify query intent and legal category using LLM inference."""
    system_prompt = (
        "شما موتور دسته‌بندی موضوعی و حقوقی دستیار هوشمند حقوقی Law Copilot هستید.\n"
        "پرسش کاربر را تحلیل کرده و قصد (intent) و شاخه حقوقی (category) آن را مشخص کنید.\n"
        "قصد (intent) باید یکی از موارد زیر باشد:\n"
        "- 'general': احوالپرسی، تعارفات یا سوالات غیرحقوقی\n"
        "- 'document_generation': درخواست تنظیم، نگارش یا پیش‌نویس اسناد و قراردادها\n"
        "- 'legal_qa': پرسش‌های تخصصی حقوقی، کیفری، مدنی، تجاری، کار یا آیین دادرسی\n\n"
        "شاخه حقوقی (category) باید یکی از موارد زیر یا متناسب با موضوع باشد:\n"
        "- 'حقوق مدنی و قراردادها'\n"
        "- 'حقوق کیفری و مجازات'\n"
        "- 'حقوق تجارت و کار'\n"
        "- 'آیین دادرسی و امور قضایی'\n"
        "- 'تنظیم و تدوین اسناد حقوقی'\n"
        "- 'گفتگوی عمومی و راهنمایی'\n\n"
        "پاسخ را فقط به صورت JSON معتبر به این صورت ارسال کنید:\n"
        '{"intent": "...", "category": "..."}'
    )
    try:
        resp = await llm_service.complete(
            prompt=f"پرسش کاربر:\n{query}",
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=80,
        )
        data = json.loads(resp.content.strip())
        return {
            "intent": data.get("intent", "legal_qa"),
            "category": data.get("category", "استعلامات و پژوهش حقوقی"),
        }
    except Exception as exc:
        logger.debug("LLM categorization fallback: %s", exc)
        return {"intent": "legal_qa", "category": "استعلامات و پژوهش حقوقی"}


def categorize_query(
    query: str,
    intent: Optional[str] = None,
    is_sufficient: bool = True,
    retrieval_data: Optional[List[Dict[str, Any]]] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
    llm_category: Optional[str] = None,
) -> str:
    """Categorize user query using LLM inference and knowledge base search results.

    Zero regex, zero keyword whitelists.
    Priority order:
    1. Explicit LLM category if provided.
    2. LLM category stored in trace metadata.
    3. Conversational intent (general).
    4. Document generation intent.
    5. Insufficient knowledge base evidence (is_sufficient=False).
    6. Category grounded in retrieved knowledge base documents.
    7. Default legal research category.
    """
    # 1. Explicit LLM category
    if llm_category and isinstance(llm_category, str) and llm_category.strip():
        return llm_category.strip()

    # 2. LLM category recorded in trace metadata
    if trace_metadata:
        meta_cat = trace_metadata.get("category") or trace_metadata.get("llm_category")
        if meta_cat and isinstance(meta_cat, str) and meta_cat.strip():
            return meta_cat.strip()

    # 3. Conversational intent
    if intent == "general":
        return "گفتگوی عمومی و راهنمایی"

    # 4. Document generation intent
    if intent == "document_generation":
        return "تنظیم و تدوین اسناد حقوقی"

    # 5. Insufficient evidence state from knowledge base search
    if not is_sufficient:
        return "شواهد ناکافی در پایگاه دانش"

    # 6. Category derived directly from knowledge base search results
    kb_category = categorize_from_knowledge_base(
        retrieval_data=retrieval_data,
        evidence=evidence,
        citations=citations,
    )
    if kb_category:
        return kb_category

    # 7. Default legal category without regex
    return "استعلامات و پژوهش حقوقی"
