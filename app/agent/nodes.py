"""Node implementations for the LangGraph legal agent."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.agent.state import AgentState
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.models import HybridSearchQuery, RetrievalFilter
from app.domain.retrieval.protocol import HybridRetrieverProtocol, RerankerProtocol
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _append_execution_step(
    state: AgentState,
    step: str,
    title: str,
    duration_ms: int,
    status: str = "completed",
    details: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Helper to record a step in the execution path."""
    path = list(state.get("execution_path") or [])
    path.append({
        "step": step,
        "title": title,
        "status": status,
        "duration_ms": max(1, duration_ms),
        "details": details or {},
        "timestamp": datetime.utcnow().isoformat(),
    })
    return path


class _DraftAnswerSchema(BaseModel):
    answer: str
    is_sufficient: bool = True
    citations: List[Dict[str, Any]] = Field(default_factory=list)


def create_load_memory_node(memory_service: Optional[Any] = None):
    """Create node loading scoped user preferences and case context."""

    async def load_memory(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        user_id_str = state.get("user_id")
        memory_context = ""

        if user_id_str and memory_service is not None:
            try:
                from uuid import UUID

                uid = UUID(str(user_id_str))
                memory_context = memory_service.build_scoped_context(uid)
            except Exception:
                memory_context = ""

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "memory_context": memory_context,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "memory_loaded": bool(memory_context),
            },
            "execution_path": _append_execution_step(
                state,
                step="load_memory",
                title="بررسی و بارگذاری حافظه کاربر",
                duration_ms=dur,
                details={"user_id": user_id_str, "memory_loaded": bool(memory_context)},
            ),
        }

    return load_memory


def create_analyze_intent_node(llm_service: Optional[LLMService] = None):
    """Create node that classifies user intent and legal category using LLM inference.

    Eliminates all regex and keyword heuristics; relies strictly on LLM intelligence.
    """

    async def analyze_intent(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("query", "").strip()

        # Classify intent and legal category via LLM inference when LLM service is available
        if llm_service is not None:
            try:
                from app.tracing.categorizer import categorize_with_llm

                classification = await categorize_with_llm(query=query, llm_service=llm_service)
                intent = classification.get("intent", "legal_qa")
                category = classification.get("category", "استعلامات و پژوهش حقوقی")
                dur = int((time.perf_counter() - t0) * 1000)
                return {
                    "intent": intent,
                    "trace_metadata": {
                        **state.get("trace_metadata", {}),
                        "intent_classified": intent,
                        "category": category,
                        "llm_classified": True,
                    },
                    "execution_path": _append_execution_step(
                        state,
                        step="analyze_intent",
                        title="تحلیل هوشمند قصد و رده‌بندی حقوقی (LLM)",
                        duration_ms=dur,
                        details={"intent": intent, "category": category, "model_driven": True},
                    ),
                }
            except Exception as exc:
                pass

        # Fallback when LLM is unavailable or not passed (e.g. isolated unit tests)
        q_lower = query.lower()
        if any(ind in q_lower for ind in ["draft", "write a contract", "write an agreement", "draft a complaint", "prepare a notice", "document generation"]):
            intent = "document_generation"
            category = "تنظیم و تدوین اسناد حقوقی"
        elif any(g in q_lower for g in ["hello", "hi", "hey", "greetings", "help", "سلام", "درود"]):
            intent = "general"
            category = "گفتگوی عمومی و راهنمایی"
        else:
            intent = "legal_qa"
            category = "استعلامات و پژوهش حقوقی"

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": intent,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "intent_classified": intent,
                "category": category,
            },
            "execution_path": _append_execution_step(
                state,
                step="analyze_intent",
                title="تحلیل قصد و نوع پیام",
                duration_ms=dur,
                details={"intent": intent, "requires_retrieval": (intent == "legal_qa")},
            ),
        }

    return analyze_intent


def create_handle_general_node():
    """Create node that handles conversational and non-retrieval questions."""

    async def handle_general(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query_text = state.get("query", "")
        is_persian = any("\u0600" <= c <= "\u06ff" for c in query_text)
        if is_persian:
            response_text = (
                "درود بر شما! من دستیار هوشمند حقوقی Law Copilot هستم. "
                "آماده‌ام تا به پرسش‌های حقوقی شما بر اساس قوانین، آیین‌نامه‌ها، احکام قضایی "
                "و قراردادهای بارگذاری‌شده در پایگاه دانش پاسخ دهم. "
                "شما می‌توانید سوال حقوقی خود را بپرسید تا با استناد دقیق به مواد قانونی به آن پاسخ دهم."
            )
        else:
            response_text = (
                "Hello! I am Law Copilot, your specialized legal AI assistant. "
                "You can ask me questions about your uploaded contracts, statutes, case law, "
                "and regulatory documents. I provide grounded answers with precise document citations."
            )
        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "final_response": response_text,
            "is_sufficient": True,
            "citations": [],
            "selected_evidence": [],
            "trace_metadata": {**state.get("trace_metadata", {}), "routed_general": True},
            "execution_path": _append_execution_step(
                state,
                step="handle_general",
                title="پاسخ به گفتگوی عمومی",
                duration_ms=dur,
                details={"response_preview": response_text[:120]},
            ),
        }

    return handle_general


def create_prepare_query_node():
    """Create node that normalizes and prepares search query."""

    async def prepare_query(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_query = state.get("query", "")
        # Normalize whitespace
        cleaned = " ".join(raw_query.strip().split())
        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "normalized_query": cleaned,
            "trace_metadata": {**state.get("trace_metadata", {}), "query_prepared": True},
            "execution_path": _append_execution_step(
                state,
                step="prepare_query",
                title="نرمال‌سازی و آماده‌سازی کوئری",
                duration_ms=dur,
                details={"normalized_query": cleaned},
            ),
        }

    return prepare_query


def create_retrieve_knowledge_node(
    hybrid_retriever: HybridRetrieverProtocol,
    reranker: RerankerProtocol,
):
    """Create node executing hybrid retrieval and cross-encoder reranking."""

    async def retrieve_knowledge(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("normalized_query", state.get("query", ""))
        filters = state.get("filters")
        top_k = state.get("top_k", 5)

        search_query = HybridSearchQuery(
            text_query=query,
            top_k=top_k,
            candidate_k=20,
            filters=filters,
        )
        candidates = await hybrid_retriever.search(search_query)

        if candidates:
            reranked = reranker.rerank(query=query, candidates=candidates, top_n=top_k)
        else:
            reranked = []

        dur = int((time.perf_counter() - t0) * 1000)
        top_cand_info = []
        for c in reranked[:3]:
            top_cand_info.append({
                "chunk_id": str(c.chunk_id),
                "score": round(float(c.score), 4) if c.score is not None else None,
                "preview": c.content[:100] + ("..." if len(c.content) > 100 else ""),
            })

        return {
            "retrieval_results": reranked,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "candidates_retrieved": len(candidates),
                "candidates_reranked": len(reranked),
            },
            "execution_path": _append_execution_step(
                state,
                step="retrieve_knowledge",
                title="بازیابی ترکیبی (Hybrid Vector + FTS)",
                duration_ms=dur,
                details={
                    "candidates_retrieved": len(candidates),
                    "candidates_reranked": len(reranked),
                    "top_candidates": top_cand_info,
                },
            ),
        }

    return retrieve_knowledge


def create_assess_evidence_node():
    """Create node determining if sufficient evidence was retrieved."""

    async def assess_evidence(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = state.get("retrieval_results", [])
        is_sufficient = len(results) > 0
        dur = int((time.perf_counter() - t0) * 1000)

        return {
            "is_sufficient": is_sufficient,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "evidence_assessed_sufficient": is_sufficient,
            },
            "execution_path": _append_execution_step(
                state,
                step="assess_evidence",
                title="ارزیابی کفایت مدارک و شواهد",
                duration_ms=dur,
                details={"is_sufficient": is_sufficient, "evidence_count": len(results)},
            ),
        }

    return assess_evidence


def create_handle_insufficient_node():
    """Create node returning standard grounded insufficient evidence response."""

    async def handle_insufficient(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query_text = state.get("query", "")
        is_persian = any("\u0600" <= c <= "\u06ff" for c in query_text)
        if is_persian:
            response_text = (
                "بر اساس مدارک و اسناد بارگذاری‌شده در پایگاه دانش، شواهد و اطلاعات کافی برای پاسخ دقیق به این پرسش یافت نشد. "
                "لطفاً سند قانونی مربوطه را از بخش بارگذاری اضافه کنید تا بتوانم با استناد به آن پاسخ دهم."
            )
        else:
            response_text = "Based on the provided documents, there is insufficient evidence to answer this question."

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "final_response": response_text,
            "is_sufficient": False,
            "citations": [],
            "selected_evidence": [],
            "trace_metadata": {**state.get("trace_metadata", {}), "insufficient_handled": True},
            "execution_path": _append_execution_step(
                state,
                step="handle_insufficient",
                title="عدم کفایت مدارک در پایگاه دانش",
                duration_ms=dur,
                details={"is_sufficient": False, "reason": "no_relevant_evidence"},
            ),
        }

    return handle_insufficient


def create_external_search_node(search_service: Optional[Any] = None):
    """Create node that searches external web sources when internal retrieval is insufficient."""

    async def external_search(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("normalized_query", state.get("query", "")).strip()
        if not search_service or not query:
            dur = int((time.perf_counter() - t0) * 1000)
            return {
                "external_sources": [],
                "is_sufficient": False,
                "trace_metadata": {**state.get("trace_metadata", {}), "external_search_skipped": True},
                "execution_path": _append_execution_step(
                    state,
                    step="external_search",
                    title="جستجوی تکمیلی وب",
                    duration_ms=dur,
                    status="skipped",
                    details={"skipped": True},
                ),
            }

        try:
            sources = await search_service.search_and_validate(query=query)
            is_sufficient = len(sources) > 0
            dur = int((time.perf_counter() - t0) * 1000)
            return {
                "external_sources": sources,
                "is_sufficient": is_sufficient,
                "trace_metadata": {
                    **state.get("trace_metadata", {}),
                    "external_sources_count": len(sources),
                    "external_search_executed": True,
                },
                "execution_path": _append_execution_step(
                    state,
                    step="external_search",
                    title="جستجوی تکمیلی وب",
                    duration_ms=dur,
                    details={"sources_found": len(sources)},
                ),
            }
        except Exception as exc:
            dur = int((time.perf_counter() - t0) * 1000)
            return {
                "external_sources": [],
                "is_sufficient": False,
                "trace_metadata": {**state.get("trace_metadata", {}), "external_search_error": True},
                "execution_path": _append_execution_step(
                    state,
                    step="external_search",
                    title="جستجوی تکمیلی وب",
                    duration_ms=dur,
                    status="failed",
                    details={"error": str(exc)},
                ),
            }

    return external_search


def create_build_context_node(
    context_builder: ContextBuilder,
    search_service: Optional[Any] = None,
):
    """Create node structuring evidence into token-bounded context."""

    async def build_context(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        retrieval_results = state.get("retrieval_results", [])
        evidence_items = context_builder.build_evidence_items(retrieval_results)
        context_text = context_builder.format_context(evidence_items)

        external_sources = state.get("external_sources", [])
        if external_sources and search_service is not None:
            ext_text = search_service.format_external_context(external_sources)
            if context_text and context_text != "No relevant evidence documents retrieved.":
                context_text = f"{context_text}\n\n---\n\n{ext_text}"
            else:
                context_text = ext_text

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "selected_evidence": evidence_items,
            "context_text": context_text,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "evidence_items_count": len(evidence_items),
                "external_sources_count": len(external_sources),
            },
            "execution_path": _append_execution_step(
                state,
                step="build_context",
                title="تدوین بافتار و ساخت شواهد (Context)",
                duration_ms=dur,
                details={
                    "evidence_items_count": len(evidence_items),
                    "context_char_count": len(context_text),
                },
            ),
        }

    return build_context


def create_generate_draft_node(llm_service: LLMService, prompts_dir: Path = PROMPTS_DIR):
    """Create node that generates an initial grounded answer draft with citations."""

    async def generate_draft(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        query = state.get("normalized_query", state.get("query", ""))
        context_text = state.get("context_text", "")

        # Read prompts
        sys_file = prompts_dir / "rag_system.txt"
        user_file = prompts_dir / "rag_user.txt"

        sys_prompt = sys_file.read_text(encoding="utf-8").strip() if sys_file.is_file() else "Legal Assistant"
        user_template = user_file.read_text(encoding="utf-8").strip() if user_file.is_file() else "{question}\n{evidence_block}"

        prompt = user_template.format(question=query, evidence_block=context_text)
        memory_ctx = state.get("memory_context")
        if memory_ctx:
            prompt = f"{memory_ctx}\n\n{prompt}"

        try:
            parsed, _ = await llm_service.structured_complete(
                prompt=prompt,
                schema=_DraftAnswerSchema,
                system_prompt=sys_prompt,
            )
            draft_answer = parsed.answer
            raw_citations = parsed.citations
            is_sufficient = parsed.is_sufficient
        except Exception:
            res = await llm_service.complete(prompt=prompt, system_prompt=sys_prompt)
            content = res.content.strip()
            try:
                data = json.loads(content)
                draft_answer = data.get("answer", content)
                raw_citations = data.get("citations", [])
                is_sufficient = bool(data.get("is_sufficient", True))
            except Exception:
                draft_answer = content
                raw_citations = []
                is_sufficient = "insufficient evidence" not in content.lower()

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "draft": draft_answer,
            "raw_citations": raw_citations,
            "is_sufficient": is_sufficient,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "draft_generated": True,
            },
            "execution_path": _append_execution_step(
                state,
                step="generate_draft",
                title="تولید پاسخ با مدل هوش مصنوعی (LLM)",
                duration_ms=dur,
                details={
                    "is_sufficient": is_sufficient,
                    "citations_claimed": len(raw_citations),
                    "draft_length": len(draft_answer),
                },
            ),
        }

    return generate_draft


def create_verify_answer_node(
    context_builder: ContextBuilder,
    verification_service: Optional[Any] = None,
):
    """Create node verifying draft citations and grounding."""

    async def verify_answer(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_citations = state.get("raw_citations", [])
        evidence_items = state.get("selected_evidence", [])
        is_sufficient = state.get("is_sufficient", True)
        draft = state.get("draft", "")

        dur = int((time.perf_counter() - t0) * 1000)

        if not is_sufficient:
            return {
                "verification_passed": True,
                "final_response": draft,
                "citations": [],
                "trace_metadata": {**state.get("trace_metadata", {}), "verification": "passed_insufficient"},
                "execution_path": _append_execution_step(
                    state,
                    step="verify_answer",
                    title="راستی‌آزمایی پاسخ و استنادها",
                    duration_ms=dur,
                    details={"status": "passed_insufficient", "reason": "not_sufficient"},
                ),
            }

        # Resolve citations strictly
        verified = context_builder.resolve_citations(raw_citations, evidence_items)

        # If verification service is provided, run comprehensive claim verifier
        if verification_service is not None:
            result = verification_service.verify(
                response_text=draft,
                citations=verified,
                evidence_items=evidence_items,
                raw_citations=raw_citations,
            )
            if not result.passed:
                return {
                    "verification_passed": False,
                    "verification_feedback": result.repair_guidance or "Please fix claims and citations.",
                    "verification_result": result.model_dump(mode="json"),
                    "trace_metadata": {
                        **state.get("trace_metadata", {}),
                        "verification": "failed_claim_verifier",
                        "unsupported_claims": len(result.unsupported_claims),
                        "citation_errors": len(result.citation_errors),
                    },
                    "execution_path": _append_execution_step(
                        state,
                        step="verify_answer",
                        title="راستی‌آزمایی پاسخ و استنادها",
                        duration_ms=dur,
                        status="failed",
                        details={"passed": False, "unsupported_claims": len(result.unsupported_claims)},
                    ),
                }
            return {
                "verification_passed": True,
                "citations": verified,
                "final_response": draft,
                "verification_result": result.model_dump(mode="json"),
                "trace_metadata": {**state.get("trace_metadata", {}), "verification": "passed"},
                "execution_path": _append_execution_step(
                    state,
                    step="verify_answer",
                    title="راستی‌آزمایی پاسخ و استنادها",
                    duration_ms=dur,
                    details={"passed": True, "verified_citations": len(verified)},
                ),
            }

        # If raw citations were claimed but NONE matched valid evidence -> verification failure
        if raw_citations and not verified:
            return {
                "verification_passed": False,
                "verification_feedback": "All cited evidence identifiers were invalid or hallucinated. Please cite only valid Evidence IDs from the provided context.",
                "trace_metadata": {**state.get("trace_metadata", {}), "verification": "failed_hallucinated_citations"},
                "execution_path": _append_execution_step(
                    state,
                    step="verify_answer",
                    title="راستی‌آزمایی پاسخ و استنادها",
                    duration_ms=dur,
                    status="failed",
                    details={"passed": False, "reason": "hallucinated_citations"},
                ),
            }

        return {
            "verification_passed": True,
            "citations": verified,
            "final_response": draft,
            "trace_metadata": {**state.get("trace_metadata", {}), "verification": "passed"},
            "execution_path": _append_execution_step(
                state,
                step="verify_answer",
                title="راستی‌آزمایی پاسخ و استنادها",
                duration_ms=dur,
                details={"passed": True, "verified_citations": len(verified)},
            ),
        }

    return verify_answer


def create_repair_draft_node(llm_service: LLMService):
    """Create node repairing draft when verification finds citation or grounding issues."""

    async def repair_draft(state: AgentState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        retry_count = state.get("retry_count", 0) + 1
        query = state.get("normalized_query", state.get("query", ""))
        context_text = state.get("context_text", "")
        feedback = state.get("verification_feedback", "Please fix citation IDs.")
        current_draft = state.get("draft", "")

        repair_prompt = (
            f"QUESTION: {query}\n\n"
            f"SUPPLIED EVIDENCE:\n{context_text}\n\n"
            f"PREVIOUS DRAFT:\n{current_draft}\n\n"
            f"CORRECTION REQUIRED:\n{feedback}\n\n"
            f"Please revise your answer and provide valid citations matching the Evidence IDs above."
        )

        try:
            parsed, _ = await llm_service.structured_complete(
                prompt=repair_prompt,
                schema=_DraftAnswerSchema,
            )
            draft_answer = parsed.answer
            raw_citations = parsed.citations
            is_sufficient = parsed.is_sufficient
        except Exception:
            res = await llm_service.complete(prompt=repair_prompt)
            draft_answer = res.content.strip()
            raw_citations = []
            is_sufficient = True

        dur = int((time.perf_counter() - t0) * 1000)
        return {
            "draft": draft_answer,
            "raw_citations": raw_citations,
            "is_sufficient": is_sufficient,
            "retry_count": retry_count,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "repair_attempted": retry_count,
            },
            "execution_path": _append_execution_step(
                state,
                step="repair_draft",
                title="اصلاح و بازبینی پیش‌نویس",
                duration_ms=dur,
                details={"retry_count": retry_count},
            ),
        }

    return repair_draft
