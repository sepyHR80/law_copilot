"""Node implementations for the LangGraph legal agent."""

import json
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


class _DraftAnswerSchema(BaseModel):
    answer: str
    is_sufficient: bool = True
    citations: List[Dict[str, Any]] = Field(default_factory=list)


def create_load_memory_node(memory_service: Optional[Any] = None):
    """Create node loading scoped user preferences and case context."""

    async def load_memory(state: AgentState) -> Dict[str, Any]:
        user_id_str = state.get("user_id")
        memory_context = ""

        if user_id_str and memory_service is not None:
            try:
                from uuid import UUID

                uid = UUID(str(user_id_str))
                memory_context = memory_service.build_scoped_context(uid)
            except Exception:
                memory_context = ""

        return {
            "memory_context": memory_context,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "memory_loaded": bool(memory_context),
            },
        }

    return load_memory


def create_analyze_intent_node():
    """Create node that classifies user intent."""


    async def analyze_intent(state: AgentState) -> Dict[str, Any]:
        query = state.get("query", "").strip()
        cleaned = query.lower().strip("!?,. ")
        words = set(cleaned.split())

        # Fast deterministic classification for common conversational intents
        greetings = {"hello", "hi", "hey", "greetings", "help", "morning", "afternoon", "evening"}
        is_greeting = (
            cleaned in greetings
            or any(cleaned.startswith(g) for g in ["hello", "hi", "hey", "greetings", "who are you", "what can you do", "help"])
            or bool(words.intersection({"hello", "hi", "hey", "greetings"}))
        )
        if is_greeting or (len(words) <= 1 and cleaned in {"test", "ping"}):
            return {
                "intent": "general",
                "trace_metadata": {**state.get("trace_metadata", {}), "intent_classified": "general"},
            }

        return {
            "intent": "legal_qa",
            "trace_metadata": {**state.get("trace_metadata", {}), "intent_classified": "legal_qa"},
        }


    return analyze_intent


def create_handle_general_node():
    """Create node that handles conversational and non-retrieval questions."""

    async def handle_general(state: AgentState) -> Dict[str, Any]:
        response_text = (
            "Hello! I am Law Copilot, your specialized legal AI assistant. "
            "You can ask me questions about your uploaded contracts, statutes, case law, "
            "and regulatory documents. I provide grounded answers with precise document citations."
        )
        return {
            "final_response": response_text,
            "is_sufficient": True,
            "citations": [],
            "selected_evidence": [],
            "trace_metadata": {**state.get("trace_metadata", {}), "routed_general": True},
        }

    return handle_general


def create_prepare_query_node():
    """Create node that normalizes and prepares search query."""

    async def prepare_query(state: AgentState) -> Dict[str, Any]:
        raw_query = state.get("query", "")
        # Normalize whitespace
        cleaned = " ".join(raw_query.strip().split())
        return {
            "normalized_query": cleaned,
            "trace_metadata": {**state.get("trace_metadata", {}), "query_prepared": True},
        }

    return prepare_query


def create_retrieve_knowledge_node(
    hybrid_retriever: HybridRetrieverProtocol,
    reranker: RerankerProtocol,
):
    """Create node executing hybrid retrieval and cross-encoder reranking."""

    async def retrieve_knowledge(state: AgentState) -> Dict[str, Any]:
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

        return {
            "retrieval_results": reranked,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "candidates_retrieved": len(candidates),
                "candidates_reranked": len(reranked),
            },
        }

    return retrieve_knowledge


def create_assess_evidence_node():
    """Create node determining if sufficient evidence was retrieved."""

    async def assess_evidence(state: AgentState) -> Dict[str, Any]:
        results = state.get("retrieval_results", [])
        is_sufficient = len(results) > 0

        return {
            "is_sufficient": is_sufficient,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "evidence_assessed_sufficient": is_sufficient,
            },
        }

    return assess_evidence


def create_handle_insufficient_node():
    """Create node returning standard grounded insufficient evidence response."""

    async def handle_insufficient(state: AgentState) -> Dict[str, Any]:
        return {
            "final_response": "Based on the provided documents, there is insufficient evidence to answer this question.",
            "is_sufficient": False,
            "citations": [],
            "selected_evidence": [],
            "trace_metadata": {**state.get("trace_metadata", {}), "insufficient_handled": True},
        }

    return handle_insufficient


def create_build_context_node(context_builder: ContextBuilder):
    """Create node structuring evidence into token-bounded context."""

    async def build_context(state: AgentState) -> Dict[str, Any]:
        retrieval_results = state.get("retrieval_results", [])
        evidence_items = context_builder.build_evidence_items(retrieval_results)
        context_text = context_builder.format_context(evidence_items)

        return {
            "selected_evidence": evidence_items,
            "context_text": context_text,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "evidence_items_count": len(evidence_items),
            },
        }

    return build_context


def create_generate_draft_node(llm_service: LLMService, prompts_dir: Path = PROMPTS_DIR):
    """Create node that generates an initial grounded answer draft with citations."""

    async def generate_draft(state: AgentState) -> Dict[str, Any]:
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

        return {
            "draft": draft_answer,
            "raw_citations": raw_citations,
            "is_sufficient": is_sufficient,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "draft_generated": True,
            },
        }

    return generate_draft


def create_verify_answer_node(context_builder: ContextBuilder):
    """Create node verifying draft citations and grounding."""

    async def verify_answer(state: AgentState) -> Dict[str, Any]:
        raw_citations = state.get("raw_citations", [])
        evidence_items = state.get("selected_evidence", [])
        is_sufficient = state.get("is_sufficient", True)
        draft = state.get("draft", "")

        if not is_sufficient:
            return {
                "verification_passed": True,
                "final_response": draft,
                "citations": [],
                "trace_metadata": {**state.get("trace_metadata", {}), "verification": "passed_insufficient"},
            }

        # Resolve citations strictly
        verified = context_builder.resolve_citations(raw_citations, evidence_items)

        # If raw citations were claimed but NONE matched valid evidence -> verification failure
        if raw_citations and not verified:
            return {
                "verification_passed": False,
                "verification_feedback": "All cited evidence identifiers were invalid or hallucinated. Please cite only valid Evidence IDs from the provided context.",
                "trace_metadata": {**state.get("trace_metadata", {}), "verification": "failed_hallucinated_citations"},
            }

        return {
            "verification_passed": True,
            "citations": verified,
            "final_response": draft,
            "trace_metadata": {**state.get("trace_metadata", {}), "verification": "passed"},
        }

    return verify_answer


def create_repair_draft_node(llm_service: LLMService):
    """Create node repairing draft when verification finds citation or grounding issues."""

    async def repair_draft(state: AgentState) -> Dict[str, Any]:
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

        return {
            "draft": draft_answer,
            "raw_citations": raw_citations,
            "is_sufficient": is_sufficient,
            "retry_count": retry_count,
            "trace_metadata": {
                **state.get("trace_metadata", {}),
                "repair_attempted": retry_count,
            },
        }

    return repair_draft
