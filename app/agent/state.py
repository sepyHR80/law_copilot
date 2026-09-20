"""Agent state definition for LangGraph orchestration."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.models import RetrievalFilter, RetrievalResult


class AgentState(TypedDict, total=False):
    """Execution state passed between LangGraph state machine nodes."""

    # User & session identifiers
    user_id: Optional[str]
    conversation_id: Optional[str]

    # Query details
    query: str
    normalized_query: str
    intent: str  # "legal_qa", "general", "insufficient"
    filters: Optional[RetrievalFilter]

    # Retrieval & Evidence
    retrieval_results: List[RetrievalResult]
    selected_evidence: List[EvidenceItem]
    external_sources: List[Any]
    enable_external_search: bool
    context_text: str
    memory_context: Optional[str]
    tool_results: Dict[str, Any]

    # Generation & Citations
    draft: Optional[str]
    raw_citations: List[Dict[str, Any]]
    citations: List[Citation]
    is_sufficient: bool

    # Verification & Control Flow
    verification_passed: bool
    verification_feedback: Optional[str]
    verification_result: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int

    # Output & Diagnostics
    final_response: Optional[str]
    errors: List[str]
    trace_metadata: Dict[str, Any]
