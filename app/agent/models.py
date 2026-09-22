"""Pydantic schemas for the agent API and client interactions."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.models import RetrievalFilter


class AgentRequest(BaseModel):
    """Input request for invoking the legal agent."""

    query: str = Field(..., min_length=1)
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    filters: Optional[RetrievalFilter] = None
    max_retries: int = Field(default=2, ge=0, le=5)
    top_k: int = Field(default=5, ge=1, le=20)
    enable_external_search: bool = Field(default=False)


class AgentResponse(BaseModel):
    """Output response returned by the legal agent."""

    query: str
    response: str
    intent: str
    citations: List[Citation] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    is_sufficient: bool = True
    retry_count: int = 0
    trace_metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_path: List[Dict[str, Any]] = Field(default_factory=list)
    trace_id: Optional[str] = None
