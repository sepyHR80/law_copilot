"""Domain models for RAG pipeline."""

from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from app.domain.llm.models import LLMUsage
from app.domain.retrieval.models import RetrievalFilter


class Citation(BaseModel):
    """Machine-readable citation referencing a specific retrieved chunk."""

    document_id: UUID
    document_version_id: UUID
    chunk_id: UUID
    page: Optional[int] = None
    section: Optional[str] = None
    snippet: Optional[str] = None


class EvidenceItem(BaseModel):
    """Structured evidence item presented to the LLM and returned to the caller."""

    evidence_id: int  # 1-based sequential evidence identifier
    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    content: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)


class RAGQuery(BaseModel):
    """Input query for the grounded RAG QA pipeline."""

    question: str = Field(..., min_length=1)
    filters: Optional[RetrievalFilter] = None
    top_k: int = Field(default=5, ge=1, le=50)
    candidate_k: int = Field(default=20, ge=1, le=100)
    max_context_tokens: Optional[int] = Field(default=None, gt=0)


class RAGResponse(BaseModel):
    """Grounded question answering response with citations and evidence."""

    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    is_sufficient: bool = True
    model: str
    usage: Optional[LLMUsage] = None
