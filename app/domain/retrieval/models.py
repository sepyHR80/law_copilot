"""Retrieval domain models."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    """Metadata filters for narrowing retrieval candidates."""

    knowledge_type: Optional[str] = None  # "factual" | "stylistic"
    document_type: Optional[str] = None
    source: Optional[str] = None
    document_id: Optional[UUID] = None
    document_version_id: Optional[UUID] = None


class VectorSearchQuery(BaseModel):
    """Query object for vector retrieval."""

    vector: List[float]
    top_k: int = Field(default=5, ge=1)
    min_score: Optional[float] = None
    filters: Optional[RetrievalFilter] = None


class LexicalSearchQuery(BaseModel):
    """Query object for lexical (full-text) retrieval."""

    query: str
    top_k: int = Field(default=5, ge=1)
    min_score: Optional[float] = None
    filters: Optional[RetrievalFilter] = None


class HybridSearchQuery(BaseModel):
    """Query object for hybrid (vector + lexical) retrieval with Reciprocal Rank Fusion."""

    text_query: str
    vector: Optional[List[float]] = None
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    filters: Optional[RetrievalFilter] = None


class RetrievalResult(BaseModel):
    """A single normalized retrieval result item.

    Matches specification per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 12.
    """

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    content: str
    score: float
    rank: int
    source: Dict[str, Any] = Field(default_factory=dict)
