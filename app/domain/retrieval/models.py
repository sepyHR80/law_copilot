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
