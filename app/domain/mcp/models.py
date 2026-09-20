"""Domain models for MCP (Model Context Protocol) knowledge tools."""

from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, Field

from app.domain.retrieval.models import RetrievalFilter


class MCPEvidenceItem(BaseModel):
    """Structured evidence item returned by MCP knowledge tools."""

    chunk_id: UUID
    document_id: UUID
    document_version_id: UUID
    content: str
    score: float
    page: Optional[int] = None
    section: Optional[str] = None
    knowledge_type: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)


class MCPSearchKnowledgeRequest(BaseModel):
    """Structured input schema for the search_knowledge MCP tool."""

    query: str = Field(..., min_length=1)
    knowledge_type: Optional[str] = Field(default="factual")
    top_k: int = Field(default=8, ge=1, le=50)
    filters: Optional[RetrievalFilter] = None


class MCPSearchKnowledgeResponse(BaseModel):
    """Structured output schema returned by the search_knowledge MCP tool."""

    evidence: List[MCPEvidenceItem] = Field(default_factory=list)
    total_count: int = Field(default=0)
    knowledge_type: str = Field(default="factual")
