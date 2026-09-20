"""Domain models for legal document drafting."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.domain.llm.models import LLMUsage
from app.domain.rag.models import Citation, EvidenceItem


class DraftSection(BaseModel):
    """A distinct structural section of a drafted legal document."""

    heading: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    citations: List[Citation] = Field(default_factory=list)


class DraftDocument(BaseModel):
    """Full drafted legal document with grounded sections and provenance."""

    title: str = Field(..., min_length=1)
    document_type: str = Field(default="legal_document")
    sections: List[DraftSection] = Field(default_factory=list)
    full_text: str = Field(default="")
    citations: List[Citation] = Field(default_factory=list)
    factual_evidence: List[EvidenceItem] = Field(default_factory=list)
    style_references: List[EvidenceItem] = Field(default_factory=list)
    model: str = Field(default="unknown")
    usage: Optional[LLMUsage] = None


class DraftingQuery(BaseModel):
    """Input query specification for legal document drafting."""

    topic: str = Field(..., min_length=1)
    document_type: Optional[str] = Field(default="agreement")
    recipient: Optional[str] = None
    jurisdiction: Optional[str] = None
    factual_top_k: int = Field(default=5, ge=1, le=50)
    style_top_k: int = Field(default=3, ge=0, le=20)
    additional_instructions: Optional[str] = None
