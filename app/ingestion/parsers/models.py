"""Pydantic models for parsed document representation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    """A single page/section of a parsed document."""
    page_number: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    """A table extracted from a document."""
    rows: List[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Normalized representation of a parsed document."""
    document_id: str
    document_version_id: str
    title: Optional[str] = None
    mime_type: str
    pages: List[ParsedPage] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
