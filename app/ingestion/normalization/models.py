"""Canonical document models.

These models represent a format-independent document structure
that can be consumed by downstream stages (chunking, embedding, retrieval).
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class CanonicalBlock(BaseModel):
    """Base class for all canonical document blocks."""
    type: str = "block"
    text: str = ""
    page_number: int = 1
    order: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CanonicalHeading(CanonicalBlock):
    """A heading block."""
    type: str = "heading"
    level: int = 1


class CanonicalParagraph(CanonicalBlock):
    """A paragraph block."""
    type: str = "paragraph"


class CanonicalList(CanonicalBlock):
    """A list block."""
    type: str = "list"
    items: List[str] = Field(default_factory=list)
    ordered: bool = False


class CanonicalTable(CanonicalBlock):
    """A table block."""
    type: str = "table"
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)


AnyCanonicalBlock = Union[
    CanonicalHeading,
    CanonicalParagraph,
    CanonicalList,
    CanonicalTable,
    CanonicalBlock,
]


class CanonicalDocument(BaseModel):
    """Canonical representation of a document, independent of source format."""
    document_id: str
    document_version_id: str
    title: Optional[str] = None
    mime_type: str
    blocks: List[AnyCanonicalBlock] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
