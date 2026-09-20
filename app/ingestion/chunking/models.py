"""Chunk domain models and chunking configuration.

These models represent the output of the structure-aware chunking stage.
They are independent of persistence — the infrastructure layer is responsible
for mapping these to SQLAlchemy models or other storage representations.
"""

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ChunkingConfig(BaseModel):
    """Configuration for the structure-aware chunker.

    Defaults per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 10.3.
    These are initial project defaults, tunable by later evaluation.
    """

    max_chars: int = 3000
    overlap_chars: int = 300
    min_chars: int = 200
    enable_parent_chunks: bool = True


class Chunk(BaseModel):
    """A single chunk produced by the structure-aware chunker.

    Fields per LAW_COPILOT_AGENT_MASTER_SPEC.md Section 10.2.

    Attributes:
        id: Unique identifier for this chunk.
        document_id: The logical document this chunk belongs to.
        document_version_id: The specific version of the document.
        parent_chunk_id: If this is a child chunk, the ID of its parent.
        chunk_type: "parent" for structural context chunks,
                    "child" for retrieval/embedding candidates.
        content: The textual content of the chunk.
        page: The primary page number, if available.
        section: Human-readable section label.
        section_path: Full hierarchical section path, e.g.
                      ["Chapter 2", "General Provisions", "Article 5"].
        chunk_index: Sequential index within the document version.
        metadata: Provenance and structural metadata including
                  source_blocks, source_pages, block_types, etc.
    """

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    document_version_id: UUID
    parent_chunk_id: Optional[UUID] = None
    chunk_type: Literal["parent", "child"] = "child"
    content: str
    page: Optional[int] = None
    section: Optional[str] = None
    section_path: List[str] = Field(default_factory=list)
    chunk_index: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
