"""Structure-aware chunking package for Law Copilot."""

from app.ingestion.chunking.chunker import StructureAwareChunker
from app.ingestion.chunking.exceptions import ChunkingError
from app.ingestion.chunking.models import Chunk, ChunkingConfig
from app.ingestion.chunking.service import DocumentChunkingService

__all__ = [
    "Chunk",
    "ChunkingConfig",
    "ChunkingError",
    "DocumentChunkingService",
    "StructureAwareChunker",
]
