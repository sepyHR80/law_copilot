"""Chunking-related exceptions."""

from app.domain.documents.exceptions import DocumentError


class ChunkingError(DocumentError):
    """Raised when document chunking fails."""
