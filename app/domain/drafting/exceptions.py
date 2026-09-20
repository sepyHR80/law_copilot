"""Exceptions for document drafting domain."""

from app.domain.documents.exceptions import DocumentError


class DraftingError(DocumentError):
    """Base exception for document drafting errors."""


class StyleLeakageError(DraftingError):
    """Raised when a stylistic reference is improperly cited as legal authority."""
