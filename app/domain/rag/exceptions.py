"""Domain exceptions for RAG pipeline."""


class RAGError(Exception):
    """Base exception for all RAG-related errors."""


class ContextBuildError(RAGError):
    """Raised when structured context construction fails."""


class CitationValidationError(RAGError):
    """Raised when generated citations reference fabricated or unverified evidence."""
