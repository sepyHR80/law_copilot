"""Domain exceptions for memory operations."""


class MemoryError(Exception):
    """Base exception for all memory operations."""


class MemoryNotFoundError(MemoryError):
    """Raised when a specific memory record is not found."""


class MemoryValidationError(MemoryError):
    """Raised when memory key, type, or value fails validation."""


class ConversationNotFoundError(MemoryError):
    """Raised when a requested conversation is not found."""
