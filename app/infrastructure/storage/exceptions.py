"""Storage-specific exceptions for Law Copilot."""


class StorageError(Exception):
    """Base exception for all storage-related errors."""


class ObjectNotFoundError(StorageError):
    """Raised when a requested object does not exist."""


class ObjectAlreadyExistsError(StorageError):
    """Raised when an object already exists and cannot be overwritten."""
