"""Storage infrastructure for Law Copilot."""

from app.infrastructure.storage.base import ObjectMetadata, StorageBackend
from app.infrastructure.storage.exceptions import (
    ObjectAlreadyExistsError,
    ObjectNotFoundError,
    StorageError,
)
from app.infrastructure.storage.memory import InMemoryStorage
from app.infrastructure.storage.minio import MinioStorage

__all__ = [
    "MinioStorage",
    "InMemoryStorage",
    "StorageBackend",
    "ObjectMetadata",
    "StorageError",
    "ObjectNotFoundError",
    "ObjectAlreadyExistsError",
]
