"""In-memory storage backend for development, testing, and fallback when MinIO is unavailable."""

from typing import Dict, Optional, Tuple
from app.infrastructure.storage.base import ObjectMetadata, StorageBackend
from app.infrastructure.storage.exceptions import ObjectNotFoundError


class InMemoryStorage(StorageBackend):
    """In-memory implementation of StorageBackend."""

    def __init__(self) -> None:
        self._objects: Dict[str, Tuple[bytes, Optional[str]]] = {}

    def initialize(self) -> None:
        """No initialization needed for in-memory storage."""
        pass

    def put_object(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        self._objects[key] = (data, content_type)

    def get_object(self, key: str) -> bytes:
        if key not in self._objects:
            raise ObjectNotFoundError(f"Object '{key}' not found in in-memory storage.")
        return self._objects[key][0]

    def delete_object(self, key: str) -> None:
        if key in self._objects:
            del self._objects[key]

    def object_exists(self, key: str) -> bool:
        return key in self._objects

    def bucket_exists(self, bucket_name: str) -> bool:
        return True

    def stat_object(self, key: str) -> ObjectMetadata:
        if key not in self._objects:
            raise ObjectNotFoundError(f"Object '{key}' not found in in-memory storage.")
        data, content_type = self._objects[key]
        return ObjectMetadata(content_type=content_type, size=len(data))
