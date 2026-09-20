"""Abstract storage backend interface for Law Copilot."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ObjectMetadata:
    """Metadata for a stored object."""
    content_type: Optional[str] = None
    size: Optional[int] = None


class StorageBackend(ABC):
    """Abstract interface for object storage operations."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the storage backend (e.g., ensure bucket exists)."""
        ...

    @abstractmethod
    def put_object(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        """Upload an object to storage.

        Args:
            key: Object key/path in the bucket.
            data: Raw bytes to upload.
            content_type: MIME type of the object.
        """
        ...

    @abstractmethod
    def get_object(self, key: str) -> bytes:
        """Download an object from storage.

        Args:
            key: Object key/path in the bucket.

        Returns:
            Raw bytes of the object.

        Raises:
            ObjectNotFoundError: If the object does not exist.
        """
        ...

    @abstractmethod
    def delete_object(self, key: str) -> None:
        """Delete an object from storage.

        Args:
            key: Object key/path in the bucket.

        Raises:
            ObjectNotFoundError: If the object does not exist.
        """
        ...

    @abstractmethod
    def object_exists(self, key: str) -> bool:
        """Check whether an object exists in storage.

        Args:
            key: Object key/path in the bucket.

        Returns:
            True if the object exists, False otherwise.
        """
        ...

    @abstractmethod
    def bucket_exists(self, bucket_name: str) -> bool:
        """Check whether a bucket exists.

        Args:
            bucket_name: Name of the bucket.

        Returns:
            True if the bucket exists, False otherwise.
        """
        ...

    @abstractmethod
    def stat_object(self, key: str) -> ObjectMetadata:
        """Get metadata for an object.

        Args:
            key: Object key/path in the bucket.

        Returns:
            ObjectMetadata with content_type and size.

        Raises:
            ObjectNotFoundError: If the object does not exist.
        """
        ...
