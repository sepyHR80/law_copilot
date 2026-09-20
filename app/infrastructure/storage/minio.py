"""MinIO storage implementation for Law Copilot."""

import re
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.core.config import Settings, get_settings
from app.infrastructure.storage.base import ObjectMetadata, StorageBackend
from app.infrastructure.storage.exceptions import ObjectNotFoundError, StorageError


class MinioStorage(StorageBackend):
    """MinIO-based object storage implementation."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._client = Minio(
            endpoint=self._settings.minio_endpoint,
            access_key=self._settings.minio_access_key,
            secret_key=self._settings.minio_secret_key,
            secure=self._settings.minio_secure,
        )
        self._bucket = self._settings.minio_bucket_name

    def initialize(self) -> None:
        """Ensure the configured bucket exists (idempotent)."""
        try:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)
        except S3Error as exc:
            raise StorageError(f"Failed to initialize bucket '{self._bucket}': {exc}") from exc

    def put_object(
        self,
        key: str,
        data: bytes,
        content_type: Optional[str] = None,
    ) -> None:
        """Upload an object to MinIO."""
        self._validate_key(key)
        try:
            stream = BytesIO(data)
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=key,
                data=stream,
                length=len(data),
                content_type=content_type or "application/octet-stream",
            )
        except S3Error as exc:
            raise StorageError(f"Failed to upload object '{key}': {exc}") from exc

    def get_object(self, key: str) -> bytes:
        """Download an object from MinIO."""
        self._validate_key(key)
        try:
            response = self._client.get_object(self._bucket, key)
            return response.read()
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object '{key}' not found") from exc
            raise StorageError(f"Failed to download object '{key}': {exc}") from exc

    def delete_object(self, key: str) -> None:
        """Delete an object from MinIO."""
        self._validate_key(key)
        try:
            self._client.remove_object(self._bucket, key)
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object '{key}' not found") from exc
            raise StorageError(f"Failed to delete object '{key}': {exc}") from exc

    def object_exists(self, key: str) -> bool:
        """Check whether an object exists in MinIO."""
        self._validate_key(key)
        try:
            self._client.stat_object(self._bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise StorageError(f"Failed to check object '{key}': {exc}") from exc

    def bucket_exists(self, bucket_name: str) -> bool:
        """Check whether a bucket exists in MinIO."""
        try:
            return self._client.bucket_exists(bucket_name)
        except S3Error as exc:
            raise StorageError(f"Failed to check bucket '{bucket_name}': {exc}") from exc

    def stat_object(self, key: str) -> ObjectMetadata:
        """Get metadata for an object in MinIO."""
        self._validate_key(key)
        try:
            stat = self._client.stat_object(self._bucket, key)
            return ObjectMetadata(
                content_type=stat.content_type,
                size=stat.size,
            )
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                raise ObjectNotFoundError(f"Object '{key}' not found") from exc
            raise StorageError(f"Failed to stat object '{key}': {exc}") from exc

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validate object key for path traversal safety."""
        if not key or not isinstance(key, str):
            raise StorageError("Object key must be a non-empty string")
        if ".." in key:
            raise StorageError(f"Object key contains path traversal: '{key}'")
        if key.startswith("/"):
            raise StorageError(f"Object key must not start with '/': '{key}'")
