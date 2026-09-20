"""Integration tests for Stage 04 MinIO storage abstraction."""

import uuid
from io import BytesIO

import pytest

from app.core.config import get_settings
from app.infrastructure.storage import MinioStorage, StorageBackend
from app.infrastructure.storage.exceptions import ObjectNotFoundError, StorageError


@pytest.fixture
def storage() -> StorageBackend:
    """Create a connected storage instance with bucket ensured."""
    settings = get_settings()
    store = MinioStorage(settings=settings)
    store.initialize()
    return store


def _key() -> str:
    return f"test/{uuid.uuid4()}.txt"


def test_bucket_exists_after_initialization(storage: StorageBackend) -> None:
    """Configured bucket should exist after storage initialization."""
    settings = get_settings()
    assert storage.bucket_exists(settings.minio_bucket_name) is True


def test_upload_and_download_text_object(storage: StorageBackend) -> None:
    """Upload and download a small UTF-8 text object."""
    key = _key()
    content = "Hello, legal world! \u06cc\u0627 \u0639\u0644\u064a\u0643\u0645"
    data = content.encode("utf-8")

    storage.put_object(key, data, content_type="text/plain")
    downloaded = storage.get_object(key)

    assert downloaded == data
    assert downloaded.decode("utf-8") == content


def test_storage_service_can_connect() -> None:
    """Storage service can connect to MinIO and initialize."""
    settings = get_settings()
    store = MinioStorage(settings=settings)
    store.initialize()
    assert store is not None
    assert isinstance(store, StorageBackend)


def test_bucket_initialization_is_idempotent(storage: StorageBackend) -> None:
    """Initializing twice does not fail."""
    settings = get_settings()
    storage.initialize()  # second time
    assert storage.bucket_exists(settings.minio_bucket_name) is True


def test_object_exists_after_upload(storage: StorageBackend) -> None:
    """Object existence check returns true after upload."""
    key = _key()
    storage.put_object(key, b"exists", content_type="application/octet-stream")
    assert storage.object_exists(key) is True


def test_object_exists_false_for_missing(storage: StorageBackend) -> None:
    """Object existence check returns false for a missing object."""
    assert storage.object_exists(f"nonexistent/{uuid.uuid4()}.txt") is False


def test_missing_object_raises_not_found(storage: StorageBackend) -> None:
    """Downloading a missing object raises ObjectNotFoundError."""
    with pytest.raises(ObjectNotFoundError):
        storage.get_object(f"missing/{uuid.uuid4()}.txt")


def test_object_can_be_deleted(storage: StorageBackend) -> None:
    """Object can be deleted."""
    key = _key()
    storage.put_object(key, b"delete me", content_type="application/octet-stream")
    assert storage.object_exists(key) is True
    storage.delete_object(key)
    assert storage.object_exists(key) is False


def test_binary_content_roundtrip(storage: StorageBackend) -> None:
    """Binary content survives upload and download."""
    key = _key()
    binary_data = bytes(range(256))
    storage.put_object(key, binary_data, content_type="application/octet-stream")
    downloaded = storage.get_object(key)
    assert downloaded == binary_data


def test_content_type_metadata(storage: StorageBackend) -> None:
    """Content type metadata is stored on the object."""
    key = _key()
    storage.put_object(key, b"pdf-bytes", content_type="application/pdf")
    stat = storage.stat_object(key)
    assert stat.content_type == "application/pdf"
