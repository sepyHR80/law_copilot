"""Unit tests for Stage 09 — Embedding Service.

Tests cover:
1. FakeEmbeddingProvider (Domain protocol verification)
2. OpenAIEmbeddingProvider with Mocked AsyncOpenAI:
   - normal embedding
   - batch embedding
   - batching respects batch_size
   - ordering preserved
   - response index reordering
   - dimension mismatch (with rich error attributes)
   - empty list returns empty list
   - empty and whitespace-only text rejected
   - provider failure (non-retryable errors)
   - timeout handling
   - retry on transient failure
   - retry storm prevention (max 1 retry)
   - configuration validation
   - client lifecycle close
"""

from typing import List, Sequence
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import openai

from app.domain.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingValidationError,
)
from app.domain.embeddings.protocol import EmbeddingProvider
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider


class FakeEmbeddingProvider:
    """Deterministic in-memory fake provider conforming to EmbeddingProvider protocol."""

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension
        self.closed = False

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise EmbeddingValidationError(f"Invalid text at index {i}")
        # Deterministic vector based on text hash
        results = []
        for t in texts:
            val = float(len(t)) / 100.0
            vec = [val] * self.dimension
            results.append(vec)
        return results

    async def close(self) -> None:
        self.closed = True


class TestDomainProtocol:
    def test_fake_provider_implements_protocol(self):
        """Verify FakeEmbeddingProvider satisfies the runtime checkable EmbeddingProvider protocol."""
        fake = FakeEmbeddingProvider()
        assert isinstance(fake, EmbeddingProvider)

    def test_openai_provider_implements_protocol(self):
        """Verify OpenAIEmbeddingProvider satisfies the runtime checkable EmbeddingProvider protocol."""
        provider = OpenAIEmbeddingProvider(
            endpoint="http://localhost:4000/v1",
            api_key="test-key",
            client=MagicMock(spec=openai.AsyncOpenAI),
        )
        assert isinstance(provider, EmbeddingProvider)


def _make_mock_response(embeddings_with_indices: List[tuple[int, List[float]]]):
    """Helper to create a mock OpenAI Embedding response."""
    items = []
    for idx, vec in embeddings_with_indices:
        item = MagicMock()
        item.index = idx
        item.embedding = vec
        items.append(item)
    resp = MagicMock()
    resp.data = items
    return resp


class TestOpenAIEmbeddingProvider:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=openai.AsyncOpenAI)
        client.embeddings = MagicMock()
        client.embeddings.create = AsyncMock()
        client.close = AsyncMock()
        return client

    @pytest.fixture
    def provider(self, mock_client):
        return OpenAIEmbeddingProvider(
            endpoint="http://localhost:4000/v1",
            api_key="test-key",
            model="text-embedding-3-small",
            expected_dimension=1536,
            timeout=30.0,
            batch_size=2,
            client=mock_client,
        )

    @pytest.mark.asyncio
    async def test_normal_embedding(self, provider, mock_client):
        """Normal embedding returns expected vector with correct dimension."""
        mock_vec = [0.1] * 1536
        mock_client.embeddings.create.return_value = _make_mock_response([(0, mock_vec)])

        result = await provider.embed_texts(["Legal document clause."])

        assert len(result) == 1
        assert len(result[0]) == 1536
        assert result[0] == mock_vec
        mock_client.embeddings.create.assert_awaited_once_with(
            input=["Legal document clause."],
            model="text-embedding-3-small",
        )

    @pytest.mark.asyncio
    async def test_batch_embedding(self, provider, mock_client):
        """Batch embedding returns multiple vectors matching input size."""
        vec1 = [0.1] * 1536
        vec2 = [0.2] * 1536
        mock_client.embeddings.create.return_value = _make_mock_response([(0, vec1), (1, vec2)])

        result = await provider.embed_texts(["Text 1", "Text 2"])

        assert len(result) == 2
        assert result[0] == vec1
        assert result[1] == vec2

    @pytest.mark.asyncio
    async def test_batching_respects_batch_size(self, provider, mock_client):
        """Inputs larger than batch_size are chunked into multiple API calls."""
        # batch_size is 2, so 5 items should result in 3 API calls: [2, 2, 1]
        mock_client.embeddings.create.side_effect = [
            _make_mock_response([(0, [0.1] * 1536), (1, [0.2] * 1536)]),
            _make_mock_response([(0, [0.3] * 1536), (1, [0.4] * 1536)]),
            _make_mock_response([(0, [0.5] * 1536)]),
        ]

        texts = ["T1", "T2", "T3", "T4", "T5"]
        result = await provider.embed_texts(texts)

        assert len(result) == 5
        assert mock_client.embeddings.create.await_count == 3

    @pytest.mark.asyncio
    async def test_ordering_preserved(self, provider, mock_client):
        """Output order strictly corresponds to input order across batches."""
        mock_client.embeddings.create.side_effect = [
            _make_mock_response([(0, [1.0] * 1536), (1, [2.0] * 1536)]),
            _make_mock_response([(0, [3.0] * 1536)]),
        ]

        result = await provider.embed_texts(["First", "Second", "Third"])

        assert result[0][0] == 1.0
        assert result[1][0] == 2.0
        assert result[2][0] == 3.0

    @pytest.mark.asyncio
    async def test_response_index_ordering(self, provider, mock_client):
        """Output is reconstructed by index even if provider returns jumbled indices."""
        vec0 = [10.0] * 1536
        vec1 = [20.0] * 1536
        # Jumbled: item with index 1 first, then item with index 0
        mock_client.embeddings.create.return_value = _make_mock_response([(1, vec1), (0, vec0)])

        result = await provider.embed_texts(["Zero", "One"])

        assert result[0] == vec0
        assert result[1] == vec1

    @pytest.mark.asyncio
    async def test_dimension_mismatch(self, provider, mock_client):
        """Raises EmbeddingDimensionMismatchError if provider returns unexpected dimension."""
        wrong_dim_vec = [0.1] * 1024  # 1024 != 1536
        mock_client.embeddings.create.return_value = _make_mock_response([(0, wrong_dim_vec)])

        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            await provider.embed_texts(["Sample text"])

        err = exc_info.value
        assert err.expected_dimension == 1536
        assert err.actual_dimension == 1024
        assert err.index == 0

    @pytest.mark.asyncio
    async def test_empty_list(self, provider, mock_client):
        """Empty list returns empty list immediately without calling the API."""
        result = await provider.embed_texts([])
        assert result == []
        mock_client.embeddings.create.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_text", ["", "   ", "\t\n", None])
    async def test_empty_text_rejected(self, provider, invalid_text):
        """Empty, whitespace-only, or None text raises EmbeddingValidationError."""
        with pytest.raises(EmbeddingValidationError):
            await provider.embed_texts(["Valid clause", invalid_text])

    @pytest.mark.asyncio
    async def test_provider_failure(self, provider, mock_client):
        """Permanent provider failure (e.g. 401 AuthenticationError) raises EmbeddingProviderError without retry."""
        mock_client.embeddings.create.side_effect = openai.AuthenticationError(
            message="Invalid API key",
            response=MagicMock(status_code=401),
            body=None,
        )

        with pytest.raises(EmbeddingProviderError) as exc_info:
            await provider.embed_texts(["Contract clause"])

        assert exc_info.value.status_code == 401
        # Should not retry permanent error
        assert mock_client.embeddings.create.await_count == 1

    @pytest.mark.asyncio
    async def test_timeout(self, provider, mock_client):
        """Timeout error triggers one retry, then raises EmbeddingProviderError."""
        mock_client.embeddings.create.side_effect = openai.APITimeoutError(request=MagicMock())

        with pytest.raises(EmbeddingProviderError):
            await provider.embed_texts(["Contract clause"])

        # Attempt 1 + Attempt 2 (retry) = 2 calls
        assert mock_client.embeddings.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, provider, mock_client):
        """Transient error on first attempt succeeds on second attempt."""
        success_vec = [0.5] * 1536
        mock_client.embeddings.create.side_effect = [
            openai.InternalServerError(
                message="Temporary server error",
                response=MagicMock(status_code=500),
                body=None,
            ),
            _make_mock_response([(0, success_vec)]),
        ]

        result = await provider.embed_texts(["Retry text"])

        assert len(result) == 1
        assert result[0] == success_vec
        assert mock_client.embeddings.create.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_storm_prevention(self, provider, mock_client):
        """Transient error on both attempts fails after exactly 2 calls (no retry storm)."""
        mock_client.embeddings.create.side_effect = [
            openai.RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            openai.RateLimitError(
                message="Rate limited again",
                response=MagicMock(status_code=429),
                body=None,
            ),
        ]

        with pytest.raises(EmbeddingProviderError):
            await provider.embed_texts(["Text"])

        # Strictly max 2 calls (initial + 1 retry)
        assert mock_client.embeddings.create.await_count == 2

    def test_configuration_validation(self):
        """Invalid configurations raise EmbeddingConfigurationError."""
        # batch_size < 1
        with pytest.raises(EmbeddingConfigurationError):
            OpenAIEmbeddingProvider(endpoint="http://url", api_key="k", batch_size=0)

        # timeout <= 0
        with pytest.raises(EmbeddingConfigurationError):
            OpenAIEmbeddingProvider(endpoint="http://url", api_key="k", timeout=0)

        # expected_dimension < 1
        with pytest.raises(EmbeddingConfigurationError):
            OpenAIEmbeddingProvider(endpoint="http://url", api_key="k", expected_dimension=0)

        # empty endpoint
        with pytest.raises(EmbeddingConfigurationError):
            OpenAIEmbeddingProvider(endpoint="", api_key="k")

        # empty model
        with pytest.raises(EmbeddingConfigurationError):
            OpenAIEmbeddingProvider(endpoint="http://url", api_key="k", model="")

    @pytest.mark.asyncio
    async def test_client_lifecycle_close(self, mock_client):
        """Calling close() closes the underlying AsyncOpenAI client."""
        # For a provider that manages its own client lifecycle
        provider = OpenAIEmbeddingProvider(
            endpoint="http://localhost:4000/v1",
            api_key="key",
            client=mock_client,
        )
        provider._owns_client = True
        await provider.close()
        mock_client.close.assert_awaited_once()
