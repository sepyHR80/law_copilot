"""Embedding provider protocol."""

from typing import List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Abstract protocol for embedding providers.

    The application and domain layers depend strictly on this protocol,
    ensuring full decoupling from model providers or SDKs.
    """

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embedding vectors for a sequence of texts.

        Args:
            texts: Sequence of non-empty strings to embed.

        Returns:
            List of embedding vectors, matching input ordering.

        Raises:
            EmbeddingValidationError: If any text is empty or whitespace.
            EmbeddingDimensionMismatchError: If returned vector dimension differs from expected.
            EmbeddingProviderError: If the underlying API call fails or times out.
        """
        ...

    async def close(self) -> None:
        """Release underlying HTTP client and connection pool resources."""
        ...
