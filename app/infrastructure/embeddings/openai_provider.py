"""OpenAI-compatible embedding provider implementation.

This provider communicates with any OpenAI-compatible embedding endpoint
(e.g., LiteLLM Proxy, OpenAI, Ollama, vLLM) using the AsyncOpenAI client.
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence
import httpx
import openai
from openai import AsyncOpenAI

from app.domain.embeddings.exceptions import (
    EmbeddingConfigurationError,
    EmbeddingDimensionMismatchError,
    EmbeddingProviderError,
    EmbeddingValidationError,
)
from app.domain.embeddings.protocol import EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible implementation of EmbeddingProvider.

    Key characteristics:
    - Retains a single reusable AsyncOpenAI client instance across calls.
    - Slices inputs into configurable batch sizes.
    - Reconstructs embeddings based on response index fields.
    - Strictly validates dimensions against expected_dimension.
    - Retries transient errors at most once (no retry storms).
    - Protects privacy: never logs full text or API keys.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        expected_dimension: int = 1536,
        timeout: float = 30.0,
        batch_size: int = 100,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        if not endpoint or not endpoint.strip():
            raise EmbeddingConfigurationError("Embedding endpoint must not be empty.")
        if not model or not model.strip():
            raise EmbeddingConfigurationError("Embedding model must not be empty.")
        if expected_dimension < 1:
            raise EmbeddingConfigurationError("expected_dimension must be >= 1.")
        if timeout <= 0:
            raise EmbeddingConfigurationError("timeout must be > 0.")
        if batch_size < 1:
            raise EmbeddingConfigurationError("batch_size must be >= 1.")

        self.endpoint = endpoint.strip()
        self.api_key = api_key
        self.model = model.strip()
        self.expected_dimension = expected_dimension
        self.timeout = timeout
        self.batch_size = batch_size

        # Retain single reusable client
        self._client = client or AsyncOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key or "EMPTY",
            timeout=self.timeout,
            max_retries=0,  # We manage retry explicitly to enforce max 1 retry policy
        )
        self._owns_client = client is None

    async def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Generate embedding vectors for a sequence of texts.

        Args:
            texts: Sequence of strings to embed.

        Returns:
            List of embedding vectors, perfectly ordered by input index.

        Raises:
            EmbeddingValidationError: If any text is empty or invalid.
            EmbeddingDimensionMismatchError: If dimension differs from expected.
            EmbeddingProviderError: If the API fails.
        """
        # Empty list handling
        if not texts:
            return []

        # Validate each input text
        for i, t in enumerate(texts):
            if t is None or not isinstance(t, str) or not t.strip():
                raise EmbeddingValidationError(
                    f"Invalid text at index {i}: must be a non-empty string."
                )

        all_embeddings: List[List[float]] = []

        # Process in batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_texts = list(texts[batch_start : batch_start + self.batch_size])
            batch_embeddings = await self._embed_batch_with_retry(batch_texts)

            # Validate dimensions and collect
            for local_idx, emb in enumerate(batch_embeddings):
                global_idx = batch_start + local_idx
                if len(emb) != self.expected_dimension:
                    raise EmbeddingDimensionMismatchError(
                        expected_dimension=self.expected_dimension,
                        actual_dimension=len(emb),
                        index=global_idx,
                    )
                all_embeddings.append(emb)

        return all_embeddings

    async def _embed_batch_with_retry(self, batch_texts: List[str]) -> List[List[float]]:
        """Embed a single batch with at most one retry on transient failures."""
        max_attempts = 2  # Attempt 1 + 1 retry

        for attempt in range(max_attempts):
            try:
                response = await self._client.embeddings.create(
                    input=batch_texts,
                    model=self.model,
                )

                # Reconstruct ordering by item.index
                ordered: List[Optional[List[float]]] = [None] * len(batch_texts)
                for item in response.data:
                    idx = getattr(item, "index", None)
                    if idx is None and isinstance(item, dict):
                        idx = item.get("index")
                    emb = getattr(item, "embedding", None)
                    if emb is None and isinstance(item, dict):
                        emb = item.get("embedding")

                    if idx is not None and 0 <= idx < len(batch_texts):
                        ordered[idx] = emb

                # Verify all slots were filled
                for i, emb in enumerate(ordered):
                    if emb is None:
                        raise EmbeddingProviderError(
                            f"Missing embedding for batch index {i} in provider response."
                        )

                return ordered  # type: ignore[return-value]

            except (
                openai.BadRequestError,
                openai.AuthenticationError,
                openai.PermissionDeniedError,
                openai.NotFoundError,
            ) as err:
                # Permanent errors: do NOT retry
                status_code = getattr(err, "status_code", None)
                raise EmbeddingProviderError(
                    f"Embedding API request failed permanently: {err}",
                    status_code=status_code,
                ) from err

            except (
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.RateLimitError,
                openai.InternalServerError,
                openai.APIStatusError,
                httpx.TimeoutException,
                httpx.NetworkError,
            ) as err:
                status_code = getattr(err, "status_code", None)
                if attempt == 0:
                    # Retry once after short backoff
                    await asyncio.sleep(0.1)
                    continue
                # Second failure: raise without further retry
                raise EmbeddingProviderError(
                    f"Embedding API request failed after retry: {err}",
                    status_code=status_code,
                ) from err

            except Exception as err:
                # General unexpected error
                if isinstance(err, (EmbeddingError,)):
                    raise
                raise EmbeddingProviderError(f"Unexpected embedding error: {err}") from err

        raise EmbeddingProviderError("Embedding request failed after maximum retries.")

    async def close(self) -> None:
        """Close the underlying AsyncOpenAI client connection pool."""
        if self._owns_client and self._client is not None:
            await self._client.close()
