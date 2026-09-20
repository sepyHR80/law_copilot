"""LLM provider protocol definition."""

from typing import Protocol, runtime_checkable

from app.domain.llm.models import LLMRequest, LLMResponse


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract interface for LLM calls.

    Decouples application logic from specific vendor SDKs or gateway APIs.
    """

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a model completion for the given request.

        Args:
            request: The validated LLMRequest instance.

        Returns:
            LLMResponse containing text content, optional parsed schema, and token usage.

        Raises:
            LLMConfigurationError: If request parameters are malformed.
            LLMTimeoutError: If the request exceeds the configured timeout.
            LLMRateLimitError: If rate limited by the upstream provider.
            LLMProviderError: If the upstream provider returns an error.
            LLMResponseValidationError: If structured parsing/validation fails.
        """
        ...

    async def close(self) -> None:
        """Release underlying HTTP client and connection pool resources."""
        ...
