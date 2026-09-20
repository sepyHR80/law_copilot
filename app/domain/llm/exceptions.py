"""Domain exceptions for LLM interactions."""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class LLMConfigurationError(LLMError):
    """Raised when the LLM configuration is invalid or missing required parameters."""


class LLMProviderError(LLMError):
    """Raised when an upstream LLM provider API call fails."""


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""


class LLMRateLimitError(LLMProviderError):
    """Raised when an upstream provider throttles or rate limits requests (HTTP 429)."""


class LLMResponseValidationError(LLMError):
    """Raised when structured response parsing or validation fails against the requested schema."""
