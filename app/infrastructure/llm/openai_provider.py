"""OpenAI-compatible LLM provider implementation communicating with LiteLLM Proxy."""

import json
import logging
from typing import Any, Dict, List, Optional, Type
import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

from app.core.config import get_settings
from app.domain.llm.exceptions import (
    LLMConfigurationError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponseValidationError,
    LLMTimeoutError,
)
from app.domain.llm.models import (
    ChatMessage,
    LLMRequest,
    LLMResponse,
    LLMUsage,
)
from app.domain.llm.protocol import LLMProvider


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible implementation of LLMProvider.

    Connects to LiteLLM Proxy or any OpenAI-compatible gateway.
    Features:
    - Persistent, reusable AsyncOpenAI client.
    - Model selection and request-level overrides.
    - Structured output support with Pydantic schema validation.
    - Comprehensive mapping of HTTP and provider exceptions.
    - Safe logging practices avoiding credential leakage.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: Optional[float] = None,
        temperature: Optional[float] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        settings = get_settings()

        self.base_url = (base_url or settings.llm_base_url).strip()
        if not self.base_url:
            raise LLMConfigurationError("LLM base_url must not be empty.")

        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.default_model = (default_model or settings.llm_model).strip()
        if not self.default_model:
            raise LLMConfigurationError("LLM default_model must not be empty.")

        self.timeout = timeout if timeout is not None else settings.llm_timeout
        if self.timeout <= 0:
            raise LLMConfigurationError("LLM timeout must be greater than 0.")

        self.temperature = temperature if temperature is not None else settings.llm_temperature

        self._client = client or AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "EMPTY",
            timeout=self.timeout,
        )
        self._owns_client = client is None

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a chat completion from the LiteLLM gateway."""
        if not request.messages:
            raise LLMConfigurationError("LLMRequest must contain at least one message.")

        primary_model = request.model or self.default_model
        candidate_models = [primary_model]
        if "generativelanguage" in self.base_url.lower():
            for fallback in ("gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.6-flash"):
                if fallback not in candidate_models:
                    candidate_models.append(fallback)

        temperature = request.temperature if request.temperature is not None else self.temperature
        timeout = request.timeout if request.timeout is not None else self.timeout

        formatted_messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            entry: Dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.name:
                entry["name"] = msg.name
            formatted_messages.append(entry)

        is_pydantic_schema = (
            request.response_format is not None
            and isinstance(request.response_format, type)
            and issubclass(request.response_format, BaseModel)
        )

        last_rate_limit_exc: Optional[openai.RateLimitError] = None

        for idx, model in enumerate(candidate_models):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": formatted_messages,
                "temperature": temperature,
                "timeout": timeout,
            }
            if request.max_tokens is not None:
                kwargs["max_tokens"] = request.max_tokens

            try:
                if is_pydantic_schema and hasattr(self._client, "beta") and hasattr(self._client.beta, "chat"):
                    completion = await self._client.beta.chat.completions.parse(
                        response_format=request.response_format,
                        **kwargs,
                    )
                    choice = completion.choices[0]
                    content = choice.message.content or ""
                    parsed = getattr(choice.message, "parsed", None)
                    if parsed is None and content:
                        try:
                            parsed = request.response_format.model_validate_json(content)
                        except ValidationError as val_err:
                            raise LLMResponseValidationError(
                                f"Structured output validation failed: {val_err}"
                            ) from val_err
                else:
                    if request.response_format is not None:
                        if is_pydantic_schema:
                            kwargs["response_format"] = {"type": "json_object"}
                        else:
                            kwargs["response_format"] = request.response_format

                    completion = await self._client.chat.completions.create(**kwargs)
                    choice = completion.choices[0]
                    content = choice.message.content or ""
                    parsed = None

                    if is_pydantic_schema:
                        try:
                            parsed = request.response_format.model_validate_json(content)
                        except (ValidationError, json.JSONDecodeError) as val_err:
                            raise LLMResponseValidationError(
                                f"Structured output validation failed: {val_err}"
                            ) from val_err

                # Usage metadata
                usage = None
                if getattr(completion, "usage", None) is not None:
                    usage = LLMUsage(
                        prompt_tokens=completion.usage.prompt_tokens or 0,
                        completion_tokens=completion.usage.completion_tokens or 0,
                        total_tokens=completion.usage.total_tokens or 0,
                    )

                finish_reason = None
                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

                return LLMResponse(
                    content=content,
                    parsed=parsed,
                    model=getattr(completion, "model", model),
                    usage=usage,
                    finish_reason=finish_reason,
                )

            except openai.RateLimitError as exc:
                last_rate_limit_exc = exc
                if idx < len(candidate_models) - 1:
                    next_model = candidate_models[idx + 1]
                    logger.warning("LLM model %s rate limited (429), failing over to %s", model, next_model)
                    continue
                raise LLMRateLimitError(f"LLM provider rate limit exceeded: {exc}") from exc
            except openai.APITimeoutError as exc:
                raise LLMTimeoutError(f"LLM request timed out after {timeout}s: {exc}") from exc
            except openai.APIConnectionError as exc:
                raise LLMProviderError(f"Failed to connect to LiteLLM gateway: {exc}") from exc
            except openai.APIStatusError as exc:
                raise LLMProviderError(f"LLM provider returned error status {exc.status_code}: {exc}") from exc
            except LLMError:
                raise
            except Exception as exc:
                raise LLMProviderError(f"Unexpected error during LLM generation: {exc}") from exc

        if last_rate_limit_exc is not None:
            raise LLMRateLimitError(f"LLM provider rate limit exceeded: {last_rate_limit_exc}") from last_rate_limit_exc

    async def close(self) -> None:
        """Close underlying client."""
        if self._owns_client and hasattr(self._client, "close"):
            await self._client.close()
