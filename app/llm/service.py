"""LLM application service providing high-level generation interfaces."""

from typing import Any, Optional, Sequence, Type, TypeVar
from pydantic import BaseModel

from app.domain.llm.models import ChatMessage, LLMRequest, LLMResponse
from app.domain.llm.protocol import LLMProvider
from app.infrastructure.llm.openai_provider import OpenAILLMProvider

T = TypeVar("T", bound=BaseModel)


class LLMService:
    """Application service for LLM text and structured output generation.

    Decoupled from vendor-specific libraries through LLMProvider.
    """

    def __init__(self, provider: Optional[LLMProvider] = None) -> None:
        self._provider = provider

    @property
    def provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = OpenAILLMProvider()
        return self._provider

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Execute a full LLM request."""
        return await self.provider.generate(request)

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> LLMResponse:
        """Convenience method for standard text completion."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return await self.generate(request)

    async def structured_complete(
        self,
        prompt: str,
        schema: Type[T],
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> tuple[T, LLMResponse]:
        """Convenience method for generating structured output matching a Pydantic schema."""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))

        request = LLMRequest(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=schema,
        )
        response = await self.generate(request)
        return response.parsed, response

    async def close(self) -> None:
        """Release provider resources."""
        if self._provider is not None:
            await self._provider.close()
