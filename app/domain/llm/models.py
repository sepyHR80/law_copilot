"""Domain models for LLM interactions."""

from typing import Any, Dict, List, Literal, Optional, Sequence, Type
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Represents a single chat message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


class LLMRequest(BaseModel):
    """Represents an invocation request to an LLM provider."""

    messages: Sequence[ChatMessage]
    model: Optional[str] = None
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    timeout: Optional[float] = Field(default=None, gt=0.0)
    response_format: Optional[Any] = None  # Type[BaseModel] or JSON schema dict
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token consumption statistics for an LLM response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Represents the completion returned by an LLM provider."""

    content: str
    parsed: Optional[Any] = None
    model: str
    usage: Optional[LLMUsage] = None
    finish_reason: Optional[str] = None
