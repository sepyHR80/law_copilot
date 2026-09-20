"""Domain models for external search and source validation."""

from typing import Optional
from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    """Raw result returned by web search provider."""

    title: str
    url: str
    snippet: str
    domain: str = ""


class ValidatedSource(BaseModel):
    """Source item evaluated and filtered for legal trustworthiness."""

    title: str
    url: str
    domain: str
    content: str
    is_trusted: bool
    trust_score: float = Field(default=0.0, ge=0.0, le=1.0)
    authority_category: str = Field(default="general")  # "government", "judiciary", "academic", "reputable_legal", "general"
