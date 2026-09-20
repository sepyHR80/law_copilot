"""Domain models for response claim verification."""

from typing import List, Optional
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured result produced by the legal claim verifier."""

    passed: bool = True
    unsupported_claims: List[str] = Field(default_factory=list)
    citation_errors: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    repair_guidance: Optional[str] = None
