"""Verification domain package."""

from app.domain.verification.exceptions import VerificationError
from app.domain.verification.models import VerificationResult

__all__ = ["VerificationResult", "VerificationError"]
