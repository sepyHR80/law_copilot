"""Document normalization layer for Law Copilot."""

from app.ingestion.normalization.models import (
    AnyCanonicalBlock,
    CanonicalBlock,
    CanonicalDocument,
    CanonicalHeading,
    CanonicalList,
    CanonicalParagraph,
    CanonicalTable,
)
from app.ingestion.normalization.normalizer import DocumentNormalizer
from app.ingestion.normalization.service import DocumentNormalizationService
from app.ingestion.normalization.exceptions import DocumentNormalizationError

__all__ = [
    "AnyCanonicalBlock",
    "CanonicalBlock",
    "CanonicalDocument",
    "CanonicalHeading",
    "CanonicalList",
    "CanonicalParagraph",
    "CanonicalTable",
    "DocumentNormalizer",
    "DocumentNormalizationService",
    "DocumentNormalizationError",
]
