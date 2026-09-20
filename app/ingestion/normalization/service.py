"""Document normalization service."""

from typing import Optional

from app.ingestion.parsers.models import ParsedDocument
from app.ingestion.normalization.models import CanonicalDocument
from app.ingestion.normalization.normalizer import DocumentNormalizer


class DocumentNormalizationService:
    """Application service for document normalization.

    This service is pure: it has no knowledge of storage, database,
    or API. It accepts a ParsedDocument and returns a CanonicalDocument.
    """

    def __init__(self, normalizer: Optional[DocumentNormalizer] = None) -> None:
        self._normalizer = normalizer or DocumentNormalizer()

    def normalize(self, parsed_document: ParsedDocument) -> CanonicalDocument:
        """Normalize a ParsedDocument into a CanonicalDocument.

        Args:
            parsed_document: The parsed document from Stage 06.

        Returns:
            CanonicalDocument with normalized blocks.
        """
        return self._normalizer.normalize(parsed_document)
