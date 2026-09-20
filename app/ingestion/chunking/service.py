"""Document chunking service."""

from typing import List, Optional

from app.ingestion.chunking.chunker import StructureAwareChunker
from app.ingestion.chunking.models import Chunk, ChunkingConfig
from app.ingestion.normalization.models import CanonicalDocument


class DocumentChunkingService:
    """Application service for structure-aware document chunking.

    Accepts a CanonicalDocument and returns a list of Chunk objects.
    Pure structural transformation with no storage or database coupling.
    """

    def __init__(
        self,
        chunker: Optional[StructureAwareChunker] = None,
        config: Optional[ChunkingConfig] = None,
    ) -> None:
        self._chunker = chunker or StructureAwareChunker(config=config)

    def chunk(
        self,
        document: CanonicalDocument,
        config: Optional[ChunkingConfig] = None,
    ) -> List[Chunk]:
        """Chunk a canonical document.

        Args:
            document: CanonicalDocument to chunk.
            config: Optional override ChunkingConfig.

        Returns:
            List of Chunk objects.
        """
        if config is not None:
            chunker = StructureAwareChunker(config=config)
            return chunker.chunk(document)
        return self._chunker.chunk(document)
