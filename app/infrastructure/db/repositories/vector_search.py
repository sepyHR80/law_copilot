"""PostgreSQL + pgvector implementation of vector similarity retrieval."""

from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.retrieval.exceptions import (
    InvalidQueryError,
    RetrievalError,
    VectorDimensionError,
)
from app.domain.retrieval.models import RetrievalResult, VectorSearchQuery
from app.domain.retrieval.protocol import VectorRetrieverProtocol
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion


class PgVectorRetriever(VectorRetrieverProtocol):
    """Vector similarity retriever backed by PostgreSQL and pgvector."""

    def __init__(
        self,
        session: Session,
        expected_dimension: Optional[int] = None,
    ) -> None:
        self.session = session
        settings = get_settings()
        self.expected_dimension = expected_dimension or settings.embedding_dimension

    def search(self, query: VectorSearchQuery) -> List[RetrievalResult]:
        """Search document chunks using vector cosine distance and metadata filters."""
        if not query.vector:
            raise InvalidQueryError("Query vector must not be empty.")
        if query.top_k < 1:
            raise InvalidQueryError("top_k must be at least 1.")
        if len(query.vector) != self.expected_dimension:
            raise VectorDimensionError(
                expected_dimension=self.expected_dimension,
                actual_dimension=len(query.vector),
            )

        try:
            # Cosine distance expression using pgvector <=> operator
            distance = DocumentChunk.embedding.cosine_distance(query.vector)

            stmt = (
                select(
                    DocumentChunk.id.label("chunk_id"),
                    DocumentVersion.document_id.label("document_id"),
                    DocumentChunk.document_version_id.label("document_version_id"),
                    DocumentChunk.content.label("content"),
                    DocumentChunk.page.label("page"),
                    DocumentChunk.section.label("section"),
                    DocumentChunk.chunk_metadata.label("chunk_metadata"),
                    DocumentChunk.chunk_index.label("chunk_index"),
                    Document.title.label("title"),
                    Document.document_type.label("document_type"),
                    Document.knowledge_type.label("knowledge_type"),
                    Document.source.label("document_source"),
                    distance.label("distance"),
                )
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(DocumentChunk.embedding.is_not(None))
            )

            # Filter out structural parent chunks (only search retrieval units)
            # In Stage 08, parent chunks have metadata["is_parent"] = True
            stmt = stmt.where(
                DocumentChunk.chunk_metadata["is_parent"].astext != "true"
            )

            # Apply early metadata filters if provided
            if query.filters:
                f = query.filters
                if f.knowledge_type:
                    stmt = stmt.where(Document.knowledge_type == f.knowledge_type)
                if f.document_type:
                    stmt = stmt.where(Document.document_type == f.document_type)
                if f.source:
                    stmt = stmt.where(Document.source == f.source)
                if f.document_id:
                    stmt = stmt.where(Document.id == f.document_id)
                if f.document_version_id:
                    stmt = stmt.where(DocumentVersion.id == f.document_version_id)

            # Order by cosine distance ascending (lowest distance first), tiebreak by chunk_index
            stmt = stmt.order_by(distance.asc(), DocumentChunk.chunk_index.asc())
            stmt = stmt.limit(query.top_k)

            rows = self.session.execute(stmt).all()

            results: List[RetrievalResult] = []
            rank = 1

            for row in rows:
                raw_distance = float(row.distance) if row.distance is not None else 1.0
                score = round(1.0 - raw_distance, 6)

                # Skip results below minimum similarity score threshold
                if query.min_score is not None and score < query.min_score:
                    continue

                source_meta: Dict[str, Any] = {
                    "title": row.title,
                    "document_type": row.document_type,
                    "knowledge_type": row.knowledge_type,
                    "source": row.document_source,
                    "page": row.page,
                    "section": row.section,
                    "chunk_index": row.chunk_index,
                    "metadata": row.chunk_metadata,
                }

                results.append(
                    RetrievalResult(
                        chunk_id=UUID(str(row.chunk_id)),
                        document_id=UUID(str(row.document_id)),
                        document_version_id=UUID(str(row.document_version_id)),
                        content=row.content,
                        score=score,
                        rank=rank,
                        source=source_meta,
                    )
                )
                rank += 1

            return results

        except Exception as exc:
            if isinstance(exc, (RetrievalError,)):
                raise
            raise RetrievalError(f"Vector retrieval query failed: {exc}") from exc
