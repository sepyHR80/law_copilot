"""PostgreSQL Full-Text Search (FTS) lexical retrieval repository."""

from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.retrieval.exceptions import (
    InvalidQueryError,
    RetrievalError,
)
from app.domain.retrieval.models import LexicalSearchQuery, RetrievalResult
from app.domain.retrieval.protocol import LexicalRetrieverProtocol
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion


class PgLexicalRetriever(LexicalRetrieverProtocol):
    """Lexical full-text retriever backed by PostgreSQL tsvector and tsquery."""

    def __init__(
        self,
        session: Session,
        language: str = "english",
    ) -> None:
        self.session = session
        self.language = language

    def search(self, query: LexicalSearchQuery) -> List[RetrievalResult]:
        """Execute full-text lexical search using websearch_to_tsquery and ts_rank_cd."""
        if query.top_k < 1:
            raise InvalidQueryError("top_k must be at least 1.")

        cleaned_query = query.query.strip()
        if not cleaned_query:
            return []

        try:
            # Parse user query using websearch_to_tsquery
            ts_query = func.websearch_to_tsquery(self.language, cleaned_query)
            ts_vector = func.to_tsvector(self.language, DocumentChunk.content)

            # Cover density ranking (flag 32 scales rank r to r/(r+1) for [0, 1] normalization)
            rank_expr = func.ts_rank_cd(ts_vector, ts_query, 32)

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
                    rank_expr.label("score"),
                )
                .join(DocumentVersion, DocumentChunk.document_version_id == DocumentVersion.id)
                .join(Document, DocumentVersion.document_id == Document.id)
                .where(ts_vector.op("@@")(ts_query))
                .where(DocumentChunk.chunk_metadata["is_parent"].astext != "true")
            )

            # Apply early metadata filters
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

            # Order by rank descending, tiebreak by chunk_index
            stmt = stmt.order_by(rank_expr.desc(), DocumentChunk.chunk_index.asc())
            stmt = stmt.limit(query.top_k)

            rows = self.session.execute(stmt).all()

            results: List[RetrievalResult] = []
            rank = 1

            for row in rows:
                score = round(float(row.score), 6) if row.score is not None else 0.0

                # Skip results below min_score threshold
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
            raise RetrievalError(f"Lexical retrieval query failed: {exc}") from exc
