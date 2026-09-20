"""Hybrid retrieval service combining Vector Search and PostgreSQL Full-Text Search."""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.domain.embeddings.protocol import EmbeddingProvider
from app.domain.retrieval.exceptions import InvalidQueryError
from app.domain.retrieval.models import (
    HybridSearchQuery,
    LexicalSearchQuery,
    RetrievalResult,
    VectorSearchQuery,
)
from app.domain.retrieval.protocol import (
    HybridRetrieverProtocol,
    LexicalRetrieverProtocol,
    VectorRetrieverProtocol,
)
from app.infrastructure.db.repositories.lexical_search import PgLexicalRetriever
from app.infrastructure.db.repositories.vector_search import PgVectorRetriever
from app.retrieval.fusion import reciprocal_rank_fusion


class HybridSearchService(HybridRetrieverProtocol):
    """Orchestrates hybrid retrieval using vector search, lexical FTS, and RRF."""

    def __init__(
        self,
        vector_retriever: Optional[VectorRetrieverProtocol] = None,
        lexical_retriever: Optional[LexicalRetrieverProtocol] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        session: Optional[Session] = None,
    ) -> None:
        if session is not None:
            self._vector_retriever = vector_retriever or PgVectorRetriever(session=session)
            self._lexical_retriever = lexical_retriever or PgLexicalRetriever(session=session)
        elif vector_retriever is not None and lexical_retriever is not None:
            self._vector_retriever = vector_retriever
            self._lexical_retriever = lexical_retriever
        else:
            raise ValueError(
                "Either a database session or both vector_retriever and lexical_retriever must be provided."
            )

        self._embedding_provider = embedding_provider

    async def search(self, query: HybridSearchQuery) -> List[RetrievalResult]:
        """Execute hybrid search using Vector Retrieval, FTS, and Reciprocal Rank Fusion.

        Args:
            query: HybridSearchQuery with query text, optional precomputed vector, and parameters.

        Returns:
            List of RetrievalResult objects ordered by fused RRF rank.
        """
        if query.top_k < 1:
            raise InvalidQueryError("top_k must be at least 1.")

        vector = query.vector
        if vector is None:
            if self._embedding_provider is not None:
                embeddings = await self._embedding_provider.embed_texts([query.text_query])
                vector = embeddings[0]
            else:
                raise InvalidQueryError(
                    "Either a precomputed query vector or an EmbeddingProvider must be supplied."
                )

        # 1. Execute vector retrieval for candidate pool
        vec_query = VectorSearchQuery(
            vector=vector,
            top_k=query.candidate_k,
            filters=query.filters,
        )
        vec_results = self._vector_retriever.search(vec_query)

        # 2. Execute lexical full-text retrieval for candidate pool
        lex_query = LexicalSearchQuery(
            query=query.text_query,
            top_k=query.candidate_k,
            filters=query.filters,
        )
        lex_results = self._lexical_retriever.search(lex_query)

        # 3. Fuse rankings via Reciprocal Rank Fusion (RRF)
        fused_results = reciprocal_rank_fusion(
            [vec_results, lex_results],
            k=query.rrf_k,
        )

        # 4. Return top_k candidates
        return fused_results[: query.top_k]
