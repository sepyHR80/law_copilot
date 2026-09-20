"""Integration tests for Stage 10 — Vector Retrieval against PostgreSQL + pgvector."""

from uuid import uuid4
import pytest
from sqlalchemy.orm import Session

from app.domain.retrieval.models import RetrievalFilter, VectorSearchQuery
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion
from app.infrastructure.db.repositories.vector_search import PgVectorRetriever
from app.infrastructure.db.session import SessionLocal
from app.retrieval.vector import VectorSearchService


@pytest.fixture
def db_session():
    """Database session fixture with rollback / cleanup."""
    session: Session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def seeded_vector_data(db_session: Session):
    """Seed documents, versions, and chunks with known 1536-d vectors."""
    doc_factual_id = uuid4()
    doc_stylistic_id = uuid4()
    ver_factual_id = uuid4()
    ver_stylistic_id = uuid4()

    # Document 1: Factual legal agreement
    doc1 = Document(
        id=doc_factual_id,
        title="Employment Contract Agreement",
        document_type="contract",
        knowledge_type="factual",
        source="internal_hr",
    )
    ver1 = DocumentVersion(
        id=ver_factual_id,
        document_id=doc_factual_id,
        version="v1",
        storage_key=f"documents/{doc_factual_id}/v1/contract.pdf",
        checksum="checksum123",
    )

    # Document 2: Stylistic complaint letter
    doc2 = Document(
        id=doc_stylistic_id,
        title="Formal Legal Notice Template",
        document_type="letter",
        knowledge_type="stylistic",
        source="internal_templates",
    )
    ver2 = DocumentVersion(
        id=ver_stylistic_id,
        document_id=doc_stylistic_id,
        version="v1",
        storage_key=f"documents/{doc_stylistic_id}/v1/notice.pdf",
        checksum="checksum456",
    )

    # Vector A: [1.0, 0.0, ..., 0.0]
    vec_target = [0.0] * 1536
    vec_target[0] = 1.0

    # Vector B: [0.0, 1.0, ..., 0.0] (orthogonal to target)
    vec_other = [0.0] * 1536
    vec_other[1] = 1.0

    # Chunk 1: Target factual chunk
    chunk1_id = uuid4()
    chunk1 = DocumentChunk(
        id=chunk1_id,
        document_version_id=ver_factual_id,
        content="Employee termination requires 30 days notice.",
        page=2,
        section="Article 4: Termination",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_target,
    )

    # Chunk 2: Other factual chunk
    chunk2_id = uuid4()
    chunk2 = DocumentChunk(
        id=chunk2_id,
        document_version_id=ver_factual_id,
        content="Compensation is payable on the last business day of each month.",
        page=1,
        section="Article 2: Compensation",
        chunk_index=1,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_other,
    )

    # Chunk 3: Structural parent chunk (should be excluded from retrieval)
    chunk3_id = uuid4()
    chunk3 = DocumentChunk(
        id=chunk3_id,
        document_version_id=ver_factual_id,
        content="Full Chapter 1 Subtree text...",
        page=1,
        section="Chapter 1",
        chunk_index=2,
        chunk_metadata={"retrieval_unit": False, "is_parent": True},
        embedding=vec_target,
    )

    # Chunk 4: Stylistic chunk with target vector
    chunk4_id = uuid4()
    chunk4 = DocumentChunk(
        id=chunk4_id,
        document_version_id=ver_stylistic_id,
        content="We hereby serve formal notice of termination in accordance with terms.",
        page=1,
        section="Opening",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_target,
    )

    db_session.add_all([doc1, ver1, chunk1, chunk2, chunk3, doc2, ver2, chunk4])
    db_session.commit()

    yield {
        "doc_factual_id": doc_factual_id,
        "doc_stylistic_id": doc_stylistic_id,
        "chunk1_id": chunk1_id,
        "chunk2_id": chunk2_id,
        "chunk3_id": chunk3_id,
        "chunk4_id": chunk4_id,
        "vec_target": vec_target,
        "vec_other": vec_other,
    }

    # Cleanup after test
    db_session.delete(doc1)
    db_session.delete(doc2)
    db_session.commit()


class TestPgVectorRetrieverIntegration:
    def test_vector_similarity_search_returns_closest_chunk_rank_1(
        self, db_session: Session, seeded_vector_data
    ):
        """Query vector returns closest embedding chunk as rank 1 with high similarity score."""
        retriever = PgVectorRetriever(session=db_session)
        service = VectorSearchService(retriever=retriever)

        query = VectorSearchQuery(vector=seeded_vector_data["vec_target"], top_k=5)
        results = service.search(query)

        assert len(results) >= 1
        # Top result must be chunk1 or chunk4 (both have target vector with distance 0 -> score 1.0)
        top_result = results[0]
        assert top_result.rank == 1
        assert top_result.score >= 0.99
        assert top_result.chunk_id in (
            seeded_vector_data["chunk1_id"],
            seeded_vector_data["chunk4_id"],
        )

    def test_parent_chunks_are_excluded_from_retrieval(
        self, db_session: Session, seeded_vector_data
    ):
        """Parent chunks (marked is_parent=true) are excluded from retrieval candidates."""
        retriever = PgVectorRetriever(session=db_session)
        query = VectorSearchQuery(vector=seeded_vector_data["vec_target"], top_k=10)
        results = retriever.search(query)

        result_chunk_ids = {r.chunk_id for r in results}
        assert seeded_vector_data["chunk3_id"] not in result_chunk_ids

    def test_filter_by_knowledge_type(
        self, db_session: Session, seeded_vector_data
    ):
        """Filtering by knowledge_type narrows candidates to factual or stylistic."""
        retriever = PgVectorRetriever(session=db_session)

        # Factual only
        query_factual = VectorSearchQuery(
            vector=seeded_vector_data["vec_target"],
            top_k=5,
            filters=RetrievalFilter(knowledge_type="factual"),
        )
        results_factual = retriever.search(query_factual)
        result_ids_factual = {r.chunk_id for r in results_factual}

        assert seeded_vector_data["chunk1_id"] in result_ids_factual
        assert seeded_vector_data["chunk4_id"] not in result_ids_factual

        # Stylistic only
        query_stylistic = VectorSearchQuery(
            vector=seeded_vector_data["vec_target"],
            top_k=5,
            filters=RetrievalFilter(knowledge_type="stylistic"),
        )
        results_stylistic = retriever.search(query_stylistic)
        result_ids_stylistic = {r.chunk_id for r in results_stylistic}

        assert seeded_vector_data["chunk4_id"] in result_ids_stylistic
        assert seeded_vector_data["chunk1_id"] not in result_ids_stylistic

    def test_filter_by_document_type(
        self, db_session: Session, seeded_vector_data
    ):
        """Filtering by document_type matches only the requested type."""
        retriever = PgVectorRetriever(session=db_session)
        query = VectorSearchQuery(
            vector=seeded_vector_data["vec_target"],
            top_k=5,
            filters=RetrievalFilter(document_type="letter"),
        )
        results = retriever.search(query)

        assert len(results) == 1
        assert results[0].chunk_id == seeded_vector_data["chunk4_id"]

    def test_filter_by_document_id(
        self, db_session: Session, seeded_vector_data
    ):
        """Filtering by document_id limits search to a single document."""
        retriever = PgVectorRetriever(session=db_session)
        query = VectorSearchQuery(
            vector=seeded_vector_data["vec_target"],
            top_k=5,
            filters=RetrievalFilter(document_id=seeded_vector_data["doc_factual_id"]),
        )
        results = retriever.search(query)

        assert all(r.document_id == seeded_vector_data["doc_factual_id"] for r in results)

    def test_min_score_threshold_filters_distant_vectors(
        self, db_session: Session, seeded_vector_data
    ):
        """Results with score below min_score threshold are omitted."""
        retriever = PgVectorRetriever(session=db_session)
        # Target vector has similarity ~1.0, other vector has orthogonal similarity ~0.0
        query = VectorSearchQuery(
            vector=seeded_vector_data["vec_target"],
            top_k=5,
            min_score=0.8,
            filters=RetrievalFilter(document_id=seeded_vector_data["doc_factual_id"]),
        )
        results = retriever.search(query)

        # Chunk 1 has score ~1.0, Chunk 2 has score ~0.0
        assert len(results) == 1
        assert results[0].chunk_id == seeded_vector_data["chunk1_id"]
        assert results[0].score >= 0.8
