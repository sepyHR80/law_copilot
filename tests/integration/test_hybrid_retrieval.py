"""Integration tests for Stage 12 — Hybrid Retrieval + RRF against PostgreSQL."""

from uuid import uuid4
import pytest
from sqlalchemy.orm import Session

from app.domain.retrieval.models import HybridSearchQuery, RetrievalFilter
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion
from app.infrastructure.db.session import SessionLocal
from app.retrieval.hybrid import HybridSearchService


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
def seeded_hybrid_data(db_session: Session):
    """Seed documents with both vector embeddings and keyword text."""
    doc_factual_id = uuid4()
    doc_stylistic_id = uuid4()
    ver_factual_id = uuid4()
    ver_stylistic_id = uuid4()

    # Document 1: Factual NDA
    doc1 = Document(
        id=doc_factual_id,
        title="Non-Disclosure and Consulting Agreement",
        document_type="contract",
        knowledge_type="factual",
        source="corp_legal",
    )
    ver1 = DocumentVersion(
        id=ver_factual_id,
        document_id=doc_factual_id,
        version="v1",
        storage_key=f"documents/{doc_factual_id}/v1/nda.pdf",
        checksum="checksum_nda",
    )

    # Document 2: Stylistic reminder letter
    doc2 = Document(
        id=doc_stylistic_id,
        title="Formal NDA Reminder Letter",
        document_type="letter",
        knowledge_type="stylistic",
        source="legal_ops",
    )
    ver2 = DocumentVersion(
        id=ver_stylistic_id,
        document_id=doc_stylistic_id,
        version="v1",
        storage_key=f"documents/{doc_stylistic_id}/v1/reminder.pdf",
        checksum="checksum_reminder",
    )

    vec_confidentiality = [0.0] * 1536
    vec_confidentiality[0] = 1.0

    vec_arbitration = [0.0] * 1536
    vec_arbitration[1] = 1.0

    # Chunk 1: Matches BOTH vector AND exact lexical terms
    chunk1_id = uuid4()
    chunk1 = DocumentChunk(
        id=chunk1_id,
        document_version_id=ver_factual_id,
        content="Section 9.2: Confidentiality obligations and non-disclosure shall survive for 5 years.",
        page=4,
        section="Section 9",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_confidentiality,
    )

    # Chunk 2: Matches only vector or lexical partially (arbitration)
    chunk2_id = uuid4()
    chunk2 = DocumentChunk(
        id=chunk2_id,
        document_version_id=ver_factual_id,
        content="Section 15: Governing law and arbitration clauses in New York.",
        page=6,
        section="Section 15",
        chunk_index=1,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_arbitration,
    )

    # Chunk 3: Stylistic chunk matching vector and some keywords
    chunk3_id = uuid4()
    chunk3 = DocumentChunk(
        id=chunk3_id,
        document_version_id=ver_stylistic_id,
        content="Please be advised that your confidentiality obligations remain active and enforceable.",
        page=1,
        section="Notice",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
        embedding=vec_confidentiality,
    )

    db_session.add_all([doc1, ver1, chunk1, chunk2, doc2, ver2, chunk3])
    db_session.commit()

    yield {
        "doc_factual_id": doc_factual_id,
        "doc_stylistic_id": doc_stylistic_id,
        "chunk1_id": chunk1_id,
        "chunk2_id": chunk2_id,
        "chunk3_id": chunk3_id,
        "vec_confidentiality": vec_confidentiality,
        "vec_arbitration": vec_arbitration,
    }

    # Cleanup
    db_session.delete(doc1)
    db_session.delete(doc2)
    db_session.commit()


class TestHybridSearchIntegration:
    @pytest.mark.asyncio
    async def test_hybrid_search_boosts_overlapping_matches_to_rank_1(
        self, db_session: Session, seeded_hybrid_data
    ):
        """A chunk matching both semantic vector and exact lexical terms ranks highest."""
        service = HybridSearchService(session=db_session)

        query = HybridSearchQuery(
            text_query="Section 9.2 Confidentiality obligations survive",
            vector=seeded_hybrid_data["vec_confidentiality"],
            top_k=5,
            candidate_k=10,
            rrf_k=60,
        )
        results = await service.search(query)

        assert len(results) >= 1
        # Chunk 1 matches both vector and exact citation/phrase -> must be Rank 1!
        assert results[0].chunk_id == seeded_hybrid_data["chunk1_id"]
        assert results[0].rank == 1
        # Score must be fused from both retrievers: > (1/61)
        assert results[0].score > (1.0 / 61.0)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_metadata_filters(
        self, db_session: Session, seeded_hybrid_data
    ):
        """Metadata filters are applied across both retrieval modalities in hybrid search."""
        service = HybridSearchService(session=db_session)

        query = HybridSearchQuery(
            text_query="confidentiality obligations",
            vector=seeded_hybrid_data["vec_confidentiality"],
            top_k=5,
            filters=RetrievalFilter(knowledge_type="stylistic"),
        )
        results = await service.search(query)

        assert len(results) == 1
        assert results[0].chunk_id == seeded_hybrid_data["chunk3_id"]
        assert results[0].source["knowledge_type"] == "stylistic"

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_top_k(
        self, db_session: Session, seeded_hybrid_data
    ):
        """Hybrid search strictly returns only the top_k candidates."""
        service = HybridSearchService(session=db_session)

        query = HybridSearchQuery(
            text_query="confidentiality obligations",
            vector=seeded_hybrid_data["vec_confidentiality"],
            top_k=1,
            candidate_k=10,
        )
        results = await service.search(query)

        assert len(results) == 1
        assert results[0].rank == 1
