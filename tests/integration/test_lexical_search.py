"""Integration tests for Stage 11 — PostgreSQL Full-Text Retrieval (FTS)."""

from uuid import uuid4
import pytest
from sqlalchemy.orm import Session

from app.domain.retrieval.models import LexicalSearchQuery, RetrievalFilter
from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion
from app.infrastructure.db.repositories.lexical_search import PgLexicalRetriever
from app.infrastructure.db.session import SessionLocal
from app.retrieval.lexical import LexicalSearchService


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
def seeded_lexical_data(db_session: Session):
    """Seed documents, versions, and chunks for lexical full-text search."""
    doc_factual_id = uuid4()
    doc_stylistic_id = uuid4()
    ver_factual_id = uuid4()
    ver_stylistic_id = uuid4()

    # Document 1: Corporate Bylaws (factual)
    doc1 = Document(
        id=doc_factual_id,
        title="Corporate Governance Bylaws",
        document_type="bylaws",
        knowledge_type="factual",
        source="corp_legal",
    )
    ver1 = DocumentVersion(
        id=ver_factual_id,
        document_id=doc_factual_id,
        version="v1",
        storage_key=f"documents/{doc_factual_id}/v1/bylaws.pdf",
        checksum="checksum_bylaws",
    )

    # Document 2: Legal letter template (stylistic)
    doc2 = Document(
        id=doc_stylistic_id,
        title="Formal Notice of Violation",
        document_type="letter",
        knowledge_type="stylistic",
        source="legal_ops",
    )
    ver2 = DocumentVersion(
        id=ver_stylistic_id,
        document_id=doc_stylistic_id,
        version="v1",
        storage_key=f"documents/{doc_stylistic_id}/v1/notice.pdf",
        checksum="checksum_notice",
    )

    # Chunk 1: Exact article citation
    chunk1_id = uuid4()
    chunk1 = DocumentChunk(
        id=chunk1_id,
        document_version_id=ver_factual_id,
        content="Article 7.3: The Board of Directors shall hold meetings on a quarterly basis in Delaware.",
        page=3,
        section="Article 7: Governance",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
    )

    # Chunk 2: Indemnification clause
    chunk2_id = uuid4()
    chunk2 = DocumentChunk(
        id=chunk2_id,
        document_version_id=ver_factual_id,
        content="Section 14: Comprehensive indemnification of officers against civil liabilities.",
        page=5,
        section="Section 14: Liability",
        chunk_index=1,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
    )

    # Chunk 3: Parent chunk (should be excluded)
    chunk3_id = uuid4()
    chunk3 = DocumentChunk(
        id=chunk3_id,
        document_version_id=ver_factual_id,
        content="Chapter VII: Summary of Board of Directors meetings Article 7.3 provisions.",
        page=3,
        section="Chapter VII",
        chunk_index=2,
        chunk_metadata={"retrieval_unit": False, "is_parent": True},
    )

    # Chunk 4: Stylistic letter chunk mentioning Board of Directors
    chunk4_id = uuid4()
    chunk4 = DocumentChunk(
        id=chunk4_id,
        document_version_id=ver_stylistic_id,
        content="We hereby notify the Board of Directors regarding non-compliance with scheduled meetings.",
        page=1,
        section="Notification",
        chunk_index=0,
        chunk_metadata={"retrieval_unit": True, "is_parent": False},
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
    }

    # Cleanup
    db_session.delete(doc1)
    db_session.delete(doc2)
    db_session.commit()


class TestPgLexicalRetrieverIntegration:
    def test_exact_term_retrieval_matches_specific_chunk(
        self, db_session: Session, seeded_lexical_data
    ):
        """Search for exact legal terms ('indemnification', 'Delaware') retrieves matching chunks."""
        retriever = PgLexicalRetriever(session=db_session)
        service = LexicalSearchService(retriever=retriever)

        # Search for unique legal term 'indemnification'
        query = LexicalSearchQuery(query="indemnification", top_k=5)
        results = service.search(query)

        assert len(results) == 1
        assert results[0].chunk_id == seeded_lexical_data["chunk2_id"]
        assert results[0].rank == 1
        assert results[0].score > 0.0

    def test_density_ranking_prioritizes_higher_match_density(
        self, db_session: Session, seeded_lexical_data
    ):
        """Query terms appearing in higher density or phrase context rank first."""
        retriever = PgLexicalRetriever(session=db_session)

        # Chunk 1 has 'Board of Directors shall hold meetings on a quarterly basis in Delaware'
        query = LexicalSearchQuery(query="quarterly meetings Delaware", top_k=5)
        results = retriever.search(query)

        assert len(results) >= 1
        assert results[0].chunk_id == seeded_lexical_data["chunk1_id"]

    def test_parent_chunks_are_excluded(
        self, db_session: Session, seeded_lexical_data
    ):
        """Parent structural chunks are excluded from lexical search results."""
        retriever = PgLexicalRetriever(session=db_session)
        query = LexicalSearchQuery(query="Board of Directors", top_k=10)
        results = retriever.search(query)

        result_ids = {r.chunk_id for r in results}
        assert seeded_lexical_data["chunk3_id"] not in result_ids

    def test_filter_by_knowledge_type(
        self, db_session: Session, seeded_lexical_data
    ):
        """Filters restrict candidates to factual or stylistic knowledge."""
        retriever = PgLexicalRetriever(session=db_session)

        # Both Chunk 1 and Chunk 4 mention 'meetings' and 'Board of Directors'
        query_factual = LexicalSearchQuery(
            query="Board Directors meetings",
            top_k=5,
            filters=RetrievalFilter(knowledge_type="factual"),
        )
        results_factual = retriever.search(query_factual)
        result_ids_factual = {r.chunk_id for r in results_factual}

        assert seeded_lexical_data["chunk1_id"] in result_ids_factual
        assert seeded_lexical_data["chunk4_id"] not in result_ids_factual

        query_stylistic = LexicalSearchQuery(
            query="Board Directors meetings",
            top_k=5,
            filters=RetrievalFilter(knowledge_type="stylistic"),
        )
        results_stylistic = retriever.search(query_stylistic)
        result_ids_stylistic = {r.chunk_id for r in results_stylistic}

        assert seeded_lexical_data["chunk4_id"] in result_ids_stylistic
        assert seeded_lexical_data["chunk1_id"] not in result_ids_stylistic

    def test_filter_by_document_type(
        self, db_session: Session, seeded_lexical_data
    ):
        """Filters match only the requested document_type."""
        retriever = PgLexicalRetriever(session=db_session)
        query = LexicalSearchQuery(
            query="Board Directors",
            top_k=5,
            filters=RetrievalFilter(document_type="letter"),
        )
        results = retriever.search(query)

        assert len(results) == 1
        assert results[0].chunk_id == seeded_lexical_data["chunk4_id"]

    def test_no_match_returns_empty_list(
        self, db_session: Session, seeded_lexical_data
    ):
        """Queries with no matching terms return empty list gracefully."""
        retriever = PgLexicalRetriever(session=db_session)
        query = LexicalSearchQuery(query="supercalifragilistic nonexistentterm", top_k=5)
        results = retriever.search(query)

        assert results == []

    def test_unicode_and_legal_punctuation_handling(
        self, db_session: Session, seeded_lexical_data
    ):
        """Queries containing legal punctuation, symbols (§, -, :), and quotes execute cleanly."""
        retriever = PgLexicalRetriever(session=db_session)
        # Complex legal query string targeting Chunk 1
        query = LexicalSearchQuery(
            query='Article 7.3: "Board of Directors" § Delaware',
            top_k=5,
        )
        results = retriever.search(query)

        assert len(results) >= 1
        assert results[0].chunk_id == seeded_lexical_data["chunk1_id"]
