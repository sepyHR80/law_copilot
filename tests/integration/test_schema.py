"""Integration tests for Stage 03 database schema.

These tests verify the SQLAlchemy models, Alembic migration, and
all expected database tables/constraints exist and work correctly.
"""

import uuid
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.infrastructure.db.session import SessionLocal, engine


EXPECTED_TABLES = [
    "users",
    "documents",
    "document_versions",
    "document_chunks",
    "conversations",
    "messages",
    "memories",
    "evaluation_cases",
]


def _uuid() -> str:
    """Generate a unique string for test isolation."""
    return str(uuid.uuid4())


def test_all_expected_tables_exist() -> None:
    """All 8 Stage 03 tables must exist in the database."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' "
                "ORDER BY tablename"
            )
        )
        existing = {row[0] for row in result}

    for table in EXPECTED_TABLES:
        assert table in existing, f"Table '{table}' does not exist"


def test_pgvector_extension_exists() -> None:
    """The pgvector extension must be installed."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        )
        assert result.scalar_one_or_none() == 1


def test_embedding_vector_column_exists() -> None:
    """document_chunks must have an embedding column of type vector."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_name = 'document_chunks' "
                "AND column_name = 'embedding'"
            )
        )
        row = result.fetchone()
        assert row is not None, "embedding column missing from document_chunks"
        assert row[1] == "USER-DEFINED" or "vector" in row[1].lower()


def test_embedding_dimension_is_1536() -> None:
    """The embedding column must be Vector(1536)."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) "
                "FROM pg_attribute "
                "WHERE attrelid = 'document_chunks'::regclass "
                "AND attname = 'embedding'"
            )
        )
        type_str = result.scalar_one()
        assert "vector(1536)" in type_str.lower(), f"Expected vector(1536), got {type_str}"


def test_foreign_keys_exist() -> None:
    """Key foreign key relationships must exist."""
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tc.constraint_name, kcu.column_name, "
                "ccu.table_name AS foreign_table, ccu.column_name AS foreign_column "
                "FROM information_schema.table_constraints tc "
                "JOIN information_schema.key_column_usage kcu "
                " ON tc.constraint_name = kcu.constraint_name "
                "JOIN information_schema.constraint_column_usage ccu "
                " ON tc.constraint_name = ccu.constraint_name "
                "WHERE tc.constraint_type = 'FOREIGN KEY' "
                "ORDER BY tc.constraint_name"
            )
        )
        fks = result.fetchall()
        fk_pairs = {(row[1], row[2], row[3]) for row in fks}

    expected = [
        ("document_id", "documents", "id"),
        ("document_version_id", "document_versions", "id"),
        ("conversation_id", "conversations", "id"),
        ("user_id", "users", "id"),
    ]
    for col, ft, fc in expected:
        assert (col, ft, fc) in fk_pairs, f"FK {col} -> {ft}.{fc} missing"


def test_document_version_unique_constraint() -> None:
    """document_versions must enforce unique (document_id, version)."""
    from app.infrastructure.db.models.document import Document, DocumentVersion

    with SessionLocal() as db:
        doc = Document(
            title=f"Test Law {_uuid()}",
            document_type="law",
            knowledge_type="factual",
            source="official",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        v1 = DocumentVersion(
            document_id=doc.id,
            version="1.0",
            storage_key=f"key-{_uuid()}",
            checksum="abc123",
        )
        db.add(v1)
        db.commit()

        v2 = DocumentVersion(
            document_id=doc.id,
            version="1.0",
            storage_key=f"key-{_uuid()}",
            checksum="def456",
        )
        db.add(v2)
        try:
            db.commit()
            assert False, "Expected IntegrityError for duplicate (document_id, version)"
        except IntegrityError:
            db.rollback()


def test_document_chunk_unique_constraint() -> None:
    """document_chunks must enforce unique (document_version_id, chunk_index)."""
    from app.infrastructure.db.models.document import Document, DocumentVersion, DocumentChunk

    with SessionLocal() as db:
        doc = Document(
            title=f"Test Reg {_uuid()}",
            document_type="regulation",
            knowledge_type="factual",
            source="internal",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        ver = DocumentVersion(
            document_id=doc.id,
            version="2.0",
            storage_key=f"regkey-{_uuid()}",
            checksum="xyz789",
        )
        db.add(ver)
        db.commit()
        db.refresh(ver)

        c1 = DocumentChunk(
            document_version_id=ver.id,
            content="First chunk",
            chunk_index=0,
            chunk_metadata={},
            embedding=[0.1] * 1536,
        )
        db.add(c1)
        db.commit()

        c2 = DocumentChunk(
            document_version_id=ver.id,
            content="Second chunk",
            chunk_index=0,
            chunk_metadata={},
            embedding=[0.2] * 1536,
        )
        db.add(c2)
        try:
            db.commit()
            assert False, "Expected IntegrityError for duplicate (document_version_id, chunk_index)"
        except IntegrityError:
            db.rollback()


def test_memory_unique_constraint() -> None:
    """memories must enforce unique (user_id, type, key)."""
    from app.infrastructure.db.models.user import User
    from app.infrastructure.db.models.memory import Memory

    with SessionLocal() as db:
        user = User(language="en", external_id=f"ext-{_uuid()}")
        db.add(user)
        db.commit()
        db.refresh(user)

        m1 = Memory(
            user_id=user.id,
            type="preference",
            key="tone",
            value={"tone": "formal"},
        )
        db.add(m1)
        db.commit()

        m2 = Memory(
            user_id=user.id,
            type="preference",
            key="tone",
            value={"tone": "casual"},
        )
        db.add(m2)
        try:
            db.commit()
            assert False, "Expected IntegrityError for duplicate (user_id, type, key)"
        except IntegrityError:
            db.rollback()


def test_user_external_id_unique() -> None:
    """users.external_id unique constraint when non-null."""
    from app.infrastructure.db.models.user import User

    ext_id = f"ext-{_uuid()}"
    with SessionLocal() as db:
        u1 = User(language="en", external_id=ext_id)
        db.add(u1)
        db.commit()

        u2 = User(language="fa", external_id=ext_id)
        db.add(u2)
        try:
            db.commit()
            assert False, "Expected IntegrityError for duplicate external_id"
        except IntegrityError:
            db.rollback()


def test_insert_document_with_version_and_chunk() -> None:
    """Full insertion flow: document -> version -> chunk with embedding."""
    from app.infrastructure.db.models.document import Document, DocumentVersion, DocumentChunk

    with SessionLocal() as db:
        doc = Document(
            title=f"Civil Code {_uuid()}",
            document_type="law",
            knowledge_type="factual",
            source="official",
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        ver = DocumentVersion(
            document_id=doc.id,
            version="2025",
            storage_key=f"laws/civil-{_uuid()}.pdf",
            checksum="sha256:abc",
        )
        db.add(ver)
        db.commit()
        db.refresh(ver)

        chunk = DocumentChunk(
            document_version_id=ver.id,
            content="Article 1: ...",
            page=1,
            section="General Provisions",
            chunk_index=0,
            chunk_metadata={"page": 1},
            embedding=[0.01] * 1536,
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        assert chunk.id is not None
        assert len(chunk.embedding) == 1536
        assert chunk.chunk_metadata == {"page": 1}


def test_insert_conversation_with_message() -> None:
    """Conversation + message roundtrip."""
    from app.infrastructure.db.models.user import User
    from app.infrastructure.db.models.conversation import Conversation, Message

    with SessionLocal() as db:
        user = User(language="en", external_id=f"conv-{_uuid()}")
        db.add(user)
        db.commit()
        db.refresh(user)

        conv = Conversation(user_id=user.id, title=f"Legal Q&A {_uuid()}")
        db.add(conv)
        db.commit()
        db.refresh(conv)

        msg = Message(
            conversation_id=conv.id,
            role="user",
            content="What is the statute of limitations?",
        )
        db.add(msg)
        db.commit()
        db.refresh(msg)

        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "What is the statute of limitations?"


def test_insert_evaluation_case() -> None:
    """Evaluation case roundtrip."""
    from app.infrastructure.db.models.evaluation import EvaluationCase

    with SessionLocal() as db:
        case = EvaluationCase(
            question=f"What is the penalty? {_uuid()}",
            reference_answer="The penalty is...",
            expected_sources=["law-1", "law-2"],
            expected_style="formal",
        )
        db.add(case)
        db.commit()
        db.refresh(case)

        assert case.id is not None
        assert case.expected_sources == ["law-1", "law-2"]
