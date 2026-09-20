"""Integration tests for Stage 17 — Memory persistence against PostgreSQL."""

from uuid import uuid4
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.domain.memory.models import MemoryUpsert
from app.infrastructure.db.models.user import User
from app.infrastructure.db.repositories.memory import (
    SQLAlchemyConversationRepository,
    SQLAlchemyMemoryRepository,
)
from app.infrastructure.db.session import SessionLocal
from app.main import app
from app.memory.service import MemoryService


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
def test_users(db_session: Session):
    """Seed two isolated users for testing isolation."""
    u1 = User(id=uuid4(), external_id=f"user_a_{uuid4().hex[:8]}", language="en")
    u2 = User(id=uuid4(), external_id=f"user_b_{uuid4().hex[:8]}", language="de")
    db_session.add_all([u1, u2])
    db_session.commit()
    return u1, u2


class TestMemoryPersistence:
    """Test long-term memory operations and PostgreSQL ON CONFLICT upserts."""

    def test_upsert_memory_creates_and_updates_on_conflict(self, db_session: Session, test_users):
        u1, _ = test_users
        repo = SQLAlchemyMemoryRepository(db_session)

        # 1. Insert initial preference
        m1 = repo.upsert_memory(
            user_id=u1.id,
            memory=MemoryUpsert(
                type="preference",
                key="tone",
                value={"text": "Formal and rigorous"},
                confidence=0.9,
            ),
        )
        assert m1.user_id == u1.id
        assert m1.key == "tone"
        assert m1.value["text"] == "Formal and rigorous"

        # 2. Upsert same (user_id, type, key) with new value
        m2 = repo.upsert_memory(
            user_id=u1.id,
            memory=MemoryUpsert(
                type="preference",
                key="tone",
                value={"text": "Concise and executive"},
                confidence=0.95,
            ),
        )
        # ID is maintained or updated atomically
        assert m2.value["text"] == "Concise and executive"
        assert m2.confidence == 0.95

        # Query all memories for user 1 - should only be 1 record
        all_mems = repo.list_memories(user_id=u1.id)
        assert len(all_mems) == 1
        assert all_mems[0].value["text"] == "Concise and executive"

    def test_user_isolation(self, db_session: Session, test_users):
        u1, u2 = test_users
        repo = SQLAlchemyMemoryRepository(db_session)

        repo.upsert_memory(
            user_id=u1.id,
            memory=MemoryUpsert(type="jurisdiction", key="region", value={"text": "California"}),
        )
        repo.upsert_memory(
            user_id=u2.id,
            memory=MemoryUpsert(type="jurisdiction", key="region", value={"text": "New York"}),
        )

        u1_mems = repo.list_memories(user_id=u1.id)
        u2_mems = repo.list_memories(user_id=u2.id)

        assert len(u1_mems) == 1
        assert u1_mems[0].value["text"] == "California"

        assert len(u2_mems) == 1
        assert u2_mems[0].value["text"] == "New York"

    def test_delete_memory(self, db_session: Session, test_users):
        u1, _ = test_users
        repo = SQLAlchemyMemoryRepository(db_session)

        repo.upsert_memory(
            user_id=u1.id,
            memory=MemoryUpsert(type="temp", key="k1", value={"text": "v1"}),
        )
        assert repo.get_memory(u1.id, "temp", "k1") is not None

        deleted = repo.delete_memory(u1.id, "temp", "k1")
        assert deleted is True
        assert repo.get_memory(u1.id, "temp", "k1") is None

        # Deleting again returns False
        assert repo.delete_memory(u1.id, "temp", "k1") is False


class TestConversationPersistence:
    """Test short-term conversation sessions and message turn ordering."""

    def test_conversation_and_messages_lifecycle(self, db_session: Session, test_users):
        u1, _ = test_users
        conv_repo = SQLAlchemyConversationRepository(db_session)

        conv_id = conv_repo.create_conversation(user_id=u1.id, title="Employment Review")
        assert conv_id is not None

        # Add message turns
        m1 = conv_repo.add_message(conv_id, role="user", content="What is the severance formula?")
        m2 = conv_repo.add_message(conv_id, role="assistant", content="Severance is 2 weeks per year of service.")

        assert m1.role == "user"
        assert m2.role == "assistant"

        detail = conv_repo.get_conversation(conv_id)
        assert detail is not None
        assert detail.title == "Employment Review"
        assert len(detail.messages) == 2
        assert detail.messages[0].content == "What is the severance formula?"
        assert detail.messages[1].content == "Severance is 2 weeks per year of service."


class TestMemoryApiEndpoints:
    """Test FastAPI memory and conversation endpoints."""

    def test_memory_and_conversation_api(self, test_users):
        u1, _ = test_users
        client = TestClient(app)

        # 1. Upsert memory
        upsert_res = client.post(
            f"/api/v1/memories?user_id={u1.id}",
            json={"type": "preference", "key": "language", "value": {"code": "en-US"}, "confidence": 1.0},
        )
        assert upsert_res.status_code == 200
        assert upsert_res.json()["key"] == "language"

        # 2. List memories
        list_res = client.get(f"/api/v1/memories?user_id={u1.id}")
        assert list_res.status_code == 200
        assert len(list_res.json()) >= 1

        # 3. Create conversation
        conv_res = client.post(f"/api/v1/conversations?user_id={u1.id}", json={"title": "Contract QA"})
        assert conv_res.status_code == 201
        cid = conv_res.json()["conversation_id"]

        # 4. Add message
        msg_res = client.post(
            f"/api/v1/conversations/{cid}/messages",
            json={"role": "user", "content": "Analyze clause 5."},
        )
        assert msg_res.status_code == 201
        assert msg_res.json()["content"] == "Analyze clause 5."

        # 5. List messages
        msgs_res = client.get(f"/api/v1/conversations/{cid}/messages")
        assert msgs_res.status_code == 200
        assert len(msgs_res.json()) == 1

        # 6. Delete memory
        del_res = client.delete(f"/api/v1/memories/preference/language?user_id={u1.id}")
        assert del_res.status_code == 204
