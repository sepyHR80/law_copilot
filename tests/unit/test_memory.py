"""Unit tests for Stage 17 — Memory service, models, and safety."""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4
from unittest.mock import AsyncMock
import pytest

from app.agent.graph import LegalAgent, build_legal_agent_graph
from app.agent.models import AgentRequest
from app.domain.memory.models import (
    ConversationDetail,
    ConversationMessage,
    MemoryRecord,
    MemoryUpsert,
)
from app.domain.memory.protocol import (
    ConversationRepositoryProtocol,
    LongTermMemoryRepositoryProtocol,
)
from app.domain.retrieval.models import RetrievalResult
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.reranking.fake import FakeReranker
from app.llm.service import LLMService
from app.memory.service import MemoryService
from app.rag.context_builder import ContextBuilder


class FakeMemoryRepository(LongTermMemoryRepositoryProtocol):
    """In-memory fake repository for long-term memory."""

    def __init__(self) -> None:
        self.memories: dict[tuple[UUID, str, str], MemoryRecord] = {}

    def upsert_memory(self, user_id: UUID, memory: MemoryUpsert) -> MemoryRecord:
        key = (user_id, memory.type, memory.key)
        now = datetime.utcnow()
        if key in self.memories:
            rec = self.memories[key]
            rec.value = memory.value
            rec.confidence = memory.confidence
            rec.updated_at = now
        else:
            rec = MemoryRecord(
                id=uuid4(),
                user_id=user_id,
                type=memory.type,
                key=memory.key,
                value=memory.value,
                confidence=memory.confidence,
                created_at=now,
            )
            self.memories[key] = rec
        return rec

    def get_memory(self, user_id: UUID, type: str, key: str) -> Optional[MemoryRecord]:
        return self.memories.get((user_id, type, key))

    def list_memories(self, user_id: UUID, type: Optional[str] = None) -> List[MemoryRecord]:
        return [
            m for (uid, m_type, _), m in self.memories.items()
            if uid == user_id and (type is None or m_type == type)
        ]

    def delete_memory(self, user_id: UUID, type: str, key: str) -> bool:
        k = (user_id, type, key)
        if k in self.memories:
            del self.memories[k]
            return True
        return False


class FakeConversationRepository(ConversationRepositoryProtocol):
    """In-memory fake repository for conversations."""

    def __init__(self) -> None:
        self.conversations: dict[UUID, ConversationDetail] = {}

    def create_conversation(self, user_id: UUID, title: Optional[str] = None) -> UUID:
        cid = uuid4()
        self.conversations[cid] = ConversationDetail(
            id=cid,
            user_id=user_id,
            title=title,
            created_at=datetime.utcnow(),
            messages=[],
        )
        return cid

    def get_conversation(self, conversation_id: UUID) -> Optional[ConversationDetail]:
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: UUID, role: str, content: str) -> ConversationMessage:
        if conversation_id not in self.conversations:
            from app.domain.memory.exceptions import ConversationNotFoundError
            raise ConversationNotFoundError("Not found")
        msg = ConversationMessage(
            id=uuid4(),
            conversation_id=conversation_id,
            role=role,
            content=content,
            created_at=datetime.utcnow(),
        )
        self.conversations[conversation_id].messages.append(msg)
        return msg

    def list_messages(self, conversation_id: UUID, limit: int = 50) -> List[ConversationMessage]:
        conv = self.conversations.get(conversation_id)
        return conv.messages[:limit] if conv else []


class TestMemoryModelsAndScopedContext:
    """Test memory domain models and scoped context assembly."""

    def test_memory_upsert_and_record_models(self) -> None:
        uid = uuid4()
        upsert = MemoryUpsert(
            type="preference",
            key="preferred_language",
            value={"text": "German"},
            confidence=0.95,
        )
        assert upsert.type == "preference"
        assert upsert.key == "preferred_language"

        record = MemoryRecord(
            user_id=uid,
            type=upsert.type,
            key=upsert.key,
            value=upsert.value,
            confidence=upsert.confidence,
        )
        assert record.user_id == uid
        assert record.value["text"] == "German"

    def test_scoped_context_filters_and_marks_non_authoritative(self) -> None:
        fake_repo = FakeMemoryRepository()
        service = MemoryService(memory_repo=fake_repo)
        uid = uuid4()

        # Add scoped preference
        service.upsert_memory(
            user_id=uid,
            memory=MemoryUpsert(type="tone", key="formality", value={"text": "Formal legal drafting"}),
        )
        service.upsert_memory(
            user_id=uid,
            memory=MemoryUpsert(type="jurisdiction", key="region", value={"text": "Bavaria, Germany"}),
        )
        # Add internal/unrelated memory
        service.upsert_memory(
            user_id=uid,
            memory=MemoryUpsert(type="session_temp", key="tab", value={"text": "tab3"}),
        )

        context_str = service.build_scoped_context(user_id=uid)

        assert "NON-AUTHORITATIVE" in context_str
        assert "Formal legal drafting" in context_str
        assert "Bavaria, Germany" in context_str
        assert "tab3" not in context_str
        assert "NOT legal evidence" in context_str

    def test_scoped_context_empty_when_no_memories(self) -> None:
        fake_repo = FakeMemoryRepository()
        service = MemoryService(memory_repo=fake_repo)
        assert service.build_scoped_context(uuid4()) == ""


class TestMemorySafetyAndAgentIntegration:
    """Test that memory cannot become legal evidence or citations."""

    @pytest.mark.asyncio
    async def test_agent_loads_memory_context_and_preserves_grounding(self) -> None:
        uid = uuid4()
        fake_mem_repo = FakeMemoryRepository()
        mem_service = MemoryService(memory_repo=fake_mem_repo)

        # Set user preference in long-term memory
        mem_service.upsert_memory(
            user_id=uid,
            memory=MemoryUpsert(type="tone", key="style", value={"text": "Concise and formal"}),
        )

        cand = RetrievalResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            content="Statutory notice is 14 business days.",
            score=0.9,
            rank=1,
            source={"type": "hybrid", "page": 2, "section": "Notice"},
        )

        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        answer_json = json.dumps({
            "answer": "The statutory notice period is 14 business days.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 1, "snippet": "14 business days"}],
        })
        fake_llm = FakeLLMProvider(responses=[answer_json])

        graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
            memory_service=mem_service,
        )
        agent = LegalAgent(graph)

        res = await agent.run(AgentRequest(query="What is the notice period?", user_id=str(uid)))

        assert res.is_sufficient is True
        assert "14 business days" in res.response
        assert len(res.citations) == 1
        assert res.citations[0].chunk_id == cand.chunk_id
        # Verify prompt received memory context
        recorded_request = fake_llm.call_history[0]
        user_prompt_content = recorded_request.messages[-1].content
        assert "Concise and formal" in user_prompt_content
        assert "NON-AUTHORITATIVE" in user_prompt_content

    @pytest.mark.asyncio
    async def test_memory_cannot_become_citation(self) -> None:
        """Verify that citations can only reference document evidence, never memory context."""
        uid = uuid4()
        fake_mem_repo = FakeMemoryRepository()
        mem_service = MemoryService(memory_repo=fake_mem_repo)
        mem_service.upsert_memory(
            user_id=uid,
            memory=MemoryUpsert(type="case_context", key="client_fact", value={"text": "Client signed NDA on Jan 1"}),
        )

        cand = RetrievalResult(
            chunk_id=uuid4(),
            document_id=uuid4(),
            document_version_id=uuid4(),
            content="Contract requires written authorization.",
            score=0.85,
            rank=1,
            source={"type": "hybrid", "page": 1},
        )
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [cand]

        # LLM attempts to cite the memory context as Evidence 99
        bad_citation_json = json.dumps({
            "answer": "Client signed NDA on Jan 1 based on memory.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 99, "snippet": "Client signed NDA"}],
        })
        repaired_json = json.dumps({
            "answer": "Contract requires written authorization.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 1, "snippet": "written authorization"}],
        })
        fake_llm = FakeLLMProvider(responses=[bad_citation_json, repaired_json])

        graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
            memory_service=mem_service,
        )
        agent = LegalAgent(graph)

        res = await agent.run(AgentRequest(query="What is required?", user_id=str(uid), max_retries=1))

        # Memory was rejected as a citation; only verified document chunk accepted
        assert len(res.citations) == 1
        assert res.citations[0].chunk_id == cand.chunk_id
