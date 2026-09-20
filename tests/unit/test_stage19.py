"""Unit tests for Stage 19 — MCP + External Search + Verification."""

import json
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4
import pytest

from app.agent.edges import route_evidence, route_external_evidence, route_verification
from app.agent.graph import LegalAgent, build_legal_agent_graph
from app.agent.models import AgentRequest
from app.agent.state import AgentState
from app.domain.mcp.exceptions import MCPToolExecutionError
from app.domain.mcp.models import (
    MCPEvidenceItem,
    MCPSearchKnowledgeRequest,
    MCPSearchKnowledgeResponse,
)
from app.domain.rag.models import Citation, EvidenceItem
from app.domain.retrieval.models import RetrievalResult
from app.domain.search.models import ValidatedSource, WebSearchResult
from app.domain.verification.models import VerificationResult
from app.infrastructure.llm.fake import FakeLLMProvider
from app.infrastructure.mcp.client import KnowledgeMCPClient
from app.infrastructure.mcp.knowledge_server import KnowledgeMCPServer
from app.infrastructure.reranking.fake import FakeReranker
from app.infrastructure.search.provider import FakeWebSearchProvider
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder
from app.search.service import ExternalSearchService
from app.verification.service import ClaimVerificationService


def _create_retrieval_item(
    content: str,
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    knowledge_type: str = "factual",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id or uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=uuid4(),
        content=content,
        score=0.9,
        rank=1,
        source={"knowledge_type": knowledge_type, "page": 1, "section": "Article 1"},
    )


class TestKnowledgeMCPServer:
    """Test Knowledge MCP server and client boundary."""

    @pytest.mark.asyncio
    async def test_mcp_tool_definitions(self) -> None:
        mock_retriever = AsyncMock()
        server = KnowledgeMCPServer(hybrid_retriever=mock_retriever)
        tools = server.get_tool_definitions()

        assert len(tools) == 1
        assert tools[0]["name"] == "search_knowledge"
        assert "parameters" in tools[0]

    @pytest.mark.asyncio
    async def test_mcp_search_knowledge_execution(self) -> None:
        chunk_id = uuid4()
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [
            _create_retrieval_item("Civil Code Article 10: Contractual freedom.", chunk_id=chunk_id)
        ]

        server = KnowledgeMCPServer(hybrid_retriever=mock_retriever, reranker=FakeReranker())
        client = KnowledgeMCPClient(server)

        raw_resp = await client.call_tool(
            "search_knowledge",
            {"query": "contractual freedom", "knowledge_type": "factual", "top_k": 5},
        )

        assert raw_resp["total_count"] == 1
        assert raw_resp["knowledge_type"] == "factual"
        assert len(raw_resp["evidence"]) == 1
        assert raw_resp["evidence"][0]["chunk_id"] == str(chunk_id)
        assert "Contractual freedom" in raw_resp["evidence"][0]["content"]

    @pytest.mark.asyncio
    async def test_mcp_unknown_tool_raises_error(self) -> None:
        server = KnowledgeMCPServer(hybrid_retriever=AsyncMock())
        client = KnowledgeMCPClient(server)
        with pytest.raises(MCPToolExecutionError, match="Unknown MCP tool"):
            await client.call_tool("invalid_tool", {})


class TestExternalSearchAndValidation:
    """Test external web search domain filtering and source validation."""

    def test_source_domain_evaluation(self) -> None:
        provider = FakeWebSearchProvider()
        service = ExternalSearchService(provider)

        # Trusted Gov
        gov_res = WebSearchResult(
            title="Labor Act",
            url="https://labor.gov/laws/statute",
            snippet="Statute details.",
        )
        gov_src = service.evaluate_source(gov_res)
        assert gov_src.is_trusted is True
        assert gov_src.authority_category == "government"
        assert gov_src.trust_score >= 0.9

        # Blocked Social Media / Spam
        spam_res = WebSearchResult(
            title="Legal Discussion",
            url="https://reddit.com/r/law/comments/123",
            snippet="Random user post.",
        )
        spam_src = service.evaluate_source(spam_res)
        assert spam_src.is_trusted is False
        assert spam_src.authority_category == "untrusted"

    @pytest.mark.asyncio
    async def test_search_and_validate_filtering(self) -> None:
        provider = FakeWebSearchProvider(
            predefined_results=[
                WebSearchResult(
                    title="Official Gazette",
                    url="https://rooznamehrasmi.ir/laws/456",
                    snippet="Official decree text.",
                ),
                WebSearchResult(
                    title="Forum Thread",
                    url="https://quora.com/what-is-the-statute",
                    snippet="Unverified opinion.",
                ),
            ]
        )
        service = ExternalSearchService(provider)
        sources = await service.search_and_validate(query="official decree", num_results=5)

        # Quora should be filtered out
        assert len(sources) == 1
        assert sources[0].domain == "rooznamehrasmi.ir"
        assert sources[0].is_trusted is True

    def test_format_external_context_includes_disclaimer(self) -> None:
        service = ExternalSearchService(FakeWebSearchProvider())
        sources = [
            ValidatedSource(
                title="Secondary Article",
                url="https://law.cornell.edu/wex/contracts",
                domain="law.cornell.edu",
                content="Definition of consideration.",
                is_trusted=True,
                trust_score=0.95,
                authority_category="academic",
            )
        ]
        context = service.format_external_context(sources)
        assert "EXTERNAL WEB SOURCE" in context
        assert "SECONDARY/NON-AUTHORITATIVE" in context
        assert "law.cornell.edu" in context


class TestClaimVerificationService:
    """Test ClaimVerificationService claim checking and structured results."""

    def test_verification_passes_grounded_response(self) -> None:
        chunk_id = uuid4()
        doc_id = uuid4()
        evidence = [
            EvidenceItem(
                evidence_id=1,
                chunk_id=chunk_id,
                document_id=doc_id,
                document_version_id=uuid4(),
                content="Under Article 10 of the Civil Code, contracts are valid unless contrary to mandatory law.",
                score=0.9,
            )
        ]
        citations = [
            Citation(
                document_id=doc_id,
                document_version_id=uuid4(),
                chunk_id=chunk_id,
            )
        ]

        verifier = ClaimVerificationService()
        response = "Under Article 10 of the Civil Code, contracts are valid unless contrary to law."
        result = verifier.verify(
            response_text=response,
            citations=citations,
            evidence_items=evidence,
            raw_citations=[{"evidence_id": 1}],
        )

        assert result.passed is True
        assert len(result.unsupported_claims) == 0
        assert len(result.citation_errors) == 0

    def test_verification_flags_unsupported_statutory_claim(self) -> None:
        chunk_id = uuid4()
        doc_id = uuid4()
        evidence = [
            EvidenceItem(
                evidence_id=1,
                chunk_id=chunk_id,
                document_id=doc_id,
                document_version_id=uuid4(),
                content="General terms of arbitration clause.",
                score=0.8,
            )
        ]
        citations = [
            Citation(
                document_id=doc_id,
                document_version_id=uuid4(),
                chunk_id=chunk_id,
            )
        ]

        verifier = ClaimVerificationService()
        # Claims Section 9999 which does NOT appear in the evidence
        hallucinated_response = "Pursuant to Section 9999 of the Penal Code, damages are treble."
        result = verifier.verify(
            response_text=hallucinated_response,
            citations=citations,
            evidence_items=evidence,
            raw_citations=[{"evidence_id": 1}],
        )

        assert result.passed is False
        assert any("Section 9999" in claim for claim in result.unsupported_claims)
        assert result.repair_guidance is not None

    def test_verification_flags_invalid_citation_id(self) -> None:
        evidence = [
            EvidenceItem(
                evidence_id=1,
                chunk_id=uuid4(),
                document_id=uuid4(),
                document_version_id=uuid4(),
                content="Valid evidence content.",
                score=0.8,
            )
        ]
        verifier = ClaimVerificationService()
        result = verifier.verify(
            response_text="Assertion citing invalid id.",
            citations=[],
            evidence_items=evidence,
            raw_citations=[{"evidence_id": 99}],  # Evidence ID 99 does not exist
        )

        assert result.passed is False
        assert any("Evidence ID: 99" in err for err in result.citation_errors)


class TestAgentStage19Integration:
    """Test Agent routing for external search fallback and bounded verification repair."""

    @pytest.mark.asyncio
    async def test_internal_first_precedence_no_external_search_when_internal_sufficient(self) -> None:
        """When internal retrieval yields evidence, external search is NOT called."""
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [
            _create_retrieval_item("Internal authoritative rule.")
        ]
        mock_search_provider = AsyncMock()
        search_service = ExternalSearchService(mock_search_provider)

        fake_llm = FakeLLMProvider(
            responses=[
                json.dumps({
                    "answer": "Answer based on internal evidence.",
                    "is_sufficient": True,
                    "citations": [{"evidence_id": 1}],
                })
            ]
        )

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
            search_service=search_service,
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(
            AgentRequest(query="What is the statute?", enable_external_search=True)
        )

        assert res.is_sufficient is True
        # Web search provider must NOT be called because internal evidence was sufficient
        mock_search_provider.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_external_search_fallback_when_internal_insufficient(self) -> None:
        """When internal retrieval is insufficient and external search is enabled, external fallback executes."""
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = []  # Internal insufficient

        web_provider = FakeWebSearchProvider(
            predefined_results=[
                WebSearchResult(
                    title="Supreme Court Precedent",
                    url="https://supremecourt.gov/precedent",
                    snippet="Precedent regarding statutory interpretation.",
                )
            ]
        )
        search_service = ExternalSearchService(web_provider)

        fake_llm = FakeLLMProvider(
            responses=[
                json.dumps({
                    "answer": "Supreme Court precedent clarifies the statute.",
                    "is_sufficient": True,
                    "citations": [],
                })
            ]
        )

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
            search_service=search_service,
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(
            AgentRequest(query="Specific novel statute question", enable_external_search=True)
        )

        assert res.is_sufficient is True
        assert len(web_provider.call_history) == 1
        assert "external_sources_count" in res.trace_metadata

    @pytest.mark.asyncio
    async def test_bounded_verification_repair_terminates_without_infinite_loop(self) -> None:
        """Verify that failed verification repairs up to max_retries and terminates cleanly."""
        mock_retriever = AsyncMock()
        mock_retriever.search.return_value = [_create_retrieval_item("Truth in Lending Act.")]

        # LLM continuously hallucinates Section 7777 on every turn
        hallucinated_answer = json.dumps({
            "answer": "Violates Section 7777 of the unverified code.",
            "is_sufficient": True,
            "citations": [{"evidence_id": 999}],
        })

        fake_llm = FakeLLMProvider(
            responses=[hallucinated_answer, hallucinated_answer, hallucinated_answer, hallucinated_answer]
        )

        compiled_graph = build_legal_agent_graph(
            hybrid_retriever=mock_retriever,
            reranker=FakeReranker(),
            llm_service=LLMService(provider=fake_llm),
            context_builder=ContextBuilder(),
            verification_service=ClaimVerificationService(),
        )
        agent = LegalAgent(compiled_graph)

        res = await agent.run(AgentRequest(query="Lending requirements", max_retries=2))

        # Must terminate at max_retries=2 without infinite loops
        assert res.retry_count == 2
