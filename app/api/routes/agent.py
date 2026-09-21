"""Agent interaction API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.agent.graph import LegalAgent, build_legal_agent_graph
from app.agent.models import AgentRequest, AgentResponse
from app.core.config import get_settings
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.reranking import get_default_reranker
from app.llm.service import LLMService
from app.rag.context_builder import ContextBuilder
from app.retrieval.hybrid import HybridSearchService

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_legal_agent(db: Session = Depends(get_db)) -> LegalAgent:
    """Constructs LegalAgent with injected services."""
    settings = get_settings()

    embedding_provider = OpenAIEmbeddingProvider(
        endpoint=settings.embedding_endpoint,
        api_key=settings.embedding_api_key,
        model=settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
        timeout=float(settings.embedding_timeout),
        batch_size=settings.embedding_batch_size,
    )

    hybrid_retriever = HybridSearchService(
        session=db,
        embedding_provider=embedding_provider,
    )

    reranker = get_default_reranker()

    llm_service = LLMService()
    context_builder = ContextBuilder()

    compiled_graph = build_legal_agent_graph(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_service=llm_service,
        context_builder=context_builder,
    )

    return LegalAgent(compiled_graph)


@router.post("/chat", response_model=AgentResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(
    request: AgentRequest,
    agent: LegalAgent = Depends(get_legal_agent),
) -> AgentResponse:
    """Invoke the LangGraph legal agent state machine."""
    try:
        return await agent.run(request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {exc}",
        ) from exc
