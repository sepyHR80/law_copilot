"""RAG Question Answering API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.rag.models import RAGQuery, RAGResponse
from app.domain.retrieval.exceptions import InvalidQueryError
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.reranking import get_default_reranker
from app.llm.service import LLMService
from app.rag.service import RAGService
from app.retrieval.hybrid import HybridSearchService

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    """Dependency provider constructing RAGService with database session."""
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

    return RAGService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_service=llm_service,
    )


@router.post("/query", response_model=RAGResponse, status_code=status.HTTP_200_OK)
async def query_rag(
    query: RAGQuery,
    service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Execute grounded question answering against ingested legal documents."""
    try:
        return await service.answer(query)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution failed: {exc}",
        ) from exc
