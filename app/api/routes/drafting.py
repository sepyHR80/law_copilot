"""Document Drafting API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.drafting.exceptions import DraftingError, StyleLeakageError
from app.domain.drafting.models import DraftDocument, DraftingQuery
from app.domain.retrieval.exceptions import InvalidQueryError
from app.drafting.service import DocumentDraftingService
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.reranking import get_default_reranker
from app.llm.service import LLMService
from app.retrieval.hybrid import HybridSearchService

router = APIRouter(prefix="/api/v1/drafting", tags=["drafting"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_drafting_service(db: Session = Depends(get_db)) -> DocumentDraftingService:
    """Dependency provider constructing DocumentDraftingService."""
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

    return DocumentDraftingService(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        llm_service=llm_service,
    )


@router.post("/draft", response_model=DraftDocument, status_code=status.HTTP_200_OK)
async def draft_document(
    query: DraftingQuery,
    service: DocumentDraftingService = Depends(get_drafting_service),
) -> DraftDocument:
    """Draft a structured legal document with separated factual grounding and stylistic guidance."""
    try:
        return await service.draft(query)
    except (InvalidQueryError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except StyleLeakageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Style leakage detected: {exc}",
        ) from exc
    except DraftingError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drafting error: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drafting failed: {exc}",
        ) from exc
