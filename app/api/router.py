"""API router aggregation for Law Copilot."""

from fastapi import APIRouter

from app.api.routes.agent import router as agent_router
from app.api.routes.documents import router as documents_router
from app.api.routes.rag import router as rag_router

api_router = APIRouter()
api_router.include_router(agent_router)
api_router.include_router(documents_router)
api_router.include_router(rag_router)


