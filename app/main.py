from fastapi import FastAPI

from app.api.router import api_router
from app.api.routes.monitoring import router as monitoring_router

app = FastAPI(
    title="Law Copilot",
    version="0.1.0",
)

app.include_router(monitoring_router)
app.include_router(api_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
