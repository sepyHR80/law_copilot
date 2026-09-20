"""Pydantic schemas for document upload API."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response schema for successful document upload."""

    id: UUID
    title: str
    document_type: str
    knowledge_type: str
    source: str
    version: str
    storage_key: str
    checksum: str
    file_name: str
    mime_type: str
    created_at: datetime

    model_config = {"from_attributes": True}
