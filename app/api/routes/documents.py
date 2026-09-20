"""Document upload API routes."""

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.domain.documents.exceptions import (
    DocumentStorageError,
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentMetadataError,
    UnsupportedDocumentTypeError,
)
from app.domain.documents.repository import DocumentRepositoryProtocol
from app.domain.documents.schemas import DocumentUploadResponse
from app.domain.documents.service import DocumentService
from app.infrastructure.db.repositories.document import SQLAlchemyDocumentRepository
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.storage import MinioStorage

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


def get_db() -> Session:
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_storage() -> MinioStorage:
    """Dependency to get storage service."""
    storage = MinioStorage()
    storage.initialize()
    return storage


@router.post("", response_model=DocumentUploadResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    title: str = Form(None),  # Accept None to allow service-level empty validation
    document_type: str = Form(...),
    knowledge_type: str = Form(...),
    source: str = Form(...),
    version: str = Form("1.0"),
    db: Session = Depends(get_db),
    storage: MinioStorage = Depends(get_storage),
) -> DocumentUploadResponse:
    """Upload a new legal document.

    Accepts a file upload with metadata, stores the file in MinIO,
    and creates database records.
    """
    service = DocumentService(
        storage=storage,
        repository=SQLAlchemyDocumentRepository(db),
    )

    try:
        content = await file.read()
        result = service.upload_document(
            file_content=content,
            filename=file.filename or "document",
            title=title or "",
            document_type=document_type,
            knowledge_type=knowledge_type,
            source=source,
            mime_type=file.content_type or "application/octet-stream",
            version=version,
        )
        return DocumentUploadResponse(**result)
    except UnsupportedDocumentTypeError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except EmptyDocumentError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except InvalidDocumentMetadataError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except DocumentStorageError as exc:
        raise HTTPException(status_code=500, detail="Internal server error during document upload")
    finally:
        await file.close()


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
    storage: MinioStorage = Depends(get_storage),
):
    """Delete a document, its versions, chunks, and stored files."""
    from uuid import UUID
    from app.infrastructure.db.models.document import Document

    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document UUID")

    doc = db.query(Document).filter(Document.id == doc_uuid).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    for version in doc.versions:
        if version.storage_key:
            try:
                storage.delete_object(version.storage_key)
            except Exception:
                pass

    db.delete(doc)
    db.commit()
    return None
