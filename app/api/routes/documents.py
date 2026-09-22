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


@router.get("/stats")
def get_documents_stats(
    db: Session = Depends(get_db),
):
    """Retrieve comprehensive statistics about documents, chunks, and embeddings."""
    from app.core.config import get_settings
    from app.infrastructure.db.models.document import Document, DocumentChunk, DocumentVersion

    settings = get_settings()

    total_docs = db.query(Document).count()
    total_versions = db.query(DocumentVersion).count()
    total_chunks = db.query(DocumentChunk).count()
    embedded_chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.embedding.is_not(None))
        .count()
    )
    unembedded_chunks = total_chunks - embedded_chunks
    coverage_pct = round((embedded_chunks / total_chunks * 100), 2) if total_chunks > 0 else 0.0

    # Document details
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    doc_list = []
    for d in docs:
        d_chunks = 0
        d_embedded = 0
        for v in d.versions:
            c_cnt = db.query(DocumentChunk).filter(DocumentChunk.document_version_id == v.id).count()
            e_cnt = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.document_version_id == v.id, DocumentChunk.embedding.is_not(None))
                .count()
            )
            d_chunks += c_cnt
            d_embedded += e_cnt

        doc_list.append({
            "id": str(d.id),
            "title": d.title,
            "document_type": d.document_type,
            "knowledge_type": d.knowledge_type,
            "source": d.source,
            "total_chunks": d_chunks,
            "embedded_chunks": d_embedded,
            "coverage_pct": round((d_embedded / d_chunks * 100), 2) if d_chunks > 0 else 0.0,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        })

    return {
        "total_documents": total_docs,
        "total_versions": total_versions,
        "total_chunks": total_chunks,
        "embedded_chunks": embedded_chunks,
        "unembedded_chunks": unembedded_chunks,
        "embedding_coverage_pct": coverage_pct,
        "embedding_config": {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "endpoint": settings.embedding_endpoint,
        },
        "documents": doc_list,
    }


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
):
    """List all registered legal documents in the knowledge base."""
    from app.infrastructure.db.models.document import Document

    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "id": str(d.id),
            "title": d.title,
            "document_type": d.document_type,
            "knowledge_type": d.knowledge_type,
            "source": d.source,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        }
        for d in docs
    ]


@router.post("/seed")
async def trigger_seed_legal_corpus():
    """Trigger seeding of essential Iranian legal statutes and embeddings."""
    try:
        from scripts.seed_legal_corpus import seed_legal_corpus
        await seed_legal_corpus()
        return {
            "status": "success",
            "message": "پایگاه دانش حقوقی با موفقیت به‌روزرسانی و بردارهای معنایی ایندکس شدند.",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to seed legal corpus: {exc}")
