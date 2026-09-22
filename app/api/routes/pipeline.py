"""Pipeline processing API route with real-time SSE progress streaming."""

import asyncio
import json
import logging
import os
import uuid
from typing import AsyncGenerator, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.documents.exceptions import (
    DocumentTooLargeError,
    EmptyDocumentError,
    InvalidDocumentMetadataError,
    UnsupportedDocumentTypeError,
)
from app.domain.documents.service import SUPPORTED_MIME_TYPES, VALID_KNOWLEDGE_TYPES
from app.infrastructure.db.models.document import DocumentChunk
from app.infrastructure.db.repositories.document import SQLAlchemyDocumentRepository
from app.infrastructure.db.session import SessionLocal
from app.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.infrastructure.storage import MinioStorage
from app.ingestion.chunking.chunker import StructureAwareChunker
from app.ingestion.chunking.models import ChunkingConfig
from app.ingestion.normalization.normalizer import DocumentNormalizer
from app.ingestion.parsers.registry import ParserRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["pipeline"])


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def sse_event(event_type: str, data: dict) -> str:
    """Format an SSE message."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


@router.post("/process")
async def process_document_pipeline(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    document_type: str = Form("law"),
    knowledge_type: str = Form("factual"),
    source: str = Form("upload"),
    version: str = Form("1.0"),
):
    """Process a document end-to-end through the ingestion pipeline with SSE streaming progress.

    Steps:
    1. Validation & Upload
    2. Parsing (PDF/DOCX/TXT/HTML) & Text Extraction check
    3. Normalization (Structural blocks extraction)
    4. Structure-aware Chunking
    5. Persistence & Completion
    """
    settings = get_settings()

    async def event_generator() -> AsyncGenerator[str, None]:
        current_stage = "upload"
        db: Optional[Session] = None
        doc_version_record = None

        try:
            # Yield initial upload started event
            yield sse_event(
                "progress",
                {
                    "stage": "upload",
                    "status": "started",
                    "progress": 10,
                    "message": "در حال دریافت و بررسی فایل...",
                    "filename": file.filename,
                },
            )
            await asyncio.sleep(0.05)

            # 1. Read & Validate content
            content = await file.read()
            filename = file.filename or "document"
            mime_type = file.content_type or "application/octet-stream"

            # Fallback mime detection by extension if octet-stream
            ext = os.path.splitext(filename)[1].lower()
            if mime_type == "application/octet-stream":
                for m_type, extensions in SUPPORTED_MIME_TYPES.items():
                    if ext in extensions:
                        mime_type = m_type
                        break

            # Validation checks
            if mime_type not in SUPPORTED_MIME_TYPES:
                raise UnsupportedDocumentTypeError(
                    f"نوع فایل '{mime_type}' پشتیبانی نمی‌شود. فرمت‌های مجاز: PDF, DOCX, TXT, HTML"
                )

            if ext not in SUPPORTED_MIME_TYPES[mime_type]:
                raise InvalidDocumentMetadataError(
                    f"پسوند فایل '{ext}' با نوع فایل '{mime_type}' مطابقت ندارد."
                )

            if len(content) > settings.max_document_size_bytes:
                max_mb = settings.max_document_size_bytes / (1024 * 1024)
                raise DocumentTooLargeError(
                    f"حجم فایل بیش از سقف مجاز ({max_mb:.1f} مگابایت) است."
                )

            if len(content) == 0:
                raise EmptyDocumentError("فایل بارگذاری شده خالی است.")

            doc_title = title.strip() if title and title.strip() else os.path.splitext(filename)[0]

            if knowledge_type not in VALID_KNOWLEDGE_TYPES:
                raise InvalidDocumentMetadataError(
                    f"نوع دانش '{knowledge_type}' نامعتبر است. مقادیر مجاز: factual, stylistic"
                )

            # Generate IDs
            doc_id = uuid.uuid4()
            version_id = uuid.uuid4()

            # Attempt DB & Storage persistence
            saved_to_storage = False
            safe_name = os.path.basename(filename).replace("..", "")
            storage_key = f"documents/{doc_id}/{version}/{safe_name}"

            # 1. MinIO / Object Storage (optional if configured)
            if settings.minio_endpoint and settings.minio_endpoint.strip():
                try:
                    storage = MinioStorage()
                    storage.initialize()
                    storage.put_object(key=storage_key, data=content, content_type=mime_type)
                    saved_to_storage = True
                except Exception as storage_exc:
                    logger.warning("Object storage upload skipped or failed: %s", storage_exc)

            # 2. Database Record Persistence
            try:
                import hashlib
                db = SessionLocal()
                repo = SQLAlchemyDocumentRepository(db)
                checksum = hashlib.sha256(content).hexdigest()
                doc_record, doc_version_record = repo.create_with_version(
                    document_id=doc_id,
                    title=doc_title,
                    document_type=document_type,
                    knowledge_type=knowledge_type,
                    source=source,
                    version=version,
                    storage_key=storage_key,
                    checksum=checksum,
                )
                repo.commit()
            except Exception as db_exc:
                logger.warning("Database record creation failed: %s", db_exc)
                if db:
                    db.rollback()

            yield sse_event(
                "progress",
                {
                    "stage": "upload",
                    "status": "completed",
                    "progress": 25,
                    "message": "بارگذاری و ذخیره‌سازی اولیه با موفقیت انجام شد.",
                    "document_id": str(doc_id),
                    "file_size_bytes": len(content),
                    "file_size_formatted": f"{len(content) / 1024:.1f} KB",
                    "mime_type": mime_type,
                    "title": doc_title,
                    "knowledge_type": knowledge_type,
                    "saved_to_db": saved_to_storage,
                },
            )
            await asyncio.sleep(0.05)

            # 2. Parsing stage
            current_stage = "parsing"
            yield sse_event(
                "progress",
                {
                    "stage": "parsing",
                    "status": "started",
                    "progress": 35,
                    "message": "در حال استخراج متن و ساختار از سند...",
                },
            )
            await asyncio.sleep(0.05)

            parser_registry = ParserRegistry()
            parser = parser_registry.get_parser(mime_type)
            parsed_doc = parser.parse(
                content=content,
                document_id=str(doc_id),
                document_version_id=str(version_id),
            )

            # Check whether text was extracted
            is_pdf = (mime_type == "application/pdf")
            total_chars = sum(len(p.text) for p in parsed_doc.pages)
            non_empty_pages = [p for p in parsed_doc.pages if p.text and p.text.strip()]
            text_extracted = len(non_empty_pages) > 0 and total_chars > 0

            # Sample extracted text
            all_text_sample = ""
            for p in parsed_doc.pages:
                if p.text and p.text.strip():
                    all_text_sample += p.text.strip() + "\n\n"
                    if len(all_text_sample) >= 800:
                        break
            text_snippet = all_text_sample[:800].strip()

            yield sse_event(
                "progress",
                {
                    "stage": "parsing",
                    "status": "completed",
                    "progress": 55,
                    "message": "استخراج متن به پایان رسید.",
                    "is_pdf": is_pdf,
                    "text_extracted": text_extracted,
                    "pages_count": len(parsed_doc.pages),
                    "tables_count": len(parsed_doc.tables),
                    "total_chars": total_chars,
                    "sample_text": text_snippet,
                },
            )
            await asyncio.sleep(0.05)

            # 3. Normalization stage
            current_stage = "normalization"
            yield sse_event(
                "progress",
                {
                    "stage": "normalization",
                    "status": "started",
                    "progress": 65,
                    "message": "در حال نرمال‌سازی بلوک‌های متنی و شناسایی سلسله‌مراتب...",
                },
            )
            await asyncio.sleep(0.05)

            normalizer = DocumentNormalizer()
            canonical_doc = normalizer.normalize(parsed_doc)

            block_counts = {}
            headings_list = []
            for block in canonical_doc.blocks:
                b_type = getattr(block, "type", "block")
                block_counts[b_type] = block_counts.get(b_type, 0) + 1
                if b_type == "heading" and hasattr(block, "text") and block.text:
                    headings_list.append(block.text)

            yield sse_event(
                "progress",
                {
                    "stage": "normalization",
                    "status": "completed",
                    "progress": 75,
                    "message": "نرمال‌سازی سند و استخراج ساختار سازمانی کامل شد.",
                    "blocks_count": len(canonical_doc.blocks),
                    "block_types": block_counts,
                    "headings": headings_list[:12],
                },
            )
            await asyncio.sleep(0.05)

            # 4. Chunking stage
            current_stage = "chunking"
            yield sse_event(
                "progress",
                {
                    "stage": "chunking",
                    "status": "started",
                    "progress": 80,
                    "message": "در حال قطعه‌بندی هوشمند با رعایت سرفصل‌ها و مرزهای قانونی...",
                },
            )
            await asyncio.sleep(0.05)

            chunker = StructureAwareChunker(config=ChunkingConfig())
            chunks = chunker.chunk(canonical_doc)

            parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
            child_chunks = [c for c in chunks if c.chunk_type == "child"]

            # Save chunks to DB first so they are never lost
            embeddings_map = {}
            saved_chunks_count = 0
            total_to_embed = len(chunks)

            if db and doc_version_record:
                try:
                    # 1. Insert parent chunks first to satisfy foreign key constraints
                    valid_parent_ids = set()
                    for chk in parent_chunks:
                        sec_str = chk.section[:500] if chk.section else None
                        db_chunk = DocumentChunk(
                            id=chk.id,
                            document_version_id=doc_version_record.id,
                            parent_chunk_id=None,
                            content=chk.content,
                            page=chk.page,
                            section=sec_str,
                            chunk_index=chk.chunk_index,
                            chunk_metadata=chk.metadata,
                            embedding=None,
                        )
                        db.add(db_chunk)
                        valid_parent_ids.add(chk.id)
                    db.commit()

                    # 2. Insert child chunks with verified parent_chunk_id
                    for chk in child_chunks:
                        sec_str = chk.section[:500] if chk.section else None
                        p_id = chk.parent_chunk_id if chk.parent_chunk_id in valid_parent_ids else None
                        db_chunk = DocumentChunk(
                            id=chk.id,
                            document_version_id=doc_version_record.id,
                            parent_chunk_id=p_id,
                            content=chk.content,
                            page=chk.page,
                            section=sec_str,
                            chunk_index=chk.chunk_index,
                            chunk_metadata=chk.metadata,
                            embedding=None,
                        )
                        db.add(db_chunk)
                    db.commit()
                    saved_chunks_count = len(chunks)
                    logger.info("Successfully persisted %d chunks to database", saved_chunks_count)
                except Exception as chunk_db_err:
                    logger.warning("Could not persist chunks to DB: %s", chunk_db_err)
                    db.rollback()

            # 3. Generate embeddings in batches and commit incrementally
            if settings.embedding_api_key and settings.embedding_api_key != "test-key" and total_to_embed > 0:
                try:
                    yield sse_event(
                        "progress",
                        {
                            "stage": "embedding",
                            "status": "started",
                            "progress": 82,
                            "total_to_embed": total_to_embed,
                            "embedded_so_far": 0,
                            "percentage": 0.0,
                            "message": f"آماده‌سازی برای تولید بردار معنایی (Embedding) با هوش مصنوعی برای مجموعاً {total_to_embed} قطعه...",
                        },
                    )
                    await asyncio.sleep(0.05)

                    embed_provider = OpenAIEmbeddingProvider(
                        endpoint=settings.embedding_endpoint,
                        api_key=settings.embedding_api_key,
                        model=settings.embedding_model,
                        expected_dimension=settings.embedding_dimension,
                        batch_size=settings.embedding_batch_size,
                    )

                    # Batch size 40 avoids 429 quota limits on Google AI Studio free tier
                    batch_size = max(1, min(settings.embedding_batch_size, 40))
                    for batch_start in range(0, total_to_embed, batch_size):
                        batch_chunks = chunks[batch_start:batch_start + batch_size]
                        chunk_texts = [chk.content for chk in batch_chunks]
                        batch_embeddings = await embed_provider.embed_texts(chunk_texts)

                        for chk, emb in zip(batch_chunks, batch_embeddings):
                            embeddings_map[chk.id] = emb

                        if db and saved_chunks_count > 0:
                            for chk, emb in zip(batch_chunks, batch_embeddings):
                                db.query(DocumentChunk).filter(DocumentChunk.id == chk.id).update(
                                    {"embedding": emb}
                                )
                            db.commit()

                        embedded_so_far = len(embeddings_map)
                        pct = round((embedded_so_far / total_to_embed) * 100, 1)
                        progress_val = 82 + int((embedded_so_far / total_to_embed) * 12)

                        yield sse_event(
                            "progress",
                            {
                                "stage": "embedding",
                                "status": "in_progress",
                                "progress": progress_val,
                                "total_to_embed": total_to_embed,
                                "embedded_so_far": embedded_so_far,
                                "percentage": pct,
                                "message": f"تولید بردار معنایی: {embedded_so_far} از {total_to_embed} قطعه بردارسازی شد ({pct}٪)...",
                            },
                        )
                        await asyncio.sleep(0.2)

                    logger.info("Successfully generated embeddings for %d chunks", len(embeddings_map))
                    final_pct = round((len(embeddings_map) / total_to_embed) * 100, 1)

                    yield sse_event(
                        "progress",
                        {
                            "stage": "embedding",
                            "status": "completed",
                            "progress": 94,
                            "total_to_embed": total_to_embed,
                            "embedded_so_far": len(embeddings_map),
                            "percentage": final_pct,
                            "message": f"بردارهای معنایی با موفقیت تولید و در پایگاه داده ایندکس شدند ({len(embeddings_map)} از {total_to_embed} قطعه - {final_pct}٪).",
                            "embedded_count": len(embeddings_map),
                        },
                    )
                    await asyncio.sleep(0.05)
                except Exception as emb_err:
                    logger.warning("Embedding generation during ingestion failed or skipped: %s", emb_err)
                    err_pct = round((len(embeddings_map) / total_to_embed) * 100, 1) if total_to_embed > 0 else 0.0
                    yield sse_event(
                        "progress",
                        {
                            "stage": "embedding",
                            "status": "warning",
                            "progress": 94,
                            "total_to_embed": total_to_embed,
                            "embedded_so_far": len(embeddings_map),
                            "percentage": err_pct,
                            "message": f"تولید بردار معنایی با خطا مواجه شد ({len(embeddings_map)} از {total_to_embed} قطعه ذخیره شد - {err_pct}٪): {emb_err}",
                            "error": str(emb_err),
                        },
                    )

            # Format sample chunks for display
            sample_chunks_data = []
            for chk in chunks[:8]:
                sample_chunks_data.append(
                    {
                        "id": str(chk.id),
                        "chunk_index": chk.chunk_index,
                        "chunk_type": chk.chunk_type,
                        "page": chk.page,
                        "section": chk.section,
                        "section_path": chk.section_path,
                        "char_count": len(chk.content),
                        "has_embedding": chk.id in embeddings_map,
                        "content_preview": chk.content[:400] + ("..." if len(chk.content) > 400 else ""),
                        "full_content": chk.content,
                    }
                )

            yield sse_event(
                "progress",
                {
                    "stage": "chunking",
                    "status": "completed",
                    "progress": 95,
                    "message": f"قطعه‌بندی با موفقیت انجام شد: {len(chunks)} قطعه تولید گردید.",
                    "total_chunks": len(chunks),
                    "parent_chunks_count": len(parent_chunks),
                    "child_chunks_count": len(child_chunks),
                    "saved_to_db": saved_chunks_count > 0,
                    "sample_chunks": sample_chunks_data,
                    "embeddings_count": len(embeddings_map),
                },
            )
            await asyncio.sleep(0.05)

            # 5. Complete stage
            yield sse_event(
                "complete",
                {
                    "stage": "complete",
                    "status": "success",
                    "progress": 100,
                    "message": "کلیه مراحل پردازش و آماده‌سازی پایگاه دانش با موفقیت به پایان رسید.",
                    "document_id": str(doc_id),
                    "title": doc_title,
                    "summary": {
                        "pages": len(parsed_doc.pages),
                        "total_chars": total_chars,
                        "text_extracted": text_extracted,
                        "blocks": len(canonical_doc.blocks),
                        "total_chunks": len(chunks),
                        "embedded_chunks": len(embeddings_map),
                        "coverage_pct": round((len(embeddings_map) / len(chunks) * 100), 1) if chunks else 0.0,
                        "parent_chunks": len(parent_chunks),
                        "child_chunks": len(child_chunks),
                    },
                },
            )

        except Exception as exc:
            logger.error("Pipeline failure in stage %s: %s", current_stage, exc, exc_info=True)
            yield sse_event(
                "error",
                {
                    "stage": current_stage,
                    "status": "error",
                    "error": str(exc),
                    "message": f"خطا در مرحله {current_stage}: {exc}",
                },
            )
        finally:
            await file.close()
            if db:
                db.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
