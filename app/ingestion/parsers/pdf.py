"""PDF parser using PyMuPDF."""

import fitz  # PyMuPDF
from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.models import ParsedDocument, ParsedPage
from app.ingestion.parsers.exceptions import DocumentParsingError


class PDFParser(DocumentParser):
    """Parser for PDF documents using PyMuPDF."""

    def parse(self, content: bytes, document_id: str, document_version_id: str) -> ParsedDocument:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open PDF: {exc}") from exc

        pages = []
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = page.get_text("text")
                pages.append(ParsedPage(
                    page_number=page_num + 1,
                    text=text,
                    metadata={"page_count": len(doc)},
                ))
            except Exception as exc:
                raise DocumentParsingError(
                    f"Failed to extract text from page {page_num + 1}: {exc}"
                ) from exc

        doc.close()

        return ParsedDocument(
            document_id=document_id,
            document_version_id=document_version_id,
            mime_type="application/pdf",
            pages=pages,
            metadata={"page_count": len(pages)},
        )
