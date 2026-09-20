"""TXT parser."""

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.models import ParsedDocument, ParsedPage


class TXTParser(DocumentParser):
    """Parser for plain text documents."""

    def parse(self, content: bytes, document_id: str, document_version_id: str) -> ParsedDocument:
        text = content.decode("utf-8")

        return ParsedDocument(
            document_id=document_id,
            document_version_id=document_version_id,
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, text=text, metadata={"type": "txt"})],
            metadata={"char_count": len(text)},
        )
