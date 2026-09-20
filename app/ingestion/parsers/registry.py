"""Parser registry for selecting parsers by MIME type."""

from typing import Dict, Type

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.docx import DOCXParser
from app.ingestion.parsers.html import HTMLParser
from app.ingestion.parsers.txt import TXTParser
from app.ingestion.parsers.exceptions import UnsupportedDocumentFormatError


class ParserRegistry:
    """Registry mapping MIME types to parser implementations."""

    _PARSERS: Dict[str, Type[DocumentParser]] = {
        "application/pdf": PDFParser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DOCXParser,
        "text/plain": TXTParser,
        "text/html": HTMLParser,
    }

    def get_parser(self, mime_type: str) -> DocumentParser:
        """Get a parser for the given MIME type.

        Args:
            mime_type: MIME type of the document.

        Returns:
            Parser instance.

        Raises:
            UnsupportedDocumentFormatError: If no parser exists for the MIME type.
        """
        parser_cls = self._PARSERS.get(mime_type)
        if parser_cls is None:
            raise UnsupportedDocumentFormatError(
                f"No parser available for MIME type: {mime_type}"
            )
        return parser_cls()
