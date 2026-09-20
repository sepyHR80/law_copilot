"""Unit tests for document parsers."""

from pathlib import Path

import pytest

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.models import ParsedDocument, ParsedPage, ParsedTable
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.parsers.exceptions import (
    DocumentParsingError,
    UnsupportedDocumentFormatError,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "parsing"


class TestPDFParser:
    def test_extracts_page_text(self):
        from app.ingestion.parsers.pdf import PDFParser

        pdf_bytes = (FIXTURES / "sample.pdf").read_bytes()
        parser = PDFParser()
        result = parser.parse(pdf_bytes, document_id="test-doc-id", document_version_id="test-ver-id")

        assert isinstance(result, ParsedDocument)
        assert result.document_id == "test-doc-id"
        assert result.document_version_id == "test-ver-id"
        assert result.mime_type == "application/pdf"
        assert len(result.pages) > 0
        assert result.pages[0].text.strip() != ""

    def test_preserves_page_numbers(self):
        from app.ingestion.parsers.pdf import PDFParser

        pdf_bytes = (FIXTURES / "multi_page.pdf").read_bytes()
        parser = PDFParser()
        result = parser.parse(pdf_bytes, document_id="test", document_version_id="v1")

        page_numbers = [p.page_number for p in result.pages]
        assert page_numbers == [1, 2, 3, 4, 5]


    def test_preserves_empty_pages(self):
        from app.ingestion.parsers.pdf import PDFParser
 
        pdf_bytes = (FIXTURES / "empty_page.pdf").read_bytes()
        parser = PDFParser()
        result = parser.parse(pdf_bytes, document_id="test", document_version_id="v1")
 
        assert len(result.pages) == 2
        assert result.pages[0].page_number == 1
        assert result.pages[0].text == ""
        assert result.pages[1].page_number == 2
        assert result.pages[1].text.strip() != ""

    def test_invalid_pdf_raises_error(self):
        from app.ingestion.parsers.pdf import PDFParser

        invalid_bytes = (FIXTURES / "invalid.pdf").read_bytes()
        parser = PDFParser()
        with pytest.raises(DocumentParsingError):
            parser.parse(invalid_bytes, document_id="test", document_version_id="v1")


class TestTXTParser:
    def test_returns_complete_text(self):
        from app.ingestion.parsers.txt import TXTParser

        txt_bytes = (FIXTURES / "sample.txt").read_bytes()
        parser = TXTParser()
        result = parser.parse(txt_bytes, document_id="test", document_version_id="v1")

        assert isinstance(result, ParsedDocument)
        assert result.mime_type == "text/plain"
        assert len(result.pages) == 1
        assert "Article 1" in result.pages[0].text

    def test_preserves_utf8_characters(self):
        from app.ingestion.parsers.txt import TXTParser

        txt_bytes = (FIXTURES / "sample.txt").read_bytes()
        parser = TXTParser()
        result = parser.parse(txt_bytes, document_id="test", document_version_id="v1")

        assert "\u06cc\u0627" in result.pages[0].text
        assert "\u0639\u0644\u064a\u0643\u0645" in result.pages[0].text


class TestHTMLParser:
    def test_extracts_text(self):
        from app.ingestion.parsers.html import HTMLParser

        html_bytes = (FIXTURES / "sample.html").read_bytes()
        parser = HTMLParser()
        result = parser.parse(html_bytes, document_id="test", document_version_id="v1")

        assert isinstance(result, ParsedDocument)
        assert result.mime_type == "text/html"
        assert "Legal Document" in result.pages[0].text
        assert "paragraph of legal text" in result.pages[0].text

    def test_preserves_title(self):
        from app.ingestion.parsers.html import HTMLParser

        html_bytes = (FIXTURES / "sample.html").read_bytes()
        parser = HTMLParser()
        result = parser.parse(html_bytes, document_id="test", document_version_id="v1")

        assert result.title == "Test HTML Document"

    def test_removes_script_content(self):
        from app.ingestion.parsers.html import HTMLParser

        html_bytes = (FIXTURES / "sample.html").read_bytes()
        parser = HTMLParser()
        result = parser.parse(html_bytes, document_id="test", document_version_id="v1")

        assert "alert" not in result.pages[0].text
        assert "script" not in result.pages[0].text.lower()

    def test_removes_style_content(self):
        from app.ingestion.parsers.html import HTMLParser

        html_bytes = (FIXTURES / "sample.html").read_bytes()
        parser = HTMLParser()
        result = parser.parse(html_bytes, document_id="test", document_version_id="v1")

        assert ".hidden" not in result.pages[0].text
        assert "display: none" not in result.pages[0].text

    def test_preserves_headings(self):
        from app.ingestion.parsers.html import HTMLParser

        html_bytes = (FIXTURES / "sample.html").read_bytes()
        parser = HTMLParser()
        result = parser.parse(html_bytes, document_id="test", document_version_id="v1")

        full_text = result.pages[0].text
        assert "Legal Document" in full_text
        assert "Section 1" in full_text


class TestDOCXParser:
    def test_extracts_paragraphs(self):
        from app.ingestion.parsers.docx import DOCXParser

        docx_bytes = (FIXTURES / "sample.docx").read_bytes()
        parser = DOCXParser()
        result = parser.parse(docx_bytes, document_id="test", document_version_id="v1")

        assert isinstance(result, ParsedDocument)
        assert result.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert "paragraph in a DOCX" in result.pages[0].text

    def test_detects_headings(self):
        from app.ingestion.parsers.docx import DOCXParser

        docx_bytes = (FIXTURES / "sample.docx").read_bytes()
        parser = DOCXParser()
        result = parser.parse(docx_bytes, document_id="test", document_version_id="v1")

        full_text = result.pages[0].text
        assert "Document Title" in full_text
        assert "Section 1" in full_text

    def test_extracts_tables(self):
        from app.ingestion.parsers.docx import DOCXParser

        docx_bytes = (FIXTURES / "sample.docx").read_bytes()
        parser = DOCXParser()
        result = parser.parse(docx_bytes, document_id="test", document_version_id="v1")

        assert len(result.tables) > 0
        table = result.tables[0]
        assert isinstance(table, ParsedTable)
        assert len(table.rows) > 0
        assert table.rows[0][0] == "Header A"
        assert table.rows[1][1] == "Value 2"

    def test_preserves_document_order(self):
        from app.ingestion.parsers.docx import DOCXParser

        docx_bytes = (FIXTURES / "sample.docx").read_bytes()
        parser = DOCXParser()
        result = parser.parse(docx_bytes, document_id="test", document_version_id="v1")

        full_text = result.pages[0].text
        title_idx = full_text.find("Document Title")
        section_idx = full_text.find("Section 1")
        para_idx = full_text.find("paragraph in a DOCX")
        assert title_idx < section_idx < para_idx

    def test_invalid_docx_raises_error(self):
        from app.ingestion.parsers.docx import DOCXParser

        invalid_bytes = b"This is not a valid DOCX file."
        parser = DOCXParser()
        with pytest.raises(DocumentParsingError):
            parser.parse(invalid_bytes, document_id="test", document_version_id="v1")


class TestParserRegistry:
    def test_returns_pdf_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("application/pdf")
        assert isinstance(parser, DocumentParser)

    def test_returns_docx_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert isinstance(parser, DocumentParser)

    def test_returns_html_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("text/html")
        assert isinstance(parser, DocumentParser)

    def test_returns_txt_parser(self):
        registry = ParserRegistry()
        parser = registry.get_parser("text/plain")
        assert isinstance(parser, DocumentParser)

    def test_rejects_unsupported_mime(self):
        registry = ParserRegistry()
        with pytest.raises(UnsupportedDocumentFormatError):
            registry.get_parser("image/png")
