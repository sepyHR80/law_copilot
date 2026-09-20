"""Unit tests for document normalization."""

import pytest

from app.ingestion.parsers.models import ParsedDocument, ParsedPage, ParsedTable
from app.ingestion.normalization.models import (
    CanonicalDocument,
    CanonicalBlock,
    CanonicalHeading,
    CanonicalParagraph,
    CanonicalList,
    CanonicalTable,
)
from app.ingestion.normalization.normalizer import DocumentNormalizer
from app.ingestion.normalization.service import DocumentNormalizationService
from app.ingestion.normalization.exceptions import DocumentNormalizationError


def _make_parsed_document():
    """Create a sample ParsedDocument for testing."""
    return ParsedDocument(
        document_id="doc-123",
        document_version_id="ver-456",
        title="Test Legal Document",
        mime_type="application/pdf",
        pages=[
            ParsedPage(
                page_number=1,
                text="  Chapter 1: General Provisions  \n\nArticle 1: This is the first article.\n\nArticle 2: This is the second article.",
                metadata={"source": "pdf"},
            ),
            ParsedPage(
                page_number=2,
                text="  Chapter 2: Specific Rules  \n\nSection 2.1: Details here.\n\n- Item 1\n- Item 2\n- Item 3",
                metadata={"source": "pdf"},
            ),
        ],
        tables=[
            ParsedTable(
                rows=[
                    ["Header A", "Header B"],
                    ["Value 1", "Value 2"],
                ],
                metadata={"page": 2},
            )
        ],
        metadata={"author": "Test Author", "year": "2024"},
    )


class TestBasicNormalization:
    def test_parsed_document_converts_to_canonical(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert isinstance(result, CanonicalDocument)

    def test_document_id_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert result.document_id == "doc-123"

    def test_document_version_id_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert result.document_version_id == "ver-456"

    def test_title_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert result.title == "Test Legal Document"

    def test_mime_type_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert result.mime_type == "application/pdf"

    def test_metadata_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert result.metadata["author"] == "Test Author"
        assert result.metadata["year"] == "2024"


class TestTextNormalization:
    def test_leading_trailing_whitespace_removed(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, text="  Hello world  ")],
        )
        result = normalizer.normalize(parsed)

        assert result.blocks[0].text == "Hello world"

    def test_line_endings_normalized(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, text="Line 1\r\nLine 2\r\nLine 3")],
        )
        result = normalizer.normalize(parsed)

        assert "\r\n" not in result.blocks[0].text
        assert "Line 1\nLine 2\nLine 3" == result.blocks[0].text

    def test_repeated_whitespace_normalized(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, text="Word1    Word2     Word3")],
        )
        result = normalizer.normalize(parsed)

        assert "Word1 Word2 Word3" == result.blocks[0].text

    def test_legal_text_not_rewritten(self):
        normalizer = DocumentNormalizer()
        legal_text = "Article 12:    The party shall comply with all applicable laws and regulations."
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[ParsedPage(page_number=1, text=legal_text)],
        )
        result = normalizer.normalize(parsed)

        # Legal wording must remain intact (only whitespace normalized)
        assert "Article 12:" in result.blocks[0].text
        assert "The party shall comply" in result.blocks[0].text
        assert "applicable laws and regulations" in result.blocks[0].text


class TestOrderPreservation:
    def test_block_order_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Heading 1\n\nParagraph 1\n\nParagraph 2",
                ),
                ParsedPage(
                    page_number=2,
                    text="Heading 2\n\nTable content",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        # Should have blocks in order: heading, para, para, heading, table
        assert len(result.blocks) >= 4
        types = [b.type for b in result.blocks]
        assert types[0] == "heading"
        assert types[1] == "paragraph"
        assert types[2] == "paragraph"
        assert types[3] == "heading"

    def test_deterministic_order(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()

        result1 = normalizer.normalize(parsed)
        result2 = normalizer.normalize(parsed)

        assert [b.order for b in result1.blocks] == [b.order for b in result2.blocks]


class TestHeadings:
    def test_heading_text_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Chapter 1: General Provisions",
                    metadata={"heading_level": 1},
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        heading_blocks = [b for b in result.blocks if b.type == "heading"]
        assert len(heading_blocks) > 0
        assert heading_blocks[0].text == "Chapter 1: General Provisions"

    def test_heading_level_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Section 1.1",
                    metadata={"heading_level": 2},
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        heading_blocks = [b for b in result.blocks if b.type == "heading"]
        assert len(heading_blocks) > 0
        assert heading_blocks[0].level == 2

    def test_missing_heading_level_no_crash(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Some heading without level",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        heading_blocks = [b for b in result.blocks if b.type == "heading"]
        assert len(heading_blocks) > 0
        # Should use conservative fallback
        assert heading_blocks[0].level is not None


class TestParagraphs:
    def test_paragraph_text_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="This is a complete paragraph with legal content.",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        para_blocks = [b for b in result.blocks if b.type == "paragraph"]
        assert len(para_blocks) > 0
        assert para_blocks[0].text == "This is a complete paragraph with legal content."

    def test_empty_paragraphs_handled(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/plain",
            pages=[
                ParsedPage(page_number=1, text=""),
                ParsedPage(page_number=1, text="Real content"),
            ],
        )
        result = normalizer.normalize(parsed)

        # Empty paragraphs should be filtered or handled consistently
        para_blocks = [b for b in result.blocks if b.type == "paragraph"]
        assert all(b.text.strip() != "" for b in para_blocks)


class TestLists:
    def test_ordered_list_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/html",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="1. First item\n2. Second item\n3. Third item",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        list_blocks = [b for b in result.blocks if b.type == "list"]
        assert len(list_blocks) > 0
        assert list_blocks[0].ordered is True
        assert len(list_blocks[0].items) == 3
        assert list_blocks[0].items[0] == "First item"

    def test_unordered_list_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/html",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="- Item A\n- Item B\n- Item C",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        list_blocks = [b for b in result.blocks if b.type == "list"]
        assert len(list_blocks) > 0
        assert list_blocks[0].ordered is False
        assert len(list_blocks[0].items) == 3

    def test_list_item_order_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="text/html",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="1. Alpha\n2. Beta\n3. Gamma",
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        list_blocks = [b for b in result.blocks if b.type == "list"]
        assert list_blocks[0].items == ["Alpha", "Beta", "Gamma"]


class TestTables:
    def test_table_structure_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[],
            tables=[
                ParsedTable(
                    rows=[
                        ["Name", "Value"],
                        ["Key1", "Val1"],
                        ["Key2", "Val2"],
                    ],
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert len(table_blocks) > 0
        assert isinstance(table_blocks[0], CanonicalTable)

    def test_table_headers_separate(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[],
            tables=[
                ParsedTable(
                    rows=[
                        ["Header A", "Header B"],
                        ["Value 1", "Value 2"],
                    ],
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert table_blocks[0].headers == ["Header A", "Header B"]
        assert table_blocks[0].rows == [["Value 1", "Value 2"]]

    def test_table_rows_preserve_cells(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[],
            tables=[
                ParsedTable(
                    rows=[
                        ["A", "B", "C"],
                        ["1", "2", "3"],
                    ],
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert table_blocks[0].rows[0] == ["A", "B", "C"]
        assert table_blocks[0].rows[1] == ["1", "2", "3"]

    def test_empty_table_headers_safe(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[],
            tables=[
                ParsedTable(
                    rows=[["Val1", "Val2"]],
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        table_blocks = [b for b in result.blocks if b.type == "table"]
        assert table_blocks[0].headers == []


class TestProvenance:
    def test_page_number_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(page_number=3, text="Content on page 3"),
            ],
        )
        result = normalizer.normalize(parsed)

        assert result.blocks[0].page_number == 3

    def test_order_is_deterministic(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        orders = [b.order for b in result.blocks]
        assert orders == sorted(orders)
        assert len(set(orders)) == len(orders)  # unique

    def test_section_metadata_preserved(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(
                    page_number=1,
                    text="Content",
                    metadata={"section": "Chapter 1"},
                ),
            ],
        )
        result = normalizer.normalize(parsed)

        assert result.blocks[0].metadata.get("section") == "Chapter 1"


class TestMultiPageDocument:
    def test_blocks_span_multiple_pages(self):
        normalizer = DocumentNormalizer()
        parsed = ParsedDocument(
            document_id="doc-1",
            document_version_id="ver-1",
            mime_type="application/pdf",
            pages=[
                ParsedPage(page_number=1, text="Page 1 content"),
                ParsedPage(page_number=2, text="Page 2 content"),
                ParsedPage(page_number=3, text="Page 3 content"),
            ],
        )
        result = normalizer.normalize(parsed)

        page_numbers = [b.page_number for b in result.blocks]
        assert 1 in page_numbers
        assert 2 in page_numbers
        assert 3 in page_numbers


class TestNoChunking:
    def test_normalization_does_not_create_chunks(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        # Should have blocks, not chunks
        assert hasattr(result, "blocks")
        assert not hasattr(result, "chunks")

    def test_canonical_document_contains_blocks(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        result = normalizer.normalize(parsed)

        assert len(result.blocks) > 0
        for block in result.blocks:
            assert isinstance(block, CanonicalBlock)


class TestNormalizationService:
    def test_service_normalize_delegates_to_normalizer(self):
        service = DocumentNormalizationService()
        parsed = _make_parsed_document()
        result = service.normalize(parsed)

        assert isinstance(result, CanonicalDocument)
        assert result.document_id == "doc-123"
        assert len(result.blocks) > 0

    def test_service_accepts_custom_normalizer(self):
        custom_normalizer = DocumentNormalizer()
        service = DocumentNormalizationService(normalizer=custom_normalizer)
        parsed = _make_parsed_document()
        result = service.normalize(parsed)

        assert isinstance(result, CanonicalDocument)
        assert result.metadata.get("normalized") is True


class TestParserSpecificNormalization:
    def test_html_tag_headings_and_tables(self):
        normalizer = DocumentNormalizer()
        html_page_text = (
            "[h1] Legal Agreement\n\n"
            "This is introductory text.\n\n"
            "[h2] Article 1: Definitions\n\n"
            "This section defines the core terms used throughout the agreement.\n\n"
            "[table]\n"
            "Header Term | Header Definition\n"
            "Party A | The Licensor\n"
            "Party B | The Licensee"
        )
        parsed = ParsedDocument(
            document_id="doc-html-1",
            document_version_id="ver-html-1",
            mime_type="text/html",
            pages=[ParsedPage(page_number=1, text=html_page_text)],
        )
        result = normalizer.normalize(parsed)

        assert result.blocks[0].type == "heading"
        assert result.blocks[0].text == "Legal Agreement"
        assert result.blocks[0].level == 1

        assert result.blocks[1].type == "paragraph"
        assert result.blocks[1].text == "This is introductory text."

        assert result.blocks[2].type == "heading"
        assert result.blocks[2].text == "Article 1: Definitions"
        assert result.blocks[2].level == 2

        assert result.blocks[3].type == "paragraph"
        assert result.blocks[3].text == "This section defines the core terms used throughout the agreement."

        table_block = result.blocks[4]
        assert table_block.type == "table"
        assert table_block.headers == ["Header Term", "Header Definition"]
        assert len(table_block.rows) == 2
        assert table_block.rows[0] == ["Party A", "The Licensor"]

    def test_docx_tag_headings(self):
        normalizer = DocumentNormalizer()
        docx_page_text = (
            "[Heading 1] Terms of Service\n\n"
            "General paragraph content.\n\n"
            "[Heading 2] Termination Clause\n\n"
            "Details about termination."
        )
        parsed = ParsedDocument(
            document_id="doc-docx-1",
            document_version_id="ver-docx-1",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            pages=[ParsedPage(page_number=1, text=docx_page_text)],
        )
        result = normalizer.normalize(parsed)

        assert result.blocks[0].type == "heading"
        assert result.blocks[0].text == "Terms of Service"
        assert result.blocks[0].level == 1

        assert result.blocks[1].type == "paragraph"

        assert result.blocks[2].type == "heading"
        assert result.blocks[2].text == "Termination Clause"
        assert result.blocks[2].level == 2


class TestSerializationRoundTrip:
    def test_canonical_document_serialization_roundtrip(self):
        normalizer = DocumentNormalizer()
        parsed = _make_parsed_document()
        canonical = normalizer.normalize(parsed)

        data = canonical.model_dump()
        reconstructed = CanonicalDocument.model_validate(data)

        assert reconstructed.document_id == canonical.document_id
        assert len(reconstructed.blocks) == len(canonical.blocks)
        for original, restored in zip(canonical.blocks, reconstructed.blocks):
            assert original.type == restored.type
            assert original.text == restored.text
            assert original.order == restored.order
            assert type(original) is type(restored)
