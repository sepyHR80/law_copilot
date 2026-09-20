"""Unit tests for Stage 08 — Structure-Aware Chunking.

Tests all 14 required categories from LAW_COPILOT_AGENT_MASTER_SPEC.md:
1. Heading/content binding
2. Heading hierarchy
3. Section path
4. Table preservation (small table intact)
5. Large table row partitioning
6. Repeated headers on partition
7. Oversize row behavior
8. Paragraph grouping
9. Long paragraph sentence splitting
10. Parent/child relationship
11. Provenance metadata
12. Multipage documents
13. Deterministic chunk ordering and IDs
14. Service integration
"""

import uuid
import pytest

from app.ingestion.chunking.chunker import StructureAwareChunker
from app.ingestion.chunking.exceptions import ChunkingError
from app.ingestion.chunking.models import Chunk, ChunkingConfig
from app.ingestion.chunking.service import DocumentChunkingService
from app.ingestion.normalization.models import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalHeading,
    CanonicalList,
    CanonicalParagraph,
    CanonicalTable,
)


def _make_doc(blocks=None, doc_id=None, ver_id=None):
    """Helper to create a CanonicalDocument for testing."""
    return CanonicalDocument(
        document_id=doc_id or str(uuid.uuid4()),
        document_version_id=ver_id or str(uuid.uuid4()),
        title="Test Legal Agreement",
        mime_type="application/pdf",
        blocks=blocks or [],
    )


class TestHeadingContentBinding:
    def test_heading_content_binding(self):
        """Spec 10.7: Heading is prepended to first following content block, not left stranded."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Article 1: Definitions", level=2, page_number=1, order=0),
                CanonicalParagraph(
                    text="In this Agreement, the following terms shall have the meanings specified below.",
                    page_number=1,
                    order=1,
                ),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=False))
        chunks = chunker.chunk(doc)

        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        assert len(child_chunks) == 1
        assert child_chunks[0].content.startswith("Article 1: Definitions")
        assert "In this Agreement" in child_chunks[0].content
        assert 0 in child_chunks[0].metadata["source_blocks"]
        assert 1 in child_chunks[0].metadata["source_blocks"]


class TestHeadingHierarchyAndSectionPath:
    def test_heading_hierarchy(self):
        """Spec 10.5: Heading stack pops higher/equal levels and pushes new headings."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Chapter 1", level=1, order=0),
                CanonicalHeading(text="Section 1.1", level=2, order=1),
                CanonicalParagraph(text="Content 1.1", order=2),
                CanonicalHeading(text="Section 1.2", level=2, order=3),
                CanonicalParagraph(text="Content 1.2", order=4),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=False))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        assert len(chunks) == 2
        assert chunks[0].section_path == ["Chapter 1", "Section 1.1"]
        assert chunks[1].section_path == ["Chapter 1", "Section 1.2"]

    def test_section_path(self):
        """Spec 10.5: section_path preserves full hierarchical list of headings."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Chapter 2", level=1, order=0),
                CanonicalHeading(text="General Provisions", level=2, order=1),
                CanonicalHeading(text="Article 5", level=3, order=2),
                CanonicalParagraph(text="Obligations of the parties under Article 5.", order=3),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=False))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        assert len(chunks) == 1
        assert chunks[0].section_path == ["Chapter 2", "General Provisions", "Article 5"]
        assert chunks[0].section == "Chapter 2 > General Provisions > Article 5"


class TestTableChunking:
    def test_table_preserved_intact(self):
        """Spec 10.6 Rule 1: Small table that fits in max_chars is preserved intact in one chunk."""
        table = CanonicalTable(
            headers=["Clause", "Obligation", "Penalty"],
            rows=[
                ["Clause 1", "Confidentiality", "$10,000"],
                ["Clause 2", "Non-compete", "$50,000"],
            ],
            order=0,
            page_number=1,
        )
        doc = _make_doc(blocks=[table])
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=1000))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        assert len(chunks) == 1
        assert chunks[0].metadata["is_table"] is True
        assert chunks[0].metadata["oversize_row"] is False
        assert "Clause 1 | Confidentiality | $10,000" in chunks[0].content
        assert "Clause 2 | Non-compete | $50,000" in chunks[0].content

    def test_large_table_row_partitioning(self):
        """Spec 10.6 Rule 2: Large table is partitioned by rows, never splitting an individual row."""
        rows = [[f"Key_{i}", f"Long Description of Item Number {i} with additional details", f"Value_{i}"] for i in range(15)]
        table = CanonicalTable(
            headers=["Key", "Description", "Value"],
            rows=rows,
            order=0,
            page_number=1,
        )
        doc = _make_doc(blocks=[table])
        # Force table to partition with small max_chars
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=250))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata["is_table"] is True
            # Each partition must contain full rows (no half lines)
            lines = c.content.split("\n")
            for line in lines:
                assert line.count("|") == 2

    def test_repeated_headers_on_partition(self):
        """Spec 10.6 Rule 3: Repeated column headers in each resulting table chunk."""
        rows = [[f"Item_{i}", f"Description_{i}"] for i in range(10)]
        table = CanonicalTable(
            headers=["Item", "Description"],
            rows=rows,
            order=0,
            page_number=1,
        )
        doc = _make_doc(blocks=[table])
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=150))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        assert len(chunks) > 1
        for c in chunks:
            assert c.content.startswith("Item | Description")

    def test_oversize_row_behavior(self):
        """Spec 10.6 Rule 5: If a single row exceeds max_chars, keep intact with oversize_row=true."""
        massive_text = "X" * 400
        table = CanonicalTable(
            headers=["Col1", "Col2"],
            rows=[
                ["Short", "Row"],
                ["Oversized", massive_text],
                ["Another", "Short"],
            ],
            order=0,
            page_number=1,
        )
        doc = _make_doc(blocks=[table])
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=200))
        chunks = [c for c in chunker.chunk(doc) if c.chunk_type == "child"]

        oversize_chunks = [c for c in chunks if c.metadata.get("oversize_row") is True]
        assert len(oversize_chunks) == 1
        assert massive_text in oversize_chunks[0].content


class TestParagraphGroupingAndSplitting:
    def test_paragraph_grouping(self):
        """Spec 10.1: Paragraphs are grouped under max_chars."""
        paras = [
            CanonicalParagraph(text=f"Paragraph {i}: Valid legal clause wording.", order=i, page_number=1)
            for i in range(5)
        ]
        doc = _make_doc(blocks=paras)
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=500, min_chars=50, enable_parent_chunks=False))
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        for i in range(5):
            assert f"Paragraph {i}" in chunks[0].content

    def test_long_paragraph_sentence_splitting(self):
        """Spec 10.8: Long paragraph is split at sentence boundaries."""
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Legal agreements require clear and unambiguous provisions.",
            "Failure to perform constitutes a material breach of contract.",
            "All disputes shall be resolved by binding arbitration.",
            "This clause survives termination of the agreement.",
        ]
        long_para = " ".join(sentences)
        doc = _make_doc(blocks=[CanonicalParagraph(text=long_para, order=0, page_number=1)])
        # Set max_chars so it splits into roughly 2-3 chunks
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=120, overlap_chars=0, enable_parent_chunks=False))
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Each chunk should end with a period (sentence boundary)
        for c in chunks:
            assert c.content.endswith(".")


class TestParentChildRelationship:
    def test_parent_child_relationship(self):
        """Spec 10.4 & Rule #3: Parent chunks created for level 1-2, children attach to nearest parent."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Chapter 1: Master Terms", level=1, order=0),
                CanonicalParagraph(text="Introductory text of chapter 1.", order=1),
                CanonicalHeading(text="Section 1.1: Scope", level=2, order=2),
                CanonicalParagraph(text="Scope paragraph text.", order=3),
                CanonicalHeading(text="Article 1: Details", level=3, order=4),
                CanonicalParagraph(text="Detailed paragraph text.", order=5),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=True))
        chunks = chunker.chunk(doc)

        parent_chunks = [c for c in chunks if c.chunk_type == "parent"]
        child_chunks = [c for c in chunks if c.chunk_type == "child"]

        assert len(parent_chunks) == 2  # Level 1 (Chapter 1) and Level 2 (Section 1.1)
        ch1_parent = [p for p in parent_chunks if p.metadata.get("heading_level") == 1][0]
        sec11_parent = [p for p in parent_chunks if p.metadata.get("heading_level") == 2][0]

        # Sec 1.1 parent attaches to Chapter 1 parent
        assert sec11_parent.parent_chunk_id == ch1_parent.id

        # Children under Section 1.1 and Article 1 attach to nearest parent (Section 1.1)
        sec11_children = [c for c in child_chunks if "Section 1.1" in c.section_path or "Article 1" in c.section_path]
        for c in sec11_children:
            assert c.parent_chunk_id == sec11_parent.id


class TestProvenanceAndMultipage:
    def test_provenance_metadata(self):
        """Spec 10.9: Provenance metadata preserved in every chunk."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Section A", level=1, order=0, page_number=1),
                CanonicalParagraph(text="Paragraph A1", order=1, page_number=1),
                CanonicalParagraph(text="Paragraph A2", order=2, page_number=2),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=False))
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        meta = chunks[0].metadata
        assert "source_blocks" in meta
        assert "source_pages" in meta
        assert "section_path" in meta
        assert "block_types" in meta
        assert meta["source_blocks"] == [0, 1, 2]
        assert meta["source_pages"] == [1, 2]
        assert "heading" in meta["block_types"]
        assert "paragraph" in meta["block_types"]

    def test_multipage_document(self):
        """Multipage documents correctly track page numbers."""
        doc = _make_doc(
            blocks=[
                CanonicalParagraph(text="This is significant content on page 1 of the document.", order=0, page_number=1),
                CanonicalParagraph(text="This is significant content on page 2 of the document.", order=1, page_number=2),
                CanonicalParagraph(text="This is significant content on page 3 of the document.", order=2, page_number=3),
            ]
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(max_chars=50, enable_parent_chunks=False))
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 3
        pages = [c.page for c in chunks]
        assert 1 in pages
        assert 2 in pages
        assert 3 in pages


class TestDeterminismAndOrdering:
    def test_deterministic_chunk_ordering(self):
        """User Rule #4: Identical input + config produces identical ordering, chunk_index, and UUIDs."""
        doc = _make_doc(
            doc_id="00000000-0000-0000-0000-000000000001",
            ver_id="00000000-0000-0000-0000-000000000002",
            blocks=[
                CanonicalHeading(text="Chapter 1", level=1, order=0),
                CanonicalParagraph(text="Paragraph 1 content.", order=1),
                CanonicalHeading(text="Section 1.1", level=2, order=2),
                CanonicalParagraph(text="Paragraph 2 content.", order=3),
            ],
        )
        chunker = StructureAwareChunker(config=ChunkingConfig(enable_parent_chunks=True))
        run1 = chunker.chunk(doc)
        run2 = chunker.chunk(doc)

        assert len(run1) == len(run2)
        for c1, c2 in zip(run1, run2):
            assert c1.id == c2.id
            assert c1.chunk_index == c2.chunk_index
            assert c1.chunk_type == c2.chunk_type
            assert c1.content == c2.content
            assert c1.parent_chunk_id == c2.parent_chunk_id

        # Verify chunk_index is strictly sequential from 0 to N-1
        indices = [c.chunk_index for c in run1]
        assert indices == list(range(len(run1)))


class TestSentenceOverlapAndMinChars:
    def test_sentence_aware_overlap(self):
        """User Rule #1: Sentence-aware overlap applies between consecutive child chunks in same section."""
        sentences = [
            "First sentence of the agreement.",
            "Second sentence explains the terms.",
            "Third sentence details the warranties.",
            "Fourth sentence covers dispute resolution.",
            "Fifth sentence covers governing law.",
        ]
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Article 1", level=3, order=0),
                CanonicalParagraph(text=" ".join(sentences), order=1),
            ]
        )
        chunker = StructureAwareChunker(
            config=ChunkingConfig(max_chars=90, overlap_chars=40, enable_parent_chunks=False)
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        # Second chunk should have overlap flag
        assert chunks[1].metadata.get("has_overlap") is True

    def test_min_chars_soft_merging(self):
        """User Rule #2: Soft merging for short chunks within same section."""
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Section 1", level=2, order=0),
                CanonicalParagraph(text="This is an initial paragraph.", order=1),
                CanonicalParagraph(text="Short end.", order=2),
            ]
        )
        # min_chars=50: "Short end." (10 chars) will merge into preceding paragraph
        chunker = StructureAwareChunker(
            config=ChunkingConfig(max_chars=1000, min_chars=50, enable_parent_chunks=False)
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert "Short end." in chunks[0].content


class TestServiceIntegration:
    def test_service_integration(self):
        """DocumentChunkingService correctly delegates to chunker."""
        service = DocumentChunkingService()
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Title", level=1, order=0),
                CanonicalParagraph(text="Paragraph content.", order=1),
            ]
        )
        chunks = service.chunk(doc)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_service_with_config_override(self):
        """DocumentChunkingService respects config override."""
        service = DocumentChunkingService()
        doc = _make_doc(
            blocks=[
                CanonicalHeading(text="Title", level=1, order=0),
                CanonicalParagraph(text="Paragraph content.", order=1),
            ]
        )
        chunks_no_parents = service.chunk(doc, config=ChunkingConfig(enable_parent_chunks=False))
        assert all(c.chunk_type == "child" for c in chunks_no_parents)


class TestEdgeCases:
    def test_empty_document_returns_empty_list(self):
        """Empty CanonicalDocument returns empty list of chunks."""
        service = DocumentChunkingService()
        doc = _make_doc(blocks=[])
        chunks = service.chunk(doc)
        assert chunks == []

    def test_document_with_only_paragraphs_no_headings(self):
        """Document with only paragraphs has None for section and empty section_path."""
        service = DocumentChunkingService()
        doc = _make_doc(
            blocks=[
                CanonicalParagraph(text="Paragraph 1.", order=0, page_number=1),
                CanonicalParagraph(text="Paragraph 2.", order=1, page_number=1),
            ]
        )
        chunks = service.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].section is None
        assert chunks[0].section_path == []
        assert chunks[0].parent_chunk_id is None
        assert chunks[0].chunk_type == "child"

    def test_chunking_error_on_invalid_input(self):
        """ChunkingError is raised if an invalid object causes chunking failure."""
        chunker = StructureAwareChunker()
        with pytest.raises(ChunkingError):
            chunker.chunk(None)
