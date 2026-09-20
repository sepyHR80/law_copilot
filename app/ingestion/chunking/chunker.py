"""Structure-aware chunker implementation.

Converts CanonicalDocument into semantically coherent, structure-preserving
retrieval chunks and hierarchical parent chunks per LAW_COPILOT_AGENT_MASTER_SPEC.md.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.ingestion.chunking.exceptions import ChunkingError
from app.ingestion.chunking.models import Chunk, ChunkingConfig
from app.ingestion.normalization.models import (
    CanonicalBlock,
    CanonicalDocument,
    CanonicalHeading,
    CanonicalList,
    CanonicalParagraph,
    CanonicalTable,
)

LAW_COPILOT_NAMESPACE = UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _to_uuid(val: Any) -> UUID:
    """Convert value to UUID, using deterministic UUID5 if not a valid hex string."""
    if isinstance(val, UUID):
        return val
    try:
        return UUID(str(val))
    except ValueError:
        return uuid.uuid5(LAW_COPILOT_NAMESPACE, str(val))


class StructureAwareChunker:
    """Structure-aware document chunker.

    Transforms a CanonicalDocument into a list of Chunk objects:
    - Maintains heading hierarchy and section_path.
    - Creates hierarchical parent chunks for major sections (level 1-2).
    - Binds headings to following content blocks.
    - Preserves tables intact, or partitions rows with repeated headers.
    - Performs sentence-aware splitting for oversized paragraphs.
    - Applies sentence-aware overlap between consecutive child chunks within the same section.
    - Preserves rich provenance metadata.
    - Generates 100% deterministic chunk IDs and ordering.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()

    def chunk(self, document: CanonicalDocument) -> List[Chunk]:
        """Chunk a CanonicalDocument into a list of Chunk objects.

        Args:
            document: CanonicalDocument from normalization stage.

        Returns:
            List of Chunk objects (parents and children).

        Raises:
            ChunkingError: If chunking fails.
        """
        try:
            return self._chunk_document(document)
        except Exception as exc:
            if isinstance(exc, ChunkingError):
                raise
            raise ChunkingError(f"Failed to chunk document: {exc}") from exc

    def _chunk_document(self, document: CanonicalDocument) -> List[Chunk]:
        if not document.blocks:
            return []

        doc_id = _to_uuid(document.document_id)
        doc_ver_id = _to_uuid(document.document_version_id)

        # 1. First pass: identify section hierarchy and structural subtrees for parent chunks
        parent_sections, parent_id_map = self._build_parent_sections(document.blocks, doc_id, doc_ver_id)

        # 2. Second pass: build child retrieval chunks
        raw_child_chunks = self._build_child_chunks(
            document.blocks, parent_sections, parent_id_map, doc_id, doc_ver_id
        )

        # 3. Combine parent chunks and child chunks in structural document order
        all_chunks: List[Chunk] = []

        if self.config.enable_parent_chunks:
            parent_by_id = {p.id: p for p in parent_sections}
            emitted_parents = set()

            for child in raw_child_chunks:
                if child.parent_chunk_id and child.parent_chunk_id not in emitted_parents:
                    ancestor_chain = []
                    curr_pid = child.parent_chunk_id
                    while curr_pid and curr_pid in parent_by_id and curr_pid not in emitted_parents:
                        ancestor_chain.append(parent_by_id[curr_pid])
                        curr_pid = parent_by_id[curr_pid].parent_chunk_id
                    for p in reversed(ancestor_chain):
                        if p.id not in emitted_parents:
                            all_chunks.append(p)
                            emitted_parents.add(p.id)

                all_chunks.append(child)

            for p in parent_sections:
                if p.id not in emitted_parents:
                    all_chunks.append(p)
                    emitted_parents.add(p.id)
        else:
            all_chunks = raw_child_chunks

        # 4. Final pass: assign sequential deterministic chunk_index and deterministic UUIDs
        final_chunks: List[Chunk] = []
        old_to_final_id: Dict[UUID, UUID] = {}

        # First map final IDs for all chunks
        for idx, ch in enumerate(all_chunks):
            det_id = self._generate_deterministic_id(
                doc_version_id=doc_ver_id,
                chunk_type=ch.chunk_type,
                section_path=ch.section_path,
                chunk_index=idx,
            )
            old_to_final_id[ch.id] = det_id

        # Then construct final Chunk objects with updated parent_chunk_id references
        for idx, ch in enumerate(all_chunks):
            det_id = old_to_final_id[ch.id]
            updated_parent_id = old_to_final_id.get(ch.parent_chunk_id, ch.parent_chunk_id)

            final_chunks.append(
                Chunk(
                    id=det_id,
                    document_id=ch.document_id,
                    document_version_id=ch.document_version_id,
                    parent_chunk_id=updated_parent_id,
                    chunk_type=ch.chunk_type,
                    content=ch.content,
                    page=ch.page,
                    section=ch.section,
                    section_path=ch.section_path,
                    chunk_index=idx,
                    metadata=ch.metadata,
                )
            )

        return final_chunks

    def _build_parent_sections(
        self,
        blocks: List[CanonicalBlock],
        doc_id: UUID,
        doc_ver_id: UUID,
    ) -> Tuple[List[Chunk], Dict[UUID, UUID]]:
        """Create parent chunks for major sections (heading levels 1-2).

        Returns:
            Tuple of (parent_chunks, parent_id_map)
        """
        if not self.config.enable_parent_chunks:
            return [], {}

        parent_chunks: List[Chunk] = []
        parent_id_map: Dict[UUID, UUID] = {}
        heading_indices: List[Tuple[int, CanonicalHeading]] = []

        for idx, block in enumerate(blocks):
            if isinstance(block, CanonicalHeading) and block.level in (1, 2):
                heading_indices.append((idx, block))

        for i, (start_idx, heading) in enumerate(heading_indices):
            # Find end of subtree: next heading with level <= this heading's level
            end_idx = len(blocks)
            for next_idx, next_heading in heading_indices[i + 1 :]:
                if next_heading.level <= heading.level:
                    end_idx = next_idx
                    break

            subtree_blocks = blocks[start_idx:end_idx]
            content = "\n\n".join(b.text for b in subtree_blocks if b.text.strip())
            source_pages = sorted(list({b.page_number for b in subtree_blocks}))
            source_blocks = [b.order for b in subtree_blocks]
            block_types = sorted(list({b.type for b in subtree_blocks}))

            sec_path = self._get_section_path_at(blocks, start_idx)

            parent_of_parent_id: Optional[UUID] = None
            if heading.level == 2:
                for prev_parent in reversed(parent_chunks):
                    if prev_parent.metadata.get("heading_level") == 1:
                        parent_of_parent_id = prev_parent.id
                        break

            provisional_id = self._generate_deterministic_id(
                doc_version_id=doc_ver_id,
                chunk_type="parent",
                section_path=sec_path,
                chunk_index=len(parent_chunks),
            )

            parent_chunk = Chunk(
                id=provisional_id,
                document_id=doc_id,
                document_version_id=doc_ver_id,
                parent_chunk_id=parent_of_parent_id,
                chunk_type="parent",
                content=content,
                page=source_pages[0] if source_pages else None,
                section=" > ".join(sec_path) if sec_path else heading.text,
                section_path=sec_path,
                chunk_index=0,
                metadata={
                    "retrieval_unit": False,
                    "is_parent": True,
                    "heading_level": heading.level,
                    "source_blocks": source_blocks,
                    "source_pages": source_pages,
                    "section_path": sec_path,
                    "block_types": block_types,
                    "char_count": len(content),
                },
            )
            parent_chunks.append(parent_chunk)
            parent_id_map[provisional_id] = provisional_id

        return parent_chunks, parent_id_map

    def _get_section_path_at(self, blocks: List[CanonicalBlock], target_idx: int) -> List[str]:
        """Compute the active heading stack path up to target_idx."""
        stack: List[Tuple[int, str]] = []
        for idx in range(target_idx + 1):
            block = blocks[idx]
            if isinstance(block, CanonicalHeading):
                lvl = block.level
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, block.text.strip()))
        return [item[1] for item in stack]

    def _build_child_chunks(
        self,
        blocks: List[CanonicalBlock],
        parent_sections: List[Chunk],
        parent_id_map: Dict[UUID, UUID],
        doc_id: UUID,
        doc_ver_id: UUID,
    ) -> List[Chunk]:
        """Build child retrieval chunks from canonical blocks."""
        child_chunks: List[Chunk] = []

        heading_stack: List[Tuple[int, str]] = []
        pending_heading_texts: List[str] = []
        pending_heading_blocks: List[CanonicalBlock] = []

        active_parent_id: Optional[UUID] = None

        accumulated_texts: List[str] = []
        accumulated_blocks: List[CanonicalBlock] = []

        def flush_accumulator() -> None:
            nonlocal accumulated_texts, accumulated_blocks
            if not accumulated_texts:
                return

            content = "\n\n".join(accumulated_texts).strip()
            if not content:
                accumulated_texts = []
                accumulated_blocks = []
                return

            sec_path = [item[1] for item in heading_stack]
            sec_name = " > ".join(sec_path) if sec_path else None
            pages = sorted(list({b.page_number for b in accumulated_blocks}))
            block_orders = [b.order for b in accumulated_blocks]
            types = sorted(list({b.type for b in accumulated_blocks}))

            ch = Chunk(
                document_id=doc_id,
                document_version_id=doc_ver_id,
                parent_chunk_id=active_parent_id,
                chunk_type="child",
                content=content,
                page=pages[0] if pages else None,
                section=sec_name,
                section_path=sec_path,
                chunk_index=0,
                metadata={
                    "retrieval_unit": True,
                    "is_parent": False,
                    "source_blocks": block_orders,
                    "source_pages": pages,
                    "section_path": sec_path,
                    "block_types": types,
                    "char_count": len(content),
                    "is_table": False,
                    "has_overlap": False,
                },
            )
            child_chunks.append(ch)

            accumulated_texts = []
            accumulated_blocks = []

        for block in blocks:
            if isinstance(block, CanonicalHeading):
                flush_accumulator()

                lvl = block.level
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, block.text.strip()))

                active_parent_id = self._find_nearest_parent(heading_stack, parent_sections)

                pending_heading_texts.append(block.text.strip())
                pending_heading_blocks.append(block)

            elif isinstance(block, CanonicalTable):
                flush_accumulator()

                table_chunks = self._chunk_table(
                    table=block,
                    heading_stack=heading_stack,
                    pending_headings=pending_heading_texts,
                    pending_blocks=pending_heading_blocks,
                    parent_id=active_parent_id,
                    doc_id=doc_id,
                    doc_ver_id=doc_ver_id,
                )
                child_chunks.extend(table_chunks)
                pending_heading_texts = []
                pending_heading_blocks = []

            elif isinstance(block, (CanonicalParagraph, CanonicalList, CanonicalBlock)):
                block_text = block.text.strip()
                if not block_text:
                    continue

                heading_prefix = ""
                prefix_blocks: List[CanonicalBlock] = []
                if pending_heading_texts:
                    heading_prefix = "\n\n".join(pending_heading_texts)
                    prefix_blocks = list(pending_heading_blocks)
                    pending_heading_texts = []
                    pending_heading_blocks = []

                if heading_prefix:
                    full_text = f"{heading_prefix}\n\n{block_text}"
                    combined_blocks = prefix_blocks + [block]
                else:
                    full_text = block_text
                    combined_blocks = [block]

                if len(full_text) > self.config.max_chars:
                    flush_accumulator()

                    sentence_chunks = self._split_long_paragraph(
                        full_text,
                        combined_blocks,
                        heading_stack,
                        active_parent_id,
                        doc_id,
                        doc_ver_id,
                    )
                    child_chunks.extend(sentence_chunks)
                else:
                    current_len = sum(len(t) for t in accumulated_texts) + (
                        2 * len(accumulated_texts) if accumulated_texts else 0
                    )
                    if current_len + 2 + len(full_text) > self.config.max_chars and accumulated_texts:
                        flush_accumulator()

                    accumulated_texts.append(full_text)
                    accumulated_blocks.extend(combined_blocks)

        flush_accumulator()

        if pending_heading_texts:
            sec_path = [item[1] for item in heading_stack]
            content = "\n\n".join(pending_heading_texts)
            pages = sorted(list({b.page_number for b in pending_heading_blocks}))
            ch = Chunk(
                document_id=doc_id,
                document_version_id=doc_ver_id,
                parent_chunk_id=active_parent_id,
                chunk_type="child",
                content=content,
                page=pages[0] if pages else None,
                section=" > ".join(sec_path) if sec_path else None,
                section_path=sec_path,
                chunk_index=0,
                metadata={
                    "retrieval_unit": True,
                    "is_parent": False,
                    "source_blocks": [b.order for b in pending_heading_blocks],
                    "source_pages": pages,
                    "section_path": sec_path,
                    "block_types": ["heading"],
                    "char_count": len(content),
                    "is_table": False,
                    "has_overlap": False,
                },
            )
            child_chunks.append(ch)

        # Apply soft min_chars merging within same section (Rule #2)
        child_chunks = self._apply_min_chars_merging(child_chunks)

        # Apply sentence-aware overlap across consecutive child chunks in same section (Rule #1)
        child_chunks = self._apply_sentence_overlap(child_chunks)

        return child_chunks

    def _find_nearest_parent(
        self,
        heading_stack: List[Tuple[int, str]],
        parent_sections: List[Chunk],
    ) -> Optional[UUID]:
        """Find nearest applicable parent chunk ID from the heading stack (Rule #3)."""
        if not parent_sections:
            return None

        # Search deepest to highest in the heading stack for level 1 or 2
        for lvl, text in reversed(heading_stack):
            if lvl in (1, 2):
                for p in reversed(parent_sections):
                    if p.metadata.get("heading_level") == lvl:
                        # Match by section_path ending with this heading
                        if p.section_path and p.section_path[-1] == text:
                            return p.id
        return None

    def _chunk_table(
        self,
        table: CanonicalTable,
        heading_stack: List[Tuple[int, str]],
        pending_headings: List[str],
        pending_blocks: List[CanonicalBlock],
        parent_id: Optional[UUID],
        doc_id: UUID,
        doc_ver_id: UUID,
    ) -> List[Chunk]:
        """Table chunking per Spec 10.6."""
        sec_path = [item[1] for item in heading_stack]
        sec_name = " > ".join(sec_path) if sec_path else None
        page = table.page_number
        source_blocks = [b.order for b in pending_blocks] + [table.order]
        source_pages = sorted(list({b.page_number for b in pending_blocks} | {page}))

        heading_prefix = "\n\n".join(pending_headings) if pending_headings else ""

        header_line = " | ".join(table.headers) if table.headers else ""
        row_lines = [" | ".join(str(c) for c in row) for row in table.rows]

        table_body = "\n".join(([header_line] if header_line else []) + row_lines)
        full_table_text = f"{heading_prefix}\n\n{table_body}".strip() if heading_prefix else table_body

        if len(full_table_text) <= self.config.max_chars:
            return [
                Chunk(
                    document_id=doc_id,
                    document_version_id=doc_ver_id,
                    parent_chunk_id=parent_id,
                    chunk_type="child",
                    content=full_table_text,
                    page=page,
                    section=sec_name,
                    section_path=sec_path,
                    chunk_index=0,
                    metadata={
                        "retrieval_unit": True,
                        "is_parent": False,
                        "is_table": True,
                        "oversize_row": False,
                        "headers": table.headers,
                        "source_blocks": source_blocks,
                        "source_pages": source_pages,
                        "section_path": sec_path,
                        "block_types": (["heading"] if pending_headings else []) + ["table"],
                        "char_count": len(full_table_text),
                    },
                )
            ]

        chunks: List[Chunk] = []
        current_rows: List[str] = []

        def emit_partition(rows: List[str], is_oversize: bool = False) -> None:
            lines = []
            if heading_prefix and not chunks:
                lines.append(heading_prefix)
            if header_line:
                lines.append(header_line)
            lines.extend(rows)
            partition_content = "\n".join(lines).strip()

            chunks.append(
                Chunk(
                    document_id=doc_id,
                    document_version_id=doc_ver_id,
                    parent_chunk_id=parent_id,
                    chunk_type="child",
                    content=partition_content,
                    page=page,
                    section=sec_name,
                    section_path=sec_path,
                    chunk_index=0,
                    metadata={
                        "retrieval_unit": True,
                        "is_parent": False,
                        "is_table": True,
                        "oversize_row": is_oversize,
                        "headers": table.headers,
                        "source_blocks": source_blocks,
                        "source_pages": source_pages,
                        "section_path": sec_path,
                        "block_types": (["heading"] if pending_headings else []) + ["table"],
                        "char_count": len(partition_content),
                    },
                )
            )

        header_overhead = len(header_line) + 1 if header_line else 0
        prefix_overhead = len(heading_prefix) + 2 if heading_prefix and not chunks else 0

        for row in row_lines:
            row_len = len(row)

            if row_len + header_overhead > self.config.max_chars:
                if current_rows:
                    emit_partition(current_rows)
                    current_rows = []
                emit_partition([row], is_oversize=True)
                continue

            current_partition_len = (
                prefix_overhead
                + header_overhead
                + sum(len(r) + 1 for r in current_rows)
                + row_len
            )

            if current_partition_len > self.config.max_chars and current_rows:
                emit_partition(current_rows)
                current_rows = [row]
                prefix_overhead = 0
            else:
                current_rows.append(row)

        if current_rows:
            emit_partition(current_rows)

        return chunks

    def _split_long_paragraph(
        self,
        text: str,
        blocks: List[CanonicalBlock],
        heading_stack: List[Tuple[int, str]],
        parent_id: Optional[UUID],
        doc_id: UUID,
        doc_ver_id: UUID,
    ) -> List[Chunk]:
        """Sentence-aware long paragraph splitting per Spec 10.8."""
        sec_path = [item[1] for item in heading_stack]
        sec_name = " > ".join(sec_path) if sec_path else None
        pages = sorted(list({b.page_number for b in blocks}))
        source_blocks = [b.order for b in blocks]
        block_types = sorted(list({b.type for b in blocks}))

        sentences = self._split_into_sentences(text)
        chunks: List[Chunk] = []

        current_sentences: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if sent_len > self.config.max_chars:
                if current_sentences:
                    content = " ".join(current_sentences).strip()
                    chunks.append(
                        self._create_child_chunk(
                            content, parent_id, sec_name, sec_path, pages, source_blocks, block_types, doc_id, doc_ver_id
                        )
                    )
                    current_sentences = []
                    current_len = 0

                words = sentence.split()
                sub_words: List[str] = []
                sub_len = 0
                for w in words:
                    if sub_len + len(w) + 1 > self.config.max_chars and sub_words:
                        content = " ".join(sub_words)
                        chunks.append(
                            self._create_child_chunk(
                                content, parent_id, sec_name, sec_path, pages, source_blocks, block_types, doc_id, doc_ver_id
                            )
                        )
                        sub_words = [w]
                        sub_len = len(w)
                    else:
                        sub_words.append(w)
                        sub_len += len(w) + 1
                if sub_words:
                    current_sentences = [" ".join(sub_words)]
                    current_len = len(current_sentences[0])
                continue

            if current_len + (1 if current_sentences else 0) + sent_len > self.config.max_chars and current_sentences:
                content = " ".join(current_sentences).strip()
                chunks.append(
                    self._create_child_chunk(
                        content, parent_id, sec_name, sec_path, pages, source_blocks, block_types, doc_id, doc_ver_id
                    )
                )
                current_sentences = [sentence]
                current_len = sent_len
            else:
                current_sentences.append(sentence)
                current_len += (1 if len(current_sentences) > 1 else 0) + sent_len

        if current_sentences:
            content = " ".join(current_sentences).strip()
            chunks.append(
                self._create_child_chunk(
                    content, parent_id, sec_name, sec_path, pages, source_blocks, block_types, doc_id, doc_ver_id
                )
            )

        return chunks

    def _create_child_chunk(
        self,
        content: str,
        parent_id: Optional[UUID],
        sec_name: Optional[str],
        sec_path: List[str],
        pages: List[int],
        source_blocks: List[int],
        block_types: List[str],
        doc_id: UUID,
        doc_ver_id: UUID,
    ) -> Chunk:
        return Chunk(
            document_id=doc_id,
            document_version_id=doc_ver_id,
            parent_chunk_id=parent_id,
            chunk_type="child",
            content=content,
            page=pages[0] if pages else None,
            section=sec_name,
            section_path=sec_path,
            chunk_index=0,
            metadata={
                "retrieval_unit": True,
                "is_parent": False,
                "source_blocks": source_blocks,
                "source_pages": pages,
                "section_path": sec_path,
                "block_types": block_types,
                "char_count": len(content),
                "is_table": False,
                "has_overlap": False,
            },
        )

    def _apply_min_chars_merging(self, chunks: List[Chunk]) -> List[Chunk]:
        """Soft merge short chunks (< min_chars) into preceding chunk within same section (Rule #2)."""
        if len(chunks) <= 1:
            return chunks

        merged: List[Chunk] = []
        for ch in chunks:
            if not merged:
                merged.append(ch)
                continue

            prev = merged[-1]
            can_merge = (
                prev.section_path == ch.section_path
                and not prev.metadata.get("is_table", False)
                and not ch.metadata.get("is_table", False)
                and prev.chunk_type == "child"
                and ch.chunk_type == "child"
                and (len(ch.content) < self.config.min_chars or len(prev.content) < self.config.min_chars)
                and (len(prev.content) + 2 + len(ch.content) <= self.config.max_chars)
            )

            if can_merge:
                combined_content = f"{prev.content}\n\n{ch.content}"
                merged[-1] = Chunk(
                    document_id=prev.document_id,
                    document_version_id=prev.document_version_id,
                    parent_chunk_id=prev.parent_chunk_id or ch.parent_chunk_id,
                    chunk_type="child",
                    content=combined_content,
                    page=prev.page or ch.page,
                    section=prev.section,
                    section_path=prev.section_path,
                    chunk_index=0,
                    metadata={
                        **prev.metadata,
                        "source_blocks": sorted(
                            list(set(prev.metadata.get("source_blocks", []) + ch.metadata.get("source_blocks", [])))
                        ),
                        "source_pages": sorted(
                            list(set(prev.metadata.get("source_pages", []) + ch.metadata.get("source_pages", [])))
                        ),
                        "block_types": sorted(
                            list(set(prev.metadata.get("block_types", []) + ch.metadata.get("block_types", [])))
                        ),
                        "char_count": len(combined_content),
                    },
                )
            else:
                merged.append(ch)

        return merged

    def _apply_sentence_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """Apply sentence-aware overlap between consecutive child retrieval chunks in the same section (Rule #1)."""
        if self.config.overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        result: List[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            curr = chunks[i]
            prev = chunks[i - 1]

            # Only overlap between consecutive child chunks within the exact same section
            # and neither is a table (Rule #1)
            can_overlap = (
                curr.section_path == prev.section_path
                and not curr.metadata.get("is_table", False)
                and not prev.metadata.get("is_table", False)
                and curr.chunk_type == "child"
                and prev.chunk_type == "child"
            )

            if can_overlap:
                snippet = self._extract_sentence_overlap(prev.content, self.config.overlap_chars)
                if snippet and not curr.content.startswith(snippet):
                    new_content = f"{snippet}\n\n{curr.content}"
                    updated_curr = Chunk(
                        document_id=curr.document_id,
                        document_version_id=curr.document_version_id,
                        parent_chunk_id=curr.parent_chunk_id,
                        chunk_type="child",
                        content=new_content,
                        page=curr.page,
                        section=curr.section,
                        section_path=curr.section_path,
                        chunk_index=0,
                        metadata={
                            **curr.metadata,
                            "char_count": len(new_content),
                            "has_overlap": True,
                        },
                    )
                    result.append(updated_curr)
                    continue

            result.append(curr)

        return result

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences preserving sentence-ending punctuation."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        sentences = [p.strip() for p in parts if p.strip()]
        return sentences if sentences else [text]

    @staticmethod
    def _extract_sentence_overlap(text: str, target_chars: int) -> str:
        """Extract trailing complete sentence(s) from text approximating target_chars (Rule #1)."""
        sentences = StructureAwareChunker._split_into_sentences(text)
        if not sentences:
            return ""

        collected: List[str] = []
        curr_len = 0

        for sent in reversed(sentences):
            if curr_len + len(sent) <= target_chars * 1.5 or not collected:
                collected.insert(0, sent)
                curr_len += len(sent) + 1
                if curr_len >= target_chars:
                    break
            else:
                break

        return " ".join(collected).strip()

    @staticmethod
    def _generate_deterministic_id(
        doc_version_id: UUID,
        chunk_type: str,
        section_path: List[str],
        chunk_index: int,
    ) -> UUID:
        """Generate a 100% deterministic UUID5 for a chunk (Rule #4)."""
        key = f"{doc_version_id}:{chunk_type}:{'>'.join(section_path)}:{chunk_index}"
        return uuid.uuid5(LAW_COPILOT_NAMESPACE, key)
