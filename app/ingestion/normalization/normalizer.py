"""Document normalizer: converts ParsedDocument to CanonicalDocument."""

import re
from typing import List, Optional

from app.ingestion.parsers.models import ParsedDocument, ParsedPage, ParsedTable
from app.ingestion.normalization.models import (
    CanonicalDocument,
    CanonicalBlock,
    CanonicalHeading,
    CanonicalParagraph,
    CanonicalList,
    CanonicalTable,
)
from app.ingestion.normalization.exceptions import DocumentNormalizationError


class DocumentNormalizer:
    """Normalizes a ParsedDocument into a CanonicalDocument.

    This normalizer is pure: it has no knowledge of storage, database,
    API, or any other infrastructure. It operates purely on ParsedDocument
    data and produces a CanonicalDocument.
    """

    def normalize(self, parsed_document: ParsedDocument) -> CanonicalDocument:
        """Convert a ParsedDocument to a CanonicalDocument.

        Args:
            parsed_document: The parsed document from Stage 06.

        Returns:
            CanonicalDocument with normalized blocks.

        Raises:
            DocumentNormalizationError: If normalization fails.
        """
        try:
            blocks: List[CanonicalBlock] = []
            order = 0

            # Process pages
            for page in parsed_document.pages:
                page_blocks = self._normalize_page(page)
                for block in page_blocks:
                    block.order = order
                    order += 1
                    blocks.append(block)

            # Process tables (append after page content)
            for table in parsed_document.tables:
                table_block = self._normalize_table(table)
                table_block.order = order
                order += 1
                blocks.append(table_block)

            return CanonicalDocument(
                document_id=parsed_document.document_id,
                document_version_id=parsed_document.document_version_id,
                title=parsed_document.title,
                mime_type=parsed_document.mime_type,
                blocks=blocks,
                metadata={
                    **parsed_document.metadata,
                    "normalized": True,
                },
            )
        except Exception as exc:
            if isinstance(exc, DocumentNormalizationError):
                raise
            raise DocumentNormalizationError(f"Normalization failed: {exc}") from exc

    def _normalize_page(self, page: ParsedPage) -> List[CanonicalBlock]:
        """Normalize a single page into canonical blocks."""
        blocks = []
        text = self._normalize_text(page.text)

        if not text.strip():
            return blocks

        # Split into lines and identify structure
        lines = text.split("\n")
        current_section = page.metadata.get("section") if page.metadata else None

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            # Check if this is a table block from parser text (e.g. HTML parser)
            if line == "[table]":
                table_rows = []
                j = i + 1
                while j < len(lines):
                    table_line = lines[j].strip()
                    if not table_line or table_line.startswith("["):
                        break
                    row_cells = [c.strip() for c in table_line.split("|")]
                    table_rows.append(row_cells)
                    j += 1
                if table_rows:
                    parsed_tbl = ParsedTable(
                        rows=table_rows,
                        metadata={"page": page.page_number, "section": current_section} if current_section else {"page": page.page_number},
                    )
                    blocks.append(self._normalize_table(parsed_tbl))
                i = j
                continue

            # Check if this is a heading
            heading_level = self._detect_heading_level(line, page.metadata)
            if heading_level is not None:
                blocks.append(CanonicalHeading(
                    text=self._clean_heading_text(line),
                    level=heading_level,
                    page_number=page.page_number,
                    metadata={"section": current_section} if current_section else {},
                ))
                i += 1
                continue

            # Check if this is a list
            list_items, is_ordered, items_consumed = self._parse_list(lines, i)
            if list_items:
                blocks.append(CanonicalList(
                    items=list_items,
                    ordered=is_ordered,
                    text="\n".join(f"{idx+1}. {item}" if is_ordered else f"- {item}" for idx, item in enumerate(list_items)),
                    page_number=page.page_number,
                    metadata={"section": current_section} if current_section else {},
                ))
                i += items_consumed
                continue

            # Otherwise, it's a paragraph - collect consecutive non-empty lines
            para_lines = [line]
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    break
                if next_line == "[table]":
                    break
                if self._detect_heading_level(next_line, page.metadata) is not None:
                    break
                if self._is_list_item(next_line):
                    break
                para_lines.append(next_line)
                j += 1

            para_text = "\n".join(para_lines)
            blocks.append(CanonicalParagraph(
                text=para_text,
                page_number=page.page_number,
                metadata={"section": current_section} if current_section else {},
            ))
            i = j

        return blocks

    def _normalize_table(self, table: ParsedTable) -> CanonicalTable:
        """Normalize a parsed table into a CanonicalTable."""
        rows = table.rows
        headers = []
        data_rows = rows

        if rows:
            if table.metadata and "headers" in table.metadata:
                headers = table.metadata["headers"]
                data_rows = rows
            elif len(rows) > 1 and self._is_header_row(rows[0]):
                headers = rows[0]
                data_rows = rows[1:]
            else:
                headers = []
                data_rows = rows

        # Build textual representation for search/provenance
        text_lines = []
        if headers:
            text_lines.append(" | ".join(headers))
        for r in data_rows:
            text_lines.append(" | ".join(str(c) for c in r))
        table_text = "\n".join(text_lines)

        return CanonicalTable(
            text=table_text,
            headers=headers,
            rows=data_rows,
            page_number=table.metadata.get("page", 1) if table.metadata else 1,
            metadata=table.metadata or {},
        )

    @staticmethod
    def _is_header_row(row: List[str]) -> bool:
        """Check if the first row is likely a header row."""
        if not row:
            return False
        return any("header" in str(cell).lower() for cell in row)

    @staticmethod
    def _clean_heading_text(line: str) -> str:
        """Strip tag or markdown prefix from heading text while preserving content."""
        # Strip [h1] or [Heading 1]
        line = re.sub(r"^\[(?:h\d+|heading\s*\d?)\]\s*", "", line, flags=re.IGNORECASE)
        # Strip markdown hashes
        line = re.sub(r"^#+\s*", "", line)
        return line.strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Apply safe text normalization.

        - Normalize line endings
        - Remove leading/trailing whitespace
        - Normalize repeated whitespace (but preserve single newlines)
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Normalize repeated whitespace within lines (but preserve newlines)
        lines = text.split("\n")
        normalized_lines = []
        for line in lines:
            # Collapse multiple spaces/tabs to single space
            line = re.sub(r"[ \t]+", " ", line)
            normalized_lines.append(line)

        text = "\n".join(normalized_lines)

        # Trim leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def _detect_heading_level(line: str, page_metadata: Optional[dict]) -> Optional[int]:
        """Detect if a line is a heading and return its level.

        Returns None if the line is not a heading.
        """
        if not line:
            return None

        # Check explicit heading level in metadata
        if page_metadata and "heading_level" in page_metadata:
            return page_metadata["heading_level"]

        # Tag-style headings from HTML/DOCX parsers, e.g. [h1], [h2], [Heading 1], [Heading]
        tag_match = re.match(r"^\[(?:h(\d+)|heading\s*(\d+)?)\]", line, re.IGNORECASE)
        if tag_match:
            lvl = tag_match.group(1) or tag_match.group(2)
            return int(lvl) if lvl else 1

        # Detect markdown-style headings
        if line.startswith("#"):
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break
            if 1 <= level <= 6:
                return level

        # Detect common heading patterns
        heading_patterns = [
            (r"^(Chapter|CHAPTER)\s+\d+", 1),
            (r"^(Section|SECTION)\s+\d+", 2),
            (r"^(Article|ARTICLE)\s+\d+", 2),
            (r"^\d+\.\d+\s+[A-Z]", 3),
            (r"^Heading\s+(\d+)", None),
        ]

        for pattern, level in heading_patterns:
            m = re.match(pattern, line)
            if m:
                if level is None:
                    return int(m.group(1))
                return level

        # Match any heading keyword with optional number or fallback to 1
        heading_keyword_match = re.search(r"\bheading\s*(\d+)?\b", line, re.IGNORECASE)
        if heading_keyword_match:
            lvl = heading_keyword_match.group(1)
            return int(lvl) if lvl else 1

        return None

    @staticmethod
    def _is_list_item(line: str) -> bool:
        """Check if a line is a list item."""
        # Ordered list: "1. ", "2. ", etc.
        if re.match(r"^\d+\.\s+", line):
            return True
        # Unordered list: "- ", "* ", "+ "
        if re.match(r"^[-*+]\s+", line):
            return True
        return False

    def _parse_list(self, lines: List[str], start_idx: int) -> tuple[List[str], bool, int]:
        """Parse a list starting at the given index.

        Returns:
            Tuple of (items, is_ordered, lines_consumed)
        """
        items = []
        is_ordered = False
        i = start_idx
        consumed = 0

        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                consumed += 1
                continue

            # Check for ordered list
            ordered_match = re.match(r"^(\d+)\.\s+(.+)", line)
            if ordered_match:
                if not is_ordered and items:
                    # Switching from unordered to ordered - stop
                    break
                is_ordered = True
                items.append(ordered_match.group(2))
                i += 1
                consumed += 1
                continue

            # Check for unordered list
            unordered_match = re.match(r"^[-*+]\s+(.+)", line)
            if unordered_match:
                if is_ordered and items:
                    # Switching from ordered to unordered - stop
                    break
                items.append(unordered_match.group(1))
                i += 1
                consumed += 1
                continue

            # Not a list item
            break

        return items, is_ordered, consumed
