"""PDF parser using PyMuPDF with specialized Persian/Arabic RTL and ligature reconstruction."""

import re
from collections import defaultdict
import fitz  # PyMuPDF

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.exceptions import DocumentParsingError
from app.ingestion.parsers.models import ParsedDocument, ParsedPage

# Common inverted ligature dictionary in broken PDF font encodings (ال -> لا)
LIGATURE_REPLACEMENTS = {
    "عالوه": "علاوه",
    "الزم": "لازم",
    "اطالعات": "اطلاعات",
    "اسالم": "اسلام",
    "انقالب": "انقلاب",
    "ابالغ": "ابلاغ",
    "اصالح": "اصلاح",
    "عالئم": "علائم",
    "خالف": "خلاف",
    "باال": "بالا",
    "اعالم": "اعلام",
    "اخالق": "اخلاق",
    "مالحظه": "ملاحظه",
    "مالقات": "ملاقات",
    "اطالع": "اطلاع",
    "ابالغات": "ابلاغات",
    "وکال": "وکلا",
    "تسهیالت": "تسهیلات",
    "اختالف": "اختلاف",
    "اصالحی": "اصلاحی",
    "اسالمی": "اسلامی",
    "عالق": "علاق",
    "کال": "کلا",
    "تکلایف": "تکالیف",
    "صالحیت": "صلاحیت",
    "صالح": "صلاح",
    "مشکالت": "مشکلات",
    "مالک": "ملاک",
    "عالیق": "علایق",
    "الزم الاجراء": "لازم‌الاجراء",
    "الزم الاجرا": "لازم‌الاجرا",
}


class PDFParser(DocumentParser):
    """Parser for PDF documents with high-fidelity Persian/Arabic text and ligature recovery."""

    def parse(self, content: bytes, document_id: str, document_version_id: str) -> ParsedDocument:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise DocumentParsingError(f"Failed to open PDF: {exc}") from exc

        pages = []
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                text = self._extract_page_text(page)
                pages.append(
                    ParsedPage(
                        page_number=page_num + 1,
                        text=text,
                        metadata={"page_count": len(doc)},
                    )
                )
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

    def _extract_page_text(self, page: fitz.Page) -> str:
        """Extract text from a PDF page with geometric RTL reconstruction for Persian/Arabic."""
        d = page.get_text("rawdict")
        chars = []
        has_persian = False

        for b in d.get("blocks", []):
            if "lines" in b:
                for l in b["lines"]:
                    for s in l["spans"]:
                        span_chars = s.get("chars", [])
                        i = 0
                        while i < len(span_chars):
                            c = span_chars[i]
                            ch = c["c"]
                            if re.search(r"[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]", ch):
                                has_persian = True
                            bbox = c["bbox"]
                            w = bbox[2] - bbox[0]

                            # Handle zero-width 'ا' + 'ل' ligature -> convert to 'لا'
                            if i + 1 < len(span_chars):
                                next_c = span_chars[i + 1]
                                next_ch = next_c["c"]
                                if ch == "ا" and next_ch == "ل" and w < 0.05:
                                    comb_bbox = (
                                        min(bbox[0], next_c["bbox"][0]),
                                        min(bbox[1], next_c["bbox"][1]),
                                        max(bbox[2], next_c["bbox"][2]),
                                        max(bbox[3], next_c["bbox"][3]),
                                    )
                                    chars.append(
                                        (comb_bbox[0], comb_bbox[1], comb_bbox[2], comb_bbox[3], "لا")
                                    )
                                    i += 2
                                    continue

                            if ch != " ":
                                chars.append((bbox[0], bbox[1], bbox[2], bbox[3], ch))
                            i += 1

        # Fallback to standard PyMuPDF text for purely Latin/English pages
        if not has_persian:
            return page.get_text("text")

        # Cluster characters into lines by Y coordinate (3.5 pt tolerance)
        lines_dict = defaultdict(list)
        for ch in chars:
            y_key = round(ch[1] / 3.5) * 3.5
            lines_dict[y_key].append(ch)

        extracted_lines = []
        for y in sorted(lines_dict.keys()):
            line_chars = lines_dict[y]
            # Primary sort: Right to Left (descending X0)
            line_chars.sort(key=lambda c: c[0], reverse=True)

            # Preserve LTR order for numbers, dates, and Latin text
            runs = []
            current_run = []
            current_is_ltr = None

            for c in line_chars:
                ch = c[4]
                is_ltr = bool(re.match(r"[0-9a-zA-Z\./\-ˏ]", ch))
                if current_is_ltr is None:
                    current_is_ltr = is_ltr
                    current_run.append(c)
                elif is_ltr == current_is_ltr and is_ltr:
                    current_run.append(c)
                else:
                    if current_is_ltr:
                        current_run.sort(key=lambda c: c[0])
                    runs.extend(current_run)
                    current_run = [c]
                    current_is_ltr = is_ltr

            if current_run:
                if current_is_ltr:
                    current_run.sort(key=lambda c: c[0])
                runs.extend(current_run)

            # Join characters and calculate word spacing based on geometric gap
            line_text = ""
            prev_c = None
            for c in runs:
                ch = c[4]
                if prev_c is not None:
                    gap = prev_c[0] - c[2]
                    if gap > 2.2:
                        line_text += " "
                line_text += ch
                prev_c = c

            cleaned = line_text.strip()

            # Normalize Persian dates: convert comma-like separators ˏ to /
            cleaned = cleaned.replace("ˏ", "/")

            # Normalize dash and spacing around Article headers: 'ماده -30' -> 'ماده 30 -'
            cleaned = re.sub(r"^(ماده)\s*[-–—]\s*(\d+)", r"\1 \2 -", cleaned)
            cleaned = re.sub(r"^(ماده\s*\d+)\s*[-–—]", r"\1 -", cleaned)

            # Normalize تبصره dashes
            cleaned = re.sub(r"^(تبصره)\s*[-–—]\s*(\d+)", r"\1 \2 -", cleaned)
            cleaned = re.sub(r"^(تبصره)\s*[-–—]", r"\1 -", cleaned)

            # Apply Persian ligature repairs
            for bad_lig, good_lig in LIGATURE_REPLACEMENTS.items():
                cleaned = cleaned.replace(bad_lig, good_lig)

            if cleaned:
                extracted_lines.append(cleaned)

        return "\n".join(extracted_lines)
