"""HTML parser using BeautifulSoup."""

from app.ingestion.parsers.base import DocumentParser
from app.ingestion.parsers.models import ParsedDocument, ParsedPage
from app.ingestion.parsers.exceptions import DocumentParsingError


class HTMLParser(DocumentParser):
    """Parser for HTML documents using BeautifulSoup."""

    def parse(self, content: bytes, document_id: str, document_version_id: str) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, "html.parser")
        except Exception as exc:
            raise DocumentParsingError(f"Failed to parse HTML: {exc}") from exc

        # Remove script and style elements
        for element in soup(["script", "style"]):
            element.decompose()

        # Extract title
        title = None
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract headings and their text
        text_parts = []
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            text_parts.append(f"[{heading.name}] {heading.get_text(strip=True)}")

        # Extract paragraph text
        for para in soup.find_all("p"):
            text = para.get_text(strip=True)
            if text:
                text_parts.append(text)

        # Extract list items
        for li in soup.find_all("li"):
            text = li.get_text(strip=True)
            if text:
                text_parts.append(f"- {text}")

        # Extract table content
        for table in soup.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            if rows:
                table_text = "\n".join(" | ".join(row) for row in rows)
                text_parts.append(f"[table]\n{table_text}")

        full_text = "\n".join(text_parts)

        return ParsedDocument(
            document_id=document_id,
            document_version_id=document_version_id,
            title=title,
            mime_type="text/html",
            pages=[ParsedPage(page_number=1, text=full_text, metadata={"type": "html"})],
            metadata={"title": title},
        )
