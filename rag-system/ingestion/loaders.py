"Document loaders for the ingestion pipeline."

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Union

from models import Document

logger = logging.getLogger(__name__)


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF, one Document per page."""
    try:
        import pypdf
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    docs: list[Document] = []
    reader = pypdf.PdfReader(str(path))
    total = len(reader.pages)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            logger.warning(
                "PDF page %d/%d had no extractable text: %s", i + 1, total, path.name
            )
            continue
        docs.append(
            Document(
                content=text,
                source=str(path),
                page_number=i + 1,
                total_pages=total,
                metadata={
                    "page": i + 1,
                    "total_pages": total,
                    "file_name": path.name,
                    "file_type": "pdf",
                },
            )
        )

    logger.info("Loaded %d pages from PDF: %s", len(docs), path.name)
    return docs


def _load_pdf_with_tables(path: Path) -> list[Document]:
    """
    Alternative PDF loader using pdfplumber — better for tables.
    Use when your PDFs contain structured tables (financial reports, etc.).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Install pdfplumber: pip install pdfplumber")

    docs: list[Document] = []
    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            for table in page.extract_tables():
                rows = [" | ".join(str(c) for c in row if c) for row in table]
                text += "\n" + "\n".join(rows)

            text = text.strip()
            if not text:
                continue
            docs.append(
                Document(
                    content=text,
                    source=str(path),
                    page_number=i + 1,
                    total_pages=total,
                    metadata={
                        "page": i + 1,
                        "total_pages": total,
                        "file_name": path.name,
                        "file_type": "pdf",
                    },
                )
            )
    return docs


def _load_docx(path: Path) -> list[Document]:
    """Load a .docx file. Returns one Document (paragraphs joined)."""
    try:
        import docx
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = docx.Document(str(path))
    parts: list[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                parts.append(row_text)

    content = "\n\n".join(parts)
    return [
        Document(
            content=content,
            source=str(path),
            metadata={"file_name": path.name, "file_type": "docx"},
        )
    ]


def _load_html(path: Path) -> list[Document]:
    """Load an HTML file, stripping tags and boilerplate."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4 lxml")

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    title = soup.title.string if soup.title else None

    return [
        Document(
            content=text,
            source=str(path),
            title=title,
            metadata={"file_name": path.name, "file_type": "html", "title": title},
        )
    ]


def _load_text(path: Path) -> list[Document]:
    """Load plain text or Markdown."""
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="latin-1", errors="replace")
        logger.warning("Fell back to latin-1 for: %s", path.name)

    return [
        Document(
            content=content,
            source=str(path),
            metadata={"file_name": path.name, "file_type": path.suffix.lstrip(".")},
        )
    ]


def _load_json(path: Path) -> list[Document]:
    """
    Load a JSON or JSONL file.
    Expects either a list of objects with a "text"/"content" key,
    or a single object. Each item becomes one Document.
    """
    import json

    raw = path.read_text(encoding="utf-8")
    docs: list[Document] = []

    if path.suffix == ".jsonl":
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        items = [json.loads(l) for l in lines]
    else:
        data = json.loads(raw)
        items = data if isinstance(data, list) else [data]

    for i, item in enumerate(items):
        if isinstance(item, str):
            content = item
        else:
            content = (
                item.get("text") or item.get("content") or item.get("body") or str(item)
            )

        docs.append(
            Document(
                content=content,
                source=str(path),
                metadata={
                    **{k: v for k, v in item.items() if k != "text"},
                    "file_name": path.name,
                    "index": i,
                },
            )
        )
    return docs


_LOADERS: dict[str, callable] = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".doc": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
    ".txt": _load_text,
    ".md": _load_text,
    ".markdown": _load_text,
    ".rst": _load_text,
    ".json": _load_json,
    ".jsonl": _load_json,
}


def load_document(path: Union[str, Path], max_mb: int = 50) -> list[Document]:
    "Load a single file and return a list of Documents."
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"File {path.name} is {size_mb:.1f} MB, exceeds {max_mb} MB limit"
        )

    suffix = path.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        supported = ", ".join(_LOADERS.keys())
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")

    try:
        docs = loader(path)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path.name, exc, exc_info=True)
        raise

    docs = [d for d in docs if d.content.strip()]
    logger.info("Loaded %d document(s) from %s", len(docs), path.name)
    return docs


def load_documents(paths: list[Union[str, Path]], max_mb: int = 50) -> list[Document]:
    """Load multiple files; log and skip failures."""
    all_docs: list[Document] = []
    for p in paths:
        try:
            all_docs.extend(load_document(p, max_mb=max_mb))
        except Exception as exc:
            logger.error("Skipping %s: %s", p, exc)
    return all_docs


def load_from_bytes(content: bytes, filename: str, max_mb: int = 50) -> list[Document]:
    """
    Load a document from raw bytes (e.g., FastAPI UploadFile).
    Writes to a temp file then delegates to load_document.
    """
    import tempfile

    size_mb = len(content) / (1024 * 1024)
    if size_mb > max_mb:
        raise ValueError(
            f"Upload {filename} is {size_mb:.1f} MB, exceeds {max_mb} MB limit"
        )

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        docs = load_document(tmp_path, max_mb=max_mb)
        for d in docs:
            d.source = filename
            d.metadata["file_name"] = filename
        return docs
    finally:
        tmp_path.unlink(missing_ok=True)
