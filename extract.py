"""
Extract plain text from PDF and DOCX files.
"""

import io
from pathlib import Path


def extract_pdf(data: bytes) -> str:
    import pymupdf
    doc = pymupdf.open(stream=data, filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def extract_docx(data: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(data))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(data)
    elif ext in (".docx", ".doc"):
        return extract_docx(data)
    elif ext == ".txt":
        return data.decode("utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
