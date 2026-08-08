"""Extraction de texte depuis des fichiers.

Chaque extracteur est importé à la demande (lazy) : la plupart des fichiers
n'ont besoin que du stdlib. `pymupdf` n'est importé que pour les PDF.

Sortie : une liste de blocs `{"text": str, "page": int|None}` — les pages
sont conservées pour les citations (PDF), `None` sinon.
"""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

logger = logging.getLogger("nuru.rag.extract")

TEXT_EXTS = {".txt", ".md", ".markdown", ".log"}
SUPPORTED_EXTS = TEXT_EXTS | {".pdf", ".docx"}

NS_DOCX = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class UnsupportedFormat(Exception):
    pass


def _text_block(text: str) -> list[dict]:
    return [{"text": text, "page": None}]


def extract_blocks(name: str, data: bytes) -> list[dict]:
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXTS:
        return _text_block(data.decode("utf-8", errors="replace"))
    if ext == ".pdf":
        return _extract_pdf(data)
    if ext == ".docx":
        return _extract_docx(data)
    raise UnsupportedFormat(f"format non supporté : {ext or '(inconnu)'}")


def _extract_pdf(data: bytes) -> list[dict]:
    import fitz  # pymupdf — import différé (lourd)

    blocks: list[dict] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                blocks.append({"text": text, "page": idx})
    if not blocks:
        logger.warning("PDF sans texte extractible (peut nécessiter un OCR — plugin)")
    return blocks


def _extract_docx(data: bytes) -> list[dict]:
    with zipfile.ZipFile(__import__("io").BytesIO(data)) as zf:
        xml = zf.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    parts: list[str] = []
    for para in root.iter(f"{NS_DOCX}p"):
        text = "".join(node.text or "" for node in para.iter(f"{NS_DOCX}t"))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return _text_block("\n\n".join(parts))
