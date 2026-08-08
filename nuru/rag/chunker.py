"""Découpage en chunks par fenêtre glissante, respectueux des paragraphes.

Les chunks ne franchissent pas une limite de page (PDF) : les citations
restent fiables. La taille est exprimée en caractères (≈ 4 chars/token).
"""

from __future__ import annotations

import re


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_blocks(
    blocks: list[dict],
    chunk_chars: int = 2048,  # ≈ 512 tokens
    overlap_chars: int = 256,  # ≈ 64 tokens
) -> list[dict]:
    chunks: list[dict] = []
    for block in blocks:
        page = block.get("page")
        buffer = ""
        for para in _split_paragraphs(block.get("text", "")):
            # paragraphe très long : découpage forcé avec chevauchement
            while len(para) > chunk_chars:
                if buffer:
                    chunks.append(_emit(buffer, page))
                    buffer = ""
                chunks.append(_emit(para[:chunk_chars], page))
                para = para[chunk_chars - overlap_chars:]
            if buffer and len(buffer) + len(para) + 1 > chunk_chars:
                chunks.append(_emit(buffer, page))
                buffer = buffer[-overlap_chars:] if overlap_chars else ""
                buffer = f"{buffer}\n{para}"
            else:
                buffer = f"{buffer}\n{para}" if buffer else para
        if buffer:
            chunks.append(_emit(buffer, page))
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
    return chunks


def _emit(text: str, page: int | None) -> dict:
    return {"content": text.strip(), "page": page, "tokens": max(1, len(text) // 4)}
