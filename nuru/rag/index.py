"""Gestionnaire RAG : ingestion incrémentale et recherche hybride.

Pipeline (voir architecture) :
  fichier → extraction texte → chunking → embedding → stockage SQLite.
Recherche : top-k vectoriel (cosine) + top-k texte (FTS5/BM25), fusion RRF.
Le gestionnaire est créé au démarrage mais n'initialise l'embedder qu'au
premier usage (aucun réseau ni modèle au boot).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING

from nuru.rag.chunker import chunk_blocks
from nuru.rag.embeddings import Embedder, build_embedder
from nuru.rag.extractors import extract_blocks
from nuru.rag.vectors import pack, top_k_cosine, unpack

if TYPE_CHECKING:
    from nuru.events import EventBus
    from nuru.settings import Settings
    from nuru.storage import Storage

logger = logging.getLogger("nuru.rag")

_RRF_K = 60


class RAGManager:
    def __init__(self, storage: "Storage", settings: "Settings", events: "EventBus") -> None:
        self.storage = storage
        self.settings = settings
        self.events = events
        self._embedder: Embedder | None = None

    # ----------------------------------------------------------------- embedder

    def configure(self) -> None:
        """Reconstruit l'embedder (appelé quand les paramètres changent)."""
        if self._embedder is not None:
            asyncio.ensure_future(self._close_embedder())
        self._embedder = None

    async def _close_embedder(self) -> None:
        embedder = self._embedder
        if embedder is not None and hasattr(embedder, "close"):
            try:
                await embedder.close()
            except Exception:  # noqa: BLE001
                pass

    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    # ------------------------------------------------------------------ ingest

    def _chunk_chars(self) -> tuple[int, int]:
        size = int(self.settings.get("rag.chunk_size", 512))
        overlap = int(self.settings.get("rag.chunk_overlap", 64))
        return size * 4, overlap * 4

    async def ingest(self, name: str, data: bytes, *,
                     mime: str | None = None,
                     path: str | None = None,
                     collection: str = "default",
                     doc_id: str | None = None) -> dict:
        file_hash = hashlib.sha256(data).hexdigest()

        existing = self.storage.find_document_by_hash(file_hash)
        if existing and existing["name"] == name:
            return existing  # déjà indexé, rien à faire (ingestion incrémentale)

        previous = self.storage.find_document_by_name(name, collection)
        if previous and (not existing or previous["id"] != existing["id"]):
            self.storage.delete_document(previous["id"])

        doc_id = doc_id or uuid.uuid4().hex[:16]
        self.storage.upsert_document({
            "id": doc_id, "collection": collection, "name": name, "path": path,
            "mime": mime, "size": len(data), "hash": file_hash, "status": "indexing",
        })
        await self.events.publish("document.indexing", {"document_id": doc_id, "name": name})

        try:
            blocks = await asyncio.to_thread(extract_blocks, name, data)
            if not blocks:
                raise RuntimeError("aucun texte extractible")
            chunk_chars, overlap_chars = self._chunk_chars()
            chunks = chunk_blocks(blocks, chunk_chars, overlap_chars)
            texts = [c["content"] for c in chunks]
            vectors = await self.embedder().embed(texts)
            if len(vectors) != len(chunks):
                raise RuntimeError("l'embedder n'a pas retourné le bon nombre de vecteurs")
        except Exception as exc:  # noqa: BLE001
            logger.warning("ingest failed for %s: %s", name, exc)
            self.storage.update_document_status(doc_id, "error", str(exc))
            await self.events.publish("document.failed", {"document_id": doc_id, "error": str(exc)})
            return self.storage.get_document(doc_id)

        rows = [
            {
                "id": uuid.uuid4().hex[:16],
                "document_id": doc_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "tokens": chunk["tokens"],
                "embedding": pack(vectors[i]),
                "page": chunk.get("page"),
                "meta": json.dumps({"collection": collection}),
            }
            for i, chunk in enumerate(chunks)
        ]
        self.storage.add_chunks(rows)
        self.storage.update_document_status(doc_id, "ready", chunks=len(rows))
        await self.events.publish("document.indexed", {
            "document_id": doc_id, "name": name, "chunks": len(rows),
        })
        logger.info("indexé: %s (%d chunks)", name, len(rows))
        return self.storage.get_document(doc_id)

    # ------------------------------------------------------------------ search

    async def search(self, query: str, top_k: int | None = None, *,
                     collection: str | None = None) -> list[dict]:
        top_k = top_k or int(self.settings.get("rag.top_k", 5))
        pool = int(self.settings.get("rag.vector_pool", top_k * 5))

        rows = self.storage.iter_chunk_vectors()
        if not rows:
            return []

        fts_results = self._fts_search(query, pool)
        fts_ids = {r["id"]: r for r in fts_results}

        try:
            qvec = (await self.embedder().embed([query]))[0]
            vectors = [unpack(r["embedding"]) for r in rows]
            vec_hits = top_k_cosine(vectors, qvec, pool)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recherche vectorielle indisponible, repli BM25: %s", exc)
            vec_hits = []

        fused: dict[str, dict] = {}
        for rank, (row_idx, _score) in enumerate(vec_hits):
            row = rows[row_idx]
            fused.setdefault(row["id"], {"row": row, "score": 0.0})
            fused[row["id"]]["score"] += 1.0 / (_RRF_K + rank + 1)
        for rank, (chunk_id, _r) in enumerate(fts_results.items()):
            fused.setdefault(chunk_id, {"row": _r, "score": 0.0})
            fused[chunk_id]["score"] += 1.0 / (_RRF_K + rank + 1)

        ordered = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]

        docs = {d["id"]: d for d in self.storage.list_documents()}
        results: list[dict] = []
        for item in ordered:
            row = item["row"]
            doc = docs.get(row["document_id"])
            if doc is None:
                continue
            if collection and doc["collection"] != collection:
                continue
            results.append({
                "chunk_id": row["id"],
                "document_id": row["document_id"],
                "document": doc["name"],
                "collection": doc["collection"],
                "page": row["page"],
                "content": row["content"],
                "score": round(item["score"], 4),
            })
        return results

    def _fts_search(self, query: str, limit: int) -> dict[str, dict]:
        import re

        tokens = re.findall(r"[a-zà-ÿ0-9]+", query.lower())
        if not tokens:
            return {}
        match = " ".join(tokens)
        rows = self.storage.fts_search(match, limit=limit)
        return {r["id"]: r for r in rows}

    # --------------------------------------------------------------- contexte

    def format_context(self, results: list[dict]) -> str:
        if not results:
            return ""
        lines = ["Contexte extrait de vos documents :"]
        for i, res in enumerate(results, start=1):
            source = res["document"]
            if res.get("page"):
                source += f" (page {res['page']})"
            lines.append(f"[{i}] — {source}")
            lines.append(res["content"])
        lines.append("Répondez en vous appuyant sur ce contexte. "
                     "Citez les sources entre crochets, comme [1].")
        return "\n\n".join(lines)
