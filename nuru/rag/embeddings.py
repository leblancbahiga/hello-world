"""Embedders pour la recherche vectorielle.

Deux implémentations, chargées à la demande (jamais au boot) :

- `OllamaEmbedder` : embeddings sémantiques via Ollama (recommandé).
- `HashEmbedder`  : plongement local déterministe, zéro dépendance.
                   Baseline lexicale pour le hors-ligne total / le développement.

L'embedder par défaut (`rag.embedder`) est configurable dans les paramètres.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import httpx

from nuru.rag.vectors import l2_normalize

logger = logging.getLogger("nuru.rag.embed")

STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "à", "au",
    "aux", "en", "sur", "dans", "pour", "par", "que", "qui", "qu", "ne", "pas",
    "plus", "est", "sont", "ce", "cet", "cette", "ces", "se", "sa", "son",
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "with",
}

_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+(?:['’-][a-zà-ÿ0-9]+)*")


class Embedder(ABC):
    name: str

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OllamaEmbedder(Embedder):
    name = "ollama"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None
        self._dim: int = 0

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(120.0, connect=5.0, read=120.0, write=30.0, pool=5.0),
            )
        return self._client

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        client = self._ensure_client()
        resp = await client.post("/api/embed", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings") or []
        if embeddings and not self._dim:
            self._dim = len(embeddings[0])
        return embeddings

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class HashEmbedder(Embedder):
    """Hashing-trick : tokens → projection déterministe, normalisée L2.

    Exécution CPU : déléguée à un thread pour ne jamais bloquer l'event loop.
    """

    name = "hash"
    DIM = 256

    def __init__(self) -> None:
        self._dim = self.DIM

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)

    @staticmethod
    def _embed_sync(texts: list[str]) -> list[list[float]]:
        return [HashEmbedder._embed_one(t) for t in texts]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [
            tok
            for tok in _WORD_RE.findall(text.lower())
            if tok not in STOPWORDS and len(tok) > 1
        ]

    @classmethod
    def _embed_one(cls, text: str) -> list[float]:
        vec = [0.0] * cls.DIM
        counts = Counter(cls._tokens(text))
        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % cls.DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        return l2_normalize(vec)


def build_embedder(settings) -> Embedder:
    """Construit l'embedder selon `rag.embedder`. Lazy : appelé au 1er usage."""
    kind = str(settings.get("rag.embedder", "hash"))
    if kind == "ollama":
        base = str(settings.get("providers.ollama.base_url", "http://localhost:11434"))
        model = str(settings.get("rag.embedder_model", "nomic-embed-text"))
        logger.info("embedder: ollama (%s)", model)
        return OllamaEmbedder(base, model)
    logger.info("embedder: hash (hors-ligne)")
    return HashEmbedder()
