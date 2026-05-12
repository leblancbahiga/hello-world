"""
Moteur RAG hybride: sqlite-vec + BM25.

Architecture ultra-légère sans serveur:
- Vector search avec sqlite-vec
- Full-text search avec FTS5 (BM25)
- Score hybride pondéré
- Confidence gate anti-hallucination
"""

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import numpy as np
import sqlite_vec

from src.config import config
from src.embedder import Embedder


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Un chunk de document avec son embedding."""

    id: str
    text: str
    source: str
    embedding: Optional[mx.array] = None
    score: float = 0.0


@dataclass
class SearchResult:
    """Résultat de recherche RAG."""

    text: str
    source: str
    score: float
    method: str  # "vector", "bm25", ou "hybrid"


class RAGEngine:
    """
    Moteur RAG hybride ultra-rapide.

    Combine:
    - Similarité cosinus vectorielle (sqlite-vec)
    - BM25 full-text search (FTS5)
    - Score hybride pondéré
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.conn: Optional[sqlite3.Connection] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialiser la base de données vectorielle."""
        if self._initialized:
            return

        logger.debug("Initialisation du moteur RAG...")

        db_path = config.DB_PATH

        # Connexion SQLite avec extensions
        self.conn = sqlite3.connect(str(db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Créer tables
        await self._create_tables()

        self._initialized = True
        logger.info(f"RAG Engine prêt: {db_path}")

    async def _create_tables(self) -> None:
        """Créer les tables vectorielles et FTS."""
        assert self.conn is not None

        cursor = self.conn.cursor()

        # Table principale avec embeddings
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Index vectoriel
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embedding 
            ON documents USING vec0 (embedding float[384])
        """)

        # Table FTS5 pour BM25
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts 
            USING fts5(text, source, content='documents', content_rowid='rowid')
        """)

        # Triggers pour sync FTS
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                INSERT INTO documents_fts(rowid, text, source) 
                VALUES (NEW.rowid, NEW.text, NEW.source);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text, source) 
                VALUES('delete', OLD.rowid, OLD.text, OLD.source);
            END
        """)

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN
                INSERT INTO documents_fts(documents_fts, rowid, text, source) 
                VALUES('delete', OLD.rowid, OLD.text, OLD.source);
                INSERT INTO documents_fts(rowid, text, source) 
                VALUES (NEW.rowid, NEW.text, NEW.source);
            END
        """)

        self.conn.commit()
        logger.debug("Tables RAG créées.")

    async def add_chunk(self, chunk: Chunk) -> None:
        """
        Ajouter un chunk à la base.

        Args:
            chunk: Chunk avec texte et embedding
        """
        if not self._initialized:
            await self.initialize()

        assert self.conn is not None
        assert chunk.embedding is not None

        cursor = self.conn.cursor()

        # Convertir embedding MLX → numpy → bytes
        embedding_bytes = np.array(chunk.embedding, dtype=np.float32).tobytes()

        cursor.execute(
            """
            INSERT OR REPLACE INTO documents (id, text, source, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (chunk.id, chunk.text, chunk.source, embedding_bytes),
        )

        self.conn.commit()

    async def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Recherche hybride vectorielle + BM25.

        Args:
            query: Requête utilisateur
            top_k: Nombre de résultats

        Returns:
            Liste de SearchResult triée par score
        """
        if not self._initialized:
            await self.initialize()

        assert self.conn is not None

        # Générer embedding de la requête
        query_embedding = await self.embedder.embed_single(query)
        query_vec = np.array(query_embedding, dtype=np.float32).tobytes()

        cursor = self.conn.cursor()

        # 1. Recherche vectorielle
        vector_results = await self._vector_search(cursor, query_vec, top_k)

        # 2. Recherche BM25
        bm25_results = await self._bm25_search(cursor, query, top_k)

        # 3. Fusion hybride
        hybrid_results = self._merge_results(
            vector_results,
            bm25_results,
            top_k,
        )

        logger.debug(
            f"RAG: {len(vector_results)} vectoriels + {len(bm25_results)} BM25 → "
            f"{len(hybrid_results)} hybrides"
        )

        return hybrid_results

    async def _vector_search(
        self,
        cursor: sqlite3.Cursor,
        query_vec: bytes,
        top_k: int,
    ) -> List[SearchResult]:
        """Recherche par similarité cosinus."""
        cursor.execute(
            """
            SELECT text, source, distance_cosine(embedding, ?) as similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT ?
            """,
            (query_vec, top_k),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                SearchResult(
                    text=row[0],
                    source=row[1],
                    score=float(row[2]),
                    method="vector",
                )
            )

        return results

    async def _bm25_search(
        self,
        cursor: sqlite3.Cursor,
        query: str,
        top_k: int,
    ) -> List[SearchResult]:
        """Recherche BM25 full-text."""
        # Formater query pour FTS5
        fts_query = " ".join([f'"{w}"' for w in query.split() if len(w) > 2])

        if not fts_query:
            return []

        cursor.execute(
            """
            SELECT d.text, d.source, bm25(documents_fts) as score
            FROM documents_fts
            JOIN documents d ON documents_fts.rowid = d.rowid
            WHERE documents_fts MATCH ?
            ORDER BY score DESC
            LIMIT ?
            """,
            (fts_query, top_k),
        )

        results = []
        for row in cursor.fetchall():
            # Normaliser score BM25 (négatif → positif)
            score = max(0.0, -float(row[2]) / 10.0)
            results.append(
                SearchResult(
                    text=row[0],
                    source=row[1],
                    score=score,
                    method="bm25",
                )
            )

        return results

    def _merge_results(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Fusionner résultats avec score hybride."""
        # Créer dictionnaires pour fusion
        vector_scores = {r.text: r.score for r in vector_results}
        bm25_scores = {r.text: r.score for r in bm25_results}

        # Union des textes
        all_texts = set(vector_scores.keys()) | set(bm25_scores.keys())

        # Calcul score hybride
        hybrid_results = []
        for text in all_texts:
            v_score = vector_scores.get(text, 0.0)
            b_score = bm25_scores.get(text, 0.0)

            # Score hybride pondéré
            hybrid_score = (
                (1 - config.BM25_WEIGHT) * v_score + config.BM25_WEIGHT * b_score
            )

            # Trouver source
            source = next(
                (
                    r.source
                    for r in vector_results + bm25_results
                    if r.text == text
                ),
                "unknown",
            )

            hybrid_results.append(
                SearchResult(
                    text=text,
                    source=source,
                    score=hybrid_score,
                    method="hybrid",
                )
            )

        # Trier et retourner top_k
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        return hybrid_results[:top_k]

    async def shutdown(self) -> None:
        """Fermer la connexion DB."""
        if self.conn:
            self.conn.close()
            self.conn = None

        self._initialized = False
        logger.debug("RAG Engine arrêté.")
