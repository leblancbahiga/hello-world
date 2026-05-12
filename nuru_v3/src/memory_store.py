"""
Mémoire des interactions avec cache MD5.

- Cache exact queries (MD5) avec TTL
- Historique conversations
- Gestion mémoire agressive
"""

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import config


logger = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Réponse en cache."""

    query_hash: str
    response: str
    timestamp: float
    ttl: int


class MemoryStore:
    """
    Stockage mémoire avec cache et historique.

    Features:
    - Cache MD5 exact match (TTL 300s)
    - Historique SQLite
    - Nettoyage automatique
    """

    def __init__(self):
        self.conn: Optional[sqlite3.Connection] = None
        self.cache: dict[str, CachedResponse] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialiser la base de données mémoire."""
        if self._initialized:
            return

        logger.debug("Initialisation MemoryStore...")

        db_path = config.DATA_DIR / "memory.db"

        self.conn = sqlite3.connect(str(db_path))
        await self._create_tables()

        self._initialized = True
        logger.debug("MemoryStore prêt.")

    async def _create_tables(self) -> None:
        """Créer tables mémoire."""
        assert self.conn is not None

        cursor = self.conn.cursor()

        # Historique interactions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                intent_type TEXT,
                model_used TEXT
            )
        """)

        # Index pour recherche rapide
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON interactions(timestamp DESC)
        """)

        self.conn.commit()

    def _compute_hash(self, text: str) -> str:
        """Calculer hash MD5 d'un texte."""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_cached(self, query: str) -> Optional[str]:
        """
        Vérifier si réponse en cache.

        Args:
            query: Requête utilisateur

        Returns:
            Réponse cached ou None
        """
        query_hash = self._compute_hash(query)

        if query_hash in self.cache:
            cached = self.cache[query_hash]

            # Vérifier TTL
            if time.time() - cached.timestamp < cached.ttl:
                logger.debug(f"Cache hit: {query_hash}")
                return cached.response
            else:
                # Expired
                del self.cache[query_hash]
                logger.debug(f"Cache expired: {query_hash}")

        return None

    async def set_cached(
        self,
        query: str,
        response: str,
        ttl: int = 300,
    ) -> None:
        """
        Mettre en cache une réponse.

        Args:
            query: Requête utilisateur
            response: Réponse à cacher
            ttl: Time-to-live en secondes
        """
        query_hash = self._compute_hash(query)

        # Limiter taille du cache
        if len(self.cache) >= config.CACHE_MAX_ENTRIES:
            await self._evict_cache()

        self.cache[query_hash] = CachedResponse(
            query_hash=query_hash,
            response=response,
            timestamp=time.time(),
            ttl=ttl,
        )

        logger.debug(f"Cache set: {query_hash}")

    async def _evict_cache(self) -> None:
        """Évincer les entrées les plus anciennes."""
        # Garder seulement 50% des entrées les plus récentes
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].timestamp,
            reverse=True,
        )

        keep_count = config.CACHE_MAX_ENTRIES // 2
        self.cache = dict(sorted_entries[:keep_count])

        logger.debug(f"Cache evicted: {len(self.cache)} entries remaining")

    async def add_interaction(
        self,
        query: str,
        response: str,
        intent_type: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> None:
        """
        Sauvegarder une interaction dans l'historique.

        Args:
            query: Requête utilisateur
            response: Réponse de l'IA
            intent_type: Type d'intent détecté
            model_used: Modèle utilisé
        """
        if not self._initialized:
            await self.initialize()

        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO interactions (query, response, intent_type, model_used)
            VALUES (?, ?, ?, ?)
            """,
            (query, response, intent_type, model_used),
        )

        self.conn.commit()

    async def get_history(
        self,
        limit: int = 20,
    ) -> List[Tuple[str, str, str]]:
        """
        Récupérer l'historique récent.

        Args:
            limit: Nombre d'interactions

        Returns:
            Liste de (query, response, timestamp)
        """
        if not self._initialized:
            await self.initialize()

        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT query, response, timestamp
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )

        return cursor.fetchall()

    async def clear_history(self) -> None:
        """Effacer tout l'historique."""
        if not self._initialized:
            await self.initialize()

        assert self.conn is not None

        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM interactions")
        self.conn.commit()

        logger.info("Historique effacé.")

    async def shutdown(self) -> None:
        """Fermer la connexion."""
        if self.conn:
            self.conn.close()
            self.conn = None

        self.cache.clear()
        self._initialized = False
        logger.debug("MemoryStore arrêté.")
