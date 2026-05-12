"""
Embedder MLX pour embeddings multilingues.

Utilise mlx-community/multilingual-e5-small-mlx
- 384 dimensions
- ~120MB RAM
- Ultra rapide sur M1
"""

import logging
from typing import List, Optional

import mlx.core as mx
from mlx_embedders import EmbeddingModel

from src.config import config


logger = logging.getLogger(__name__)


class Embedder:
    """
    Embedder basé sur MLX avec gestion mémoire agressive.

    Chargement lazy et déchargement explicite pour minimiser RAM.
    """

    def __init__(self):
        self.model: Optional[EmbeddingModel] = None
        self._initialized = False
        self._unloaded = False

    async def initialize(self) -> None:
        """Charger le modèle d'embeddings."""
        if self._initialized:
            return

        logger.debug("Initialisation de l'embedder...")

        model_path = config.EMBEDDING_MODEL_PATH

        # Télécharger/charger le modèle
        try:
            self.model = EmbeddingModel.from_pretrained(
                config.EMBEDDING_MODEL_ID,
                model_path=model_path,
            )
            logger.info(f"Embedder chargé: {config.EMBEDDING_MODEL_ID}")
        except Exception as e:
            logger.error(f"Erreur chargement embedder: {e}")
            raise

        self._initialized = True
        self._unloaded = False
        logger.debug("Embedder prêt.")

    async def unload(self) -> None:
        """Décharger le modèle pour libérer la RAM."""
        if self._unloaded or not self._initialized:
            return

        logger.debug("Déchargement de l'embedder...")

        self.model = None
        self._initialized = False
        self._unloaded = True

        # Nettoyer cache Metal
        mx.metal.clear_cache()

        import gc
        gc.collect()

        logger.debug("Embedder déchargé.")

    def is_unloaded(self) -> bool:
        """Vérifier si l'embedder est déchargé."""
        return self._unloaded

    async def embed(self, texts: List[str]) -> mx.array:
        """
        Générer des embeddings pour une liste de textes.

        Args:
            texts: Liste de textes à embedder

        Returns:
            Matrice d'embeddings (n_texts x 384)
        """
        if not self._initialized:
            await self.initialize()

        assert self.model is not None

        # Batch processing pour efficacité
        embeddings = self.model.encode(texts)

        return embeddings

    async def embed_single(self, text: str) -> mx.array:
        """
        Générer un embedding pour un seul texte.

        Args:
            text: Texte à embedder

        Returns:
            Embedding vector (384,)
        """
        embeddings = await self.embed([text])
        return embeddings[0]
