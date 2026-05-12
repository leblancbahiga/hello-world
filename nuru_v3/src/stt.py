"""
Speech-to-Text avec MLX Whisper.

Transcription audio locale ultra-rapide.
Lazy loading et libération immédiate après usage.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import mlx.core as mx
from mlx_whisper import transcribe

from src.config import config


logger = logging.getLogger(__name__)


class STT:
    """
    Speech-to-Text avec Whisper via MLX.

    Optimisations:
    - Chargement lazy uniquement lors du premier usage
    - Libération explicite après transcription
    - Modèle tiny par défaut pour vitesse
    - Exécution dans thread pool pour non-blocage
    """

    def __init__(self):
        self._model_path: Optional[Path] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Préparer le STT (chargement différé)."""
        if self._initialized:
            return

        logger.debug("STT initialisé (chargement lazy)")
        self._initialized = True

    async def transcribe(
        self,
        audio_path: str,
        language: str = "fr",
    ) -> str:
        """
        Transcrire un fichier audio.

        Args:
            audio_path: Chemin vers fichier audio (wav, mp3, m4a)
            language: Langue de l'audio

        Returns:
            Texte transcrit
        """
        if not self._initialized:
            await self.initialize()

        path = Path(audio_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio non trouvé: {audio_path}")

        logger.info(f"Transcription de {path.name}...")

        # Whisper dans executor pour non-blocage
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            str(path),
            language,
        )

        logger.debug(f"Transcription terminée: {len(result)} caractères")
        return result

    def _transcribe_sync(self, audio_path: str, language: str) -> str:
        """Transcription synchrone (dans executor)."""
        try:
            result = transcribe(
                audio_path,
                path_or_hf_repo=config.WHISPER_MODEL,
                language=language,
                verbose=False,
            )

            # Nettoyer cache Metal après usage
            mx.metal.clear_cache()

            import gc
            gc.collect()

            return result.get("text", "").strip()

        except Exception as e:
            logger.error(f"Erreur transcription: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Libérer ressources."""
        self._initialized = False
        logger.debug("STT arrêté.")
