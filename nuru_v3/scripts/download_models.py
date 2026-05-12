#!/usr/bin/env python3
"""
Script de téléchargement des modèles MLX.

À exécuter après installation pour préparer les modèles locaux.
"""

import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def download_nano_model():
    """Télécharger le modèle nano 1.5B."""
    logger.info("Téléchargement modèle nano (DeepSeek-R1-Distill-Qwen-1.5B)...")

    try:
        from mlx_lm import load

        model, tokenizer = load(
            "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            model_path="models/nano",
        )

        logger.info("✅ Modèle nano téléchargé")

    except Exception as e:
        logger.error(f"Erreur téléchargement nano: {e}")


async def download_standard_model():
    """Télécharger le modèle standard 3B."""
    logger.info("Téléchargement modèle standard (Qwen2.5-3B-Instruct)...")

    try:
        from mlx_lm import load

        model, tokenizer = load(
            "mlx-community/Qwen2.5-3B-Instruct-4bit",
            model_path="models/standard",
        )

        logger.info("✅ Modèle standard téléchargé")

    except Exception as e:
        logger.error(f"Erreur téléchargement standard: {e}")


async def download_embedder():
    """Télécharger le modèle d'embeddings."""
    logger.info("Téléchargement embedder (multilingual-e5-small)...")

    try:
        from mlx_embedders import EmbeddingModel

        model = EmbeddingModel.from_pretrained(
            "mlx-community/multilingual-e5-small-mlx",
            model_path="models/embedder",
        )

        logger.info("✅ Embedder téléchargé")

    except Exception as e:
        logger.error(f"Erreur téléchargement embedder: {e}")


async def download_whisper():
    """Télécharger Whisper tiny."""
    logger.info("Téléchargement Whisper tiny...")

    try:
        from mlx_whisper import transcribe
        import tempfile

        # Télécharger en créant un fichier temporaire
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            # Fichier silence de 1 seconde
            import wave
            import struct

            with wave.open(f.name, "w") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(22050)
                for _ in range(22050):
                    wav.writeframes(struct.pack("<h", 0))

            transcribe(f.name, path_or_hf_repo="tiny", verbose=False)

        logger.info("✅ Whisper téléchargé")

    except Exception as e:
        logger.error(f"Erreur téléchargement Whisper: {e}")


async def main():
    """Télécharger tous les modèles."""
    logger.info("=" * 60)
    logger.info("NURU V3 - Téléchargement des modèles")
    logger.info("=" * 60)

    # Créer répertoires
    Path("models/nano").mkdir(parents=True, exist_ok=True)
    Path("models/standard").mkdir(parents=True, exist_ok=True)
    Path("models/embedder").mkdir(parents=True, exist_ok=True)

    # Télécharger modèles
    await download_nano_model()
    await download_standard_model()
    await download_embedder()
    await download_whisper()

    logger.info("=" * 60)
    logger.info("✅ Tous les modèles sont prêts")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
