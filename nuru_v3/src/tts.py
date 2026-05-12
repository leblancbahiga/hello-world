"""
Text-to-Speech avec Piper.

Synthèse vocale locale ultra-rapide.
Bufferisation par phrases pour latence minimale.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from src.config import config


logger = logging.getLogger(__name__)


class TTS:
    """
    Text-to-Speech avec Piper.

    Optimisations:
    - Chargement lazy du modèle
    - Bufferisation par phrases
    - Queue async pour streaming
    - Libération après usage
    """

    def __init__(self):
        self._engine = None
        self._voice = None
        self._initialized = False
        self._queue: Optional[asyncio.Queue] = None
        self._playback_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialiser le moteur TTS."""
        if self._initialized:
            return

        logger.debug("TTS initialisé (chargement lazy)")

        try:
            from piper import PiperVoice

            # Charger voix
            voice_path = await self._download_voice()

            if voice_path and voice_path.exists():
                self._voice = PiperVoice.load(str(voice_path))
                logger.info(f"Voix Piper chargée: {config.PIPER_VOICE}")
            else:
                logger.warning("Voix Piper non disponible")

        except ImportError:
            logger.warning("piper-tts non installé, TTS désactivé")

        self._queue = asyncio.Queue()
        self._initialized = True

    async def _download_voice(self) -> Optional[Path]:
        """Télécharger la voix Piper si nécessaire."""
        voice_dir = config.MODELS_DIR / "piper"
        voice_dir.mkdir(parents=True, exist_ok=True)

        voice_path = voice_dir / f"{config.PIPER_VOICE}.onnx"

        if not voice_path.exists():
            logger.info(f"Téléchargement voix Piper: {config.PIPER_VOICE}")

            try:
                import subprocess

                # Utiliser le script de téléchargement Piper
                subprocess.run(
                    [
                        "python",
                        "-m",
                        "piper.download",
                        config.PIPER_VOICE,
                        str(voice_dir),
                    ],
                    check=False,
                    capture_output=True,
                )

            except Exception as e:
                logger.error(f"Erreur téléchargement voix: {e}")
                return None

        return voice_path

    async def speak(self, text: str) -> None:
        """
        Synthétiser et jouer un texte.

        Args:
            text: Texte à synthétiser
        """
        if not self._initialized:
            await self.initialize()

        if not self._voice:
            logger.debug("TTS non disponible, skip")
            return

        logger.debug(f"TTS: {len(text)} caractères")

        try:
            from piper import synthesize

            # Synthétiser audio
            audio_data = synthesize(text, self._voice)

            # Jouer audio (platform-specific)
            await self._play_audio(audio_data)

        except Exception as e:
            logger.error(f"Erreur TTS: {e}", exc_info=True)

    async def speak_buffered(self, sentence: str) -> None:
        """
        Ajouter une phrase au buffer de lecture.

        Pour lecture fluide en streaming.
        """
        if self._queue:
            await self._queue.put(sentence)

            # Démarrer playback si pas déjà fait
            if not self._playback_task or self._playback_task.done():
                self._playback_task = asyncio.create_task(self._playback_loop())

    async def _playback_loop(self) -> None:
        """Boucle de lecture du buffer."""
        assert self._queue is not None

        while True:
            try:
                sentence = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=2.0,
                )
                await self.speak(sentence)
                self._queue.task_done()

            except asyncio.TimeoutError:
                break

            except Exception as e:
                logger.error(f"Erreur playback: {e}")
                break

    async def _play_audio(self, audio_data: bytes) -> None:
        """Jouer des données audio (implementation platform-specific)."""
        # macOS: utiliser afplay
        import tempfile
        import subprocess

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay",
                temp_path,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

        except Exception as e:
            logger.error(f"Erreur lecture audio: {e}")

        finally:
            # Nettoyer fichier temporaire
            try:
                Path(temp_path).unlink()
            except:
                pass

    async def shutdown(self) -> None:
        """Arrêter le TTS."""
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

        if self._queue:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except:
                    break

        self._voice = None
        self._initialized = False
        logger.debug("TTS arrêté.")
