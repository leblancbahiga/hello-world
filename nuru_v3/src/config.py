"""
Configuration centrale de NURU V3.

Optimisée pour MacBook M1 8 Go avec gestion mémoire agressive.
"""

import os
import sys
from pathlib import Path
from typing import Final

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration globale de NURU V3."""

    model_config = SettingsConfigDict(
        env_prefix="NURU_",
        env_file=".env",
        case_sensitive=False,
    )

    # =====================================================================
    # CHEMINS
    # =====================================================================

    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEXES_DIR: Path = BASE_DIR / "indexes"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # =====================================================================
    # MODÈLES LLM
    # =====================================================================

    # Nano modèle - ultra rapide (~600MB RAM)
    MODEL_NANO_ID: str = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"
    MODEL_NANO_PATH: Path = None  # type: ignore

    # Modèle standard - équilibre qualité/vitesse (~2.5GB RAM)
    MODEL_STANDARD_ID: str = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    MODEL_STANDARD_PATH: Path = None  # type: ignore

    # Cloud fallback
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1/chat/completions"

    # =====================================================================
    # EMBEDDINGS
    # =====================================================================

    EMBEDDING_MODEL_ID: str = "mlx-community/multilingual-e5-small-mlx"
    EMBEDDING_MODEL_PATH: Path = None  # type: ignore
    EMBEDDING_DIM: int = 384

    # =====================================================================
    # RAG
    # =====================================================================

    DB_PATH: Path = None  # type: ignore
    CHUNK_SIZE: int = 400  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    RAG_TOP_K: int = 5
    RAG_CONFIDENCE_THRESHOLD: float = 0.35
    BM25_WEIGHT: float = 0.3  # Poids du BM25 dans le score hybride

    # =====================================================================
    # CACHE
    # =====================================================================

    CACHE_TTL_SECONDS: int = 300
    CACHE_MAX_ENTRIES: int = 1000

    # =====================================================================
    # AUDIO
    # =====================================================================

    WHISPER_MODEL: str = "tiny"  # tiny, base, small
    PIPER_VOICE: str = "fr_FR-siwis-medium"
    AUDIO_SAMPLE_RATE: int = 22050
    AUDIO_CHUNK_DURATION: float = 0.5  # secondes

    # =====================================================================
    # MÉMOIRE
    # =====================================================================

    RAM_THRESHOLD_LOW_GB: float = 2.5
    RAM_THRESHOLD_CRITICAL_GB: float = 1.5
    MEMORY_CHECK_INTERVAL_SEC: float = 60.0

    # =====================================================================
    # UI
    # =====================================================================

    UI_WIDTH: int = 800
    UI_HEIGHT: int = 600
    UI_OPACITY: float = 0.92
    UI_FONT_FAMILY: str = "SF Mono"
    UI_PRIMARY_COLOR: str = "#00FFFF"  # Cyan cyberpunk
    UI_SECONDARY_COLOR: str = "#FF00FF"  # Magenta
    UI_BG_COLOR: str = "#0A0A0F"  # Noir profond

    # =====================================================================
    # PERFORMANCE
    # =====================================================================

    MAX_TOKENS_NANO: int = 512
    MAX_TOKENS_STANDARD: int = 1024
    MAX_TOKENS_CLOUD: int = 2048
    TEMPERATURE: float = 0.3
    TOP_P: float = 0.9

    # =====================================================================
    # LOGGING
    # =====================================================================

    LOG_LEVEL: str = "INFO"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 3

    # =====================================================================
    # INTENT CLASSIFIER
    # =====================================================================

    INTENT_MODEL_PATH: Path = None  # type: ignore

    # =====================================================================
    # INITIALISATION
    # =====================================================================

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialiser les chemins dérivés
        self.MODEL_NANO_PATH = self.MODELS_DIR / "nano"
        self.MODEL_STANDARD_PATH = self.MODELS_DIR / "standard"
        self.EMBEDDING_MODEL_PATH = self.MODELS_DIR / "embedder"
        self.DB_PATH = self.INDEXES_DIR / "nuru_vector.db"
        self.INTENT_MODEL_PATH = self.MODELS_DIR / "intent" / "model.bin"

        # Créer les répertoires nécessaires
        for directory in [
            self.DATA_DIR,
            self.INDEXES_DIR,
            self.MODELS_DIR,
            self.LOGS_DIR,
            self.MODEL_NANO_PATH,
            self.MODEL_STANDARD_PATH,
            self.EMBEDDING_MODEL_PATH,
            self.INTENT_MODEL_PATH.parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


# Instance globale unique
config: Final[Config] = Config()
