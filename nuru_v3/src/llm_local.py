"""
LLM local avec MLX.

Supporte deux modèles:
- Nano (1.5B): ultra rapide, faible RAM
- Standard (3B): équilibre qualité/vitesse

Gestion mémoire agressive avec déchargement explicite.
"""

import logging
from typing import AsyncGenerator, Optional

import mlx.core as mx
from mlx_lm import load, generate

from src.config import config


logger = logging.getLogger(__name__)


class LLMLocal:
    """
    Interface locale pour LLM via MLX.

    Optimisé pour Apple Silicon avec:
    - Chargement lazy des modèles
    - Déchargement explicite
    - Streaming tokens async
    - Cache Metal nettoyé
    """

    def __init__(self):
        self.model_nano = None
        self.model_standard = None
        self.tokenizer_nano = None
        self.tokenizer_standard = None
        self._nano_loaded = False
        self._standard_loaded = False

    async def initialize(self) -> None:
        """Initialiser les modèles (lazy loading réel au premier usage)."""
        logger.debug("LLM Local initialisé (chargement lazy)")

    async def _load_model(self, model_type: str = "standard") -> None:
        """
        Charger un modèle spécifique.

        Args:
            model_type: "nano" ou "standard"
        """
        if model_type == "nano":
            if self._nano_loaded:
                return

            logger.info(f"Chargement modèle nano: {config.MODEL_NANO_ID}")
            self.model_nano, self.tokenizer_nano = load(
                config.MODEL_NANO_ID,
                model_path=config.MODEL_NANO_PATH,
            )
            self._nano_loaded = True
            logger.debug("Modèle nano chargé.")

        else:  # standard
            if self._standard_loaded:
                return

            logger.info(f"Chargement modèle standard: {config.MODEL_STANDARD_ID}")
            self.model_standard, self.tokenizer_standard = load(
                config.MODEL_STANDARD_ID,
                model_path=config.MODEL_STANDARD_PATH,
            )
            self._standard_loaded = True
            logger.debug("Modèle standard chargé.")

    async def unload(self) -> None:
        """Décharger tous les modèles pour libérer la RAM."""
        logger.debug("Déchargement des modèles LLM...")

        self.model_nano = None
        self.model_standard = None
        self.tokenizer_nano = None
        self.tokenizer_standard = None
        self._nano_loaded = False
        self._standard_loaded = False

        # Nettoyer cache Metal
        mx.metal.clear_cache()

        import gc
        gc.collect()

        logger.info("Modèles LLM déchargés.")

    async def generate(
        self,
        prompt: str,
        model: str = "standard",
        system_prompt: str = "",
        max_tokens: Optional[int] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ) -> AsyncGenerator[str, None]:
        """
        Générer une réponse en streaming.

        Args:
            prompt: Prompt utilisateur
            model: "nano" ou "standard"
            system_prompt: Instructions système
            max_tokens: Maximum de tokens à générer
            temperature: Température de sampling
            top_p: Top-p sampling

        Yields:
            Tokens générés un par un
        """
        # Charger le modèle demandé
        await self._load_model(model)

        # Sélectionner modèle et tokenizer
        if model == "nano":
            model_obj = self.model_nano
            tokenizer = self.tokenizer_nano
            default_max_tokens = config.MAX_TOKENS_NANO
        else:
            model_obj = self.model_standard
            tokenizer = self.tokenizer_standard
            default_max_tokens = config.MAX_TOKENS_STANDARD

        assert model_obj is not None
        assert tokenizer is not None

        # Construire prompt complet
        full_prompt = self._build_prompt(prompt, system_prompt)

        # Déterminer max_tokens
        if max_tokens is None:
            max_tokens = default_max_tokens

        logger.debug(f"Génération avec {model} (max_tokens={max_tokens})")

        # Génération streaming avec MLX
        try:
            for token in generate(
                model=model_obj,
                tokenizer=tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                verbose=False,
            ):
                yield token

        except Exception as e:
            logger.error(f"Erreur génération: {e}", exc_info=True)
            yield f"[Erreur: {str(e)}]"

    def _build_prompt(self, user_prompt: str, system_prompt: str) -> str:
        """
        Construire le prompt formaté pour Qwen/DeepSeek.

        Format ChatML compatible.
        """
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
