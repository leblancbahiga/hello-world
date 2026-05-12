"""
Client LLM cloud pour fallback.

Utilise DeepSeek V4 Flash API quand:
- Intent COMPLEX détecté
- Requête dépasse capacités locales
- RAM insuffisante pour modèle standard
"""

import logging
from typing import AsyncGenerator, Optional

import httpx

from src.config import config


logger = logging.getLogger(__name__)


class LLMCloud:
    """
    Client HTTP pour API DeepSeek.

    Fallback intelligent avec:
    - Timeout strict
    - Retry minimal
    - Streaming support
    """

    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialiser le client HTTP."""
        if self._initialized:
            return

        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=2),
        )

        self._initialized = True
        logger.debug("Client cloud initialisé.")

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """
        Générer une réponse via API DeepSeek.

        Args:
            prompt: Prompt utilisateur
            system_prompt: Instructions système
            max_tokens: Maximum de tokens

        Yields:
            Tokens générés
        """
        if not self._initialized:
            await self.initialize()

        assert self.client is not None

        if not config.DEEPSEEK_API_KEY:
            raise ValueError("DeepSeek API key non configurée")

        logger.debug("Requête vers DeepSeek Cloud...")

        try:
            response = await self.client.post(
                config.DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": config.TEMPERATURE,
                    "top_p": config.TOP_P,
                    "stream": True,
                },
                timeout=30.0,
            )
            response.raise_for_status()

            # Parser stream SSE
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        break

                    import json
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

        except httpx.TimeoutException:
            logger.error("Timeout API DeepSeek")
            yield "[Erreur: Timeout API]"

        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP DeepSeek: {e}")
            yield f"[Erreur API: {e.response.status_code}]"

        except Exception as e:
            logger.error(f"Erreur cloud: {e}", exc_info=True)
            yield f"[Erreur: {str(e)}]"

    async def shutdown(self) -> None:
        """Fermer le client HTTP."""
        if self.client:
            await self.client.aclose()
            self.client = None

        self._initialized = False
        logger.debug("Client cloud fermé.")
