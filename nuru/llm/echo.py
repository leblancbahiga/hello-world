"""Fournisseur « echo » : aucun réseau, aucun modèle.

Permet de valider le pipeline complet (UI → API → bus → streaming → SQLite)
sans Ollama ni clé API. Très utile pour les tests et la démo hors ligne.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from nuru.llm.base import ChatMessage, LLMProvider, ModelInfo


class EchoProvider(LLMProvider):
    name = "echo"

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id="echo", provider=self.name)]

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        last = next((m.content for m in reversed(messages) if m.role == "user"), "")
        reply = f"[echo] Vous avez dit : « {last} » — le pipeline NURU Lite fonctionne."
        for token in reply.split(" "):
            await asyncio.sleep(0.02)
            yield token + " "
