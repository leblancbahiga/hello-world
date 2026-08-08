"""Contrats LLM de NURU Lite.

Un seul protocole d'agent fournisseur : OpenAI-compatible en ligne de mire.
Ollama, LM Studio, MLX et les API cloud exposent tous ce contrat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

ProviderConfig = dict[str, Any]


@dataclass(slots=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class ModelInfo:
    id: str
    provider: str
    size: str = ""
    family: str = ""


class LLMProvider(ABC):
    """Client minimal d'un fournisseur LLM (locale ou cloud)."""

    name: str

    @abstractmethod
    async def list_models(self) -> list[ModelInfo]:
        """Liste les modèles disponibles. Aucun réseau au constructeur."""

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream du texte généré, delta par delta."""

    async def close(self) -> None:
        pass

    async def __aenter__(self) -> "LLMProvider":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


def estimate_tokens(text: str) -> int:
    """Estimation grossière (~4 caractères/token) sans tokenizer."""
    return max(1, len(text) // 4)
