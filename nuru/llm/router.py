"""Router : sélection locale du fournisseur/modèle.

Règles 100 % en mémoire (aucun réseau) → < 5 ms garanti.
Les capacités (chat, embeddings, vision…) serviront à orienter le choix
vers le bon modèle ; la v1 gère déjà le « défaut » et la chute de secours.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nuru.llm.base import ModelInfo
from nuru.llm.echo import EchoProvider

if TYPE_CHECKING:
    from nuru.llm.base import LLMProvider
    from nuru.settings import Settings

logger = logging.getLogger("nuru.llm.router")


class Router:
    def __init__(self, settings: "Settings", providers: dict[str, "LLMProvider"]) -> None:
        self._settings = settings
        self.providers = providers

    def provider_names(self) -> list[str]:
        return list(self.providers.keys())

    def get(self, provider: str | None) -> "LLMProvider":
        name = provider or str(self._settings.get("llm.default_provider", "echo"))
        return self.providers.get(name) or EchoProvider()

    def resolve(self, provider: str | None = None, model: str | None = None) -> tuple[str, str]:
        """Retourne (provider_name, model_id) à partir des demandes et des défauts."""
        provider = provider or str(self._settings.get("llm.default_provider", "echo"))
        prov = self.providers.get(provider)
        if prov is None:
            provider = "echo"
            prov = self.providers[provider]
        if not model:
            model = str(self._settings.get("llm.default_model", "") or "")
        if not model:
            model = getattr(prov, "default_model", "") or ""
        if not model:
            models = self._cached_models(prov)
            model = models[0].id if models else "echo"
        return provider, model

    def _cached_models(self, prov: "LLMProvider") -> list[ModelInfo]:
        """Liste locale sans réseau (configuration). Évite un appel au boot."""
        if prov.name == "echo":
            return [ModelInfo(id="echo", provider="echo")]
        configured = self._settings.get(f"providers.{prov.name}.models")
        if isinstance(configured, list) and configured:
            return [ModelInfo(id=m, provider=prov.name) for m in configured]
        return []
