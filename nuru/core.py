"""Core de NURU Lite : cycle de vie et orchestration des services.

Règle : le Core ne contient aucune logique métier et n'importe aucune
bibliothèque lourde. Il câble les modules et les lance.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from nuru import __version__
from nuru.events import EventBus
from nuru.llm.base import LLMProvider
from nuru.llm.echo import EchoProvider
from nuru.llm.ollama import OllamaProvider
from nuru.llm.openai_compat import OpenAICompatProvider
from nuru.llm.router import Router
from nuru.rag import RAGManager
from nuru.settings import Settings
from nuru.storage import Storage

logger = logging.getLogger("nuru.core")


class NuruApp:
    def __init__(self, home_dir: Path) -> None:
        self.home_dir = home_dir
        self.version = __version__
        self.events = EventBus()
        self.settings = Settings(home_dir / "settings.json")
        self.storage = Storage(home_dir / "data.db")
        self.router: Router | None = None
        self.rag: RAGManager | None = None
        self.server = None  # ApiServer, construit par app.start()
        self._started_at = 0.0

    def build_llm(self) -> None:
        providers: dict[str, LLMProvider] = {}
        raw = self.settings.get("providers")
        if isinstance(raw, dict):
            for name, cfg in raw.items():
                cfg = cfg or {}
                if name == "ollama":
                    providers[name] = OllamaProvider(name, cfg)
                elif name == "echo":
                    providers[name] = EchoProvider(cfg)
                else:
                    # openai et tout fournisseur inconnu → protocole OpenAI-compatible
                    providers[name] = OpenAICompatProvider(name, cfg)
        providers.setdefault("echo", EchoProvider())
        self.router = Router(self.settings, providers)
        logger.info("providers prêts: %s", ", ".join(providers))

    async def start(self) -> None:
        self._started_at = time.time()
        self.storage.connect()
        self.build_llm()
        self.rag = RAGManager(self.storage, self.settings, self.events)
        # import différé : l'API (FastAPI) n'est pas nécessaire au boot métier
        from nuru.api.server import ApiServer

        self.server = ApiServer(self)
        await self.events.publish("app.started", {"version": self.version})

    async def shutdown(self) -> None:
        await self.events.publish("app.stopped", {})
        await self.events.wait_for_handlers()
        if self.router:
            for provider in self.router.providers.values():
                try:
                    await provider.close()
                except Exception:  # noqa: BLE001
                    logger.exception("close failed for %s", provider.name)
        self.storage.close()
        logger.info("nuru arrêté proprement")
