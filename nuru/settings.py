"""Configuration de NURU Lite en couches.

Hiérarchie : défauts < fichiers (utilisateur). Les plugins recevront leur
propre espace de configuration (scope `plugins.<id>`), sans toucher au noyau.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("nuru.settings")

DEFAULTS: dict = {
    "server": {"host": "127.0.0.1", "port": 8734},
    "llm": {
        "system_prompt": (
            "Tu es NURU, un assistant personnel discret et efficace. "
            "Réponds en français, de manière concise et utile."
        ),
        "default_provider": "ollama",
        "default_model": "",
        "temperature": 0.7,
        "max_tokens": None,
        "context_window": 8192,
    },
    "providers": {
        "openai": {"base_url": "", "api_key": "", "models": []},
        "ollama": {"base_url": "http://localhost:11434", "models": []},
        "echo": {"models": ["echo"]},
    },
    "chat": {"max_history_messages": 20},
    "rag": {
        "embedder": "hash",  # "hash" (hors-ligne) | "ollama"
        "embedder_model": "nomic-embed-text",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 5,
        "vector_pool": 25,
        "collection": "default",
    },
    "ui": {"title": "NURU Lite", "theme": "dark"},
}


def _dig(data: dict, key: str) -> object | None:
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _poke(data: dict, key: str, value: object) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


class Settings:
    """Paramètres persistés dans un fichier JSON (config utilisateur)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict = copy.deepcopy(DEFAULTS)
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                self._merge(self._data, loaded)
                logger.info("settings loaded from %s", self.path)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("settings unreadable (%s), using defaults", exc)
        else:
            logger.info("no settings file at %s, using defaults", self.path)

    @staticmethod
    def _merge(base: dict, overlay: dict) -> None:
        for key, value in overlay.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                Settings._merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: object | None = None) -> object | None:
        return _dig(self._data, key) if _dig(self._data, key) is not None else default

    def set(self, key: str, value: object) -> None:
        _poke(self._data, key, value)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        return copy.deepcopy(self._data)

    @property
    def home_dir(self) -> Path:
        return self.path.parent


def default_home() -> Path:
    return Path(os.environ.get("NURU_HOME", "~/.nuru")).expanduser()
