"""Client Ollama local via son endpoint OpenAI-compatible (/v1).

Le même client que les API OpenAI-compatibles, sur le port Ollama.
"""

from __future__ import annotations

from nuru.llm.base import ProviderConfig
from nuru.llm.openai_compat import OpenAICompatProvider

OLLAMA_DEFAULT_MODEL = "llama3.2"


class OllamaProvider(OpenAICompatProvider):
    def __init__(self, name: str, config: ProviderConfig) -> None:
        base = str(config.get("base_url", "http://localhost:11434")).rstrip("/")
        super().__init__(name, {**config, "base_url": base + "/v1"})

    @property
    def default_model(self) -> str:
        return OLLAMA_DEFAULT_MODEL
