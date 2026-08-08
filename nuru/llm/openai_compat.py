"""Client OpenAI-compatible générique (LM Studio, vLLM, cloud…).

Parsing SSE du flux `data:` — aucune dépendance au SDK OpenAI.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx

from nuru.llm.base import ChatMessage, LLMProvider, ModelInfo, ProviderConfig

logger = logging.getLogger("nuru.llm.openai")


class OpenAICompatProvider(LLMProvider):
    def __init__(self, name: str, config: ProviderConfig) -> None:
        self.name = name
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", ""))
        self._configured_models = list(config.get("models") or [])
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    120.0, connect=5.0, read=120.0, write=30.0, pool=5.0
                ),
            )
        return self._client

    async def list_models(self) -> list[ModelInfo]:
        if self._configured_models:
            return [ModelInfo(id=m, provider=self.name) for m in self._configured_models]
        if not self.base_url:
            return []
        client = self._ensure_client()
        try:
            resp = await client.get("/models")
            resp.raise_for_status()
            data = resp.json()
            return [
                ModelInfo(id=item["id"], provider=self.name)
                for item in data.get("data", [])
            ]
        except Exception as exc:  # noqa: BLE001 — fournisseur indisponible = pas de panique
            logger.warning("list_models failed for %s: %s", self.name, exc)
            return []

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        body: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        try:
            async with client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode(errors="replace")[:300]
                    raise RuntimeError(
                        f"{self.name} HTTP {resp.status_code}: {text}"
                    )
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {}).get("content")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise RuntimeError(f"{self.name} connection failed: {exc}") from exc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
