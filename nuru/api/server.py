"""Couche API locale : REST (conversations, modèles, santé) + WebSocket de chat.

Délègue toute la logique aux managers (Storage, Router, LLM). Aucune
persistance directe ici, aucune connexion directe aux fournisseurs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from nuru import __version__
from nuru.llm.base import ChatMessage, estimate_tokens
from nuru.meta import rss_mb, uptime_s

if TYPE_CHECKING:
    from nuru.core import NuruApp

logger = logging.getLogger("nuru.api")

UI_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "ui"

_MODELS_TTL = 30.0  # cache de la liste des modèles (s)


class ApiServer:
    def __init__(self, app: "NuruApp") -> None:
        self.app = app
        self.fastapi_app = FastAPI(title="NURU Lite", version=__version__)
        self._models_cache: dict[str, Any] = {"ts": 0.0, "data": None}
        self._register_routes()

    # ------------------------------------------------------------------ routes

    def _register_routes(self) -> None:
        api = self.fastapi_app
        api.get("/api/health")(self.health)
        api.get("/api/conversations")(self.list_conversations)
        api.post("/api/conversations")(self.create_conversation)
        api.get("/api/conversations/{cid}")(self.get_conversation)
        api.patch("/api/conversations/{cid}")(self.rename_conversation)
        api.delete("/api/conversations/{cid}")(self.delete_conversation)
        api.get("/api/models")(self.list_models)
        api.get("/api/settings")(self.get_settings)
        api.put("/api/settings")(self.put_settings)
        api.get("/api/documents")(self.list_documents)
        api.post("/api/documents/import")(self.import_documents)
        api.post("/api/documents/import/path")(self.import_document_path)
        api.get("/api/documents/{did}")(self.get_document)
        api.get("/api/documents/{did}/chunks")(self.get_document_chunks)
        api.delete("/api/documents/{did}")(self.delete_document)
        api.post("/api/rag/search")(self.rag_search)
        api.websocket("/api/ws/chat")(self.chat_ws)
        # servation de l'UI statique en dernier (les routes passent avant)
        api.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    # ------------------------------------------------------------------- REST

    async def health(self):
        return {
            "status": "ok",
            "version": __version__,
            "uptime_s": round(uptime_s(self.app._started_at), 2),
            "ram_mb": round(rss_mb(), 1),
            "providers": self.app.router.provider_names(),
        }

    async def list_conversations(self):
        items = await asyncio.to_thread(self.app.storage.list_conversations)
        return {"items": items}

    async def create_conversation(self, body: dict):
        model = body.get("model")
        provider = body.get("provider")
        title = body.get("title")
        conv = await asyncio.to_thread(
            self.app.storage.create_conversation, model, provider, title
        )
        return conv

    async def get_conversation(self, cid: str):
        conv = await asyncio.to_thread(self.app.storage.get_conversation, cid)
        if conv is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        conv["messages"] = await asyncio.to_thread(
            self.app.storage.list_messages, cid
        )
        return conv

    async def rename_conversation(self, cid: str, body: dict):
        title = (body.get("title") or "").strip()
        if not title:
            return JSONResponse({"error": "title required"}, status_code=400)
        await asyncio.to_thread(self.app.storage.rename_conversation, cid, title)
        return {"ok": True}

    async def delete_conversation(self, cid: str):
        await asyncio.to_thread(self.app.storage.delete_conversation, cid)
        return {"ok": True}

    async def list_models(self):
        cache = self._models_cache
        if time.time() - cache["ts"] > _MODELS_TTL or cache["data"] is None:
            raw = self.app.settings.get("providers")
            configured = raw if isinstance(raw, dict) else {}
            providers = {}
            if self.app.router:
                providers = self.app.router.providers
            items: list[dict] = []
            for name, prov in providers.items():
                try:
                    models = await prov.list_models()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("models listing failed for %s: %s", name, exc)
                    models = []
                items.append({
                    "provider": name,
                    "base_url": str(configured.get(name, {}).get("base_url", "")),
                    "models": [m.id for m in models],
                })
            provider, model = self.app.router.resolve()
            cache["data"] = {
                "items": items,
                "defaults": {"provider": provider, "model": model},
            }
            cache["ts"] = time.time()
        return cache["data"]

    async def get_settings(self):
        data = self.app.settings.to_dict()
        _mask_keys(data, "api_key")
        return data

    async def put_settings(self, body: dict):
        for key, value in _flatten(body):
            self.app.settings.set(key, value)
        self.app.settings.save()
        # la liste des modèles et l'embedder dépendent de la config
        self._models_cache["ts"] = 0.0
        self.app.rag.configure()
        return {"ok": True}

    # -------------------------------------------------------------- documents

    async def list_documents(self):
        collection = None
        items = await asyncio.to_thread(self.app.storage.list_documents, collection)
        return {"items": items}

    async def import_documents(self, files: list[UploadFile] | None = File(None),
                               collection: str = Form("default")):
        files = files or []
        if not files:
            return JSONResponse({"error": "aucun fichier"}, status_code=400)
        results = []
        for fh in files:
            data = await fh.read()
            results.append(await self._spawn_ingest(
                fh.filename or "fichier", data,
                mime=fh.content_type, collection=collection,
            ))
        return {"items": results}

    async def import_document_path(self, body: dict):
        path = (body.get("path") or "").strip()
        if not path:
            return JSONResponse({"error": "path required"}, status_code=400)
        p = Path(path).expanduser()
        if not p.is_file():
            return JSONResponse({"error": f"fichier introuvable: {p}"}, status_code=404)
        data = await asyncio.to_thread(p.read_bytes)
        doc = await self._spawn_ingest(
            p.name, data, mime=None, path=str(p),
            collection=str(body.get("collection", "default")),
        )
        return doc

    async def _spawn_ingest(self, name: str, data: bytes, *,
                            mime: str | None, collection: str,
                            path: str | None = None) -> dict:
        """Démarre l'indexation en tâche de fond (l'API ne bloque jamais)."""
        import hashlib
        import uuid

        # idempotence : déjà indexé avec le même contenu → on renvoie tel quel
        file_hash = hashlib.sha256(data).hexdigest()
        existing = await asyncio.to_thread(
            self.app.storage.find_document_by_hash, file_hash
        )
        if existing and existing["name"] == name:
            return existing

        doc_id = uuid.uuid4().hex[:16]
        await asyncio.to_thread(
            self.app.storage.upsert_document, {
                "id": doc_id, "collection": collection, "name": name, "mime": mime,
                "size": len(data), "hash": "", "status": "pending",
            }
        )
        asyncio.create_task(
            self.app.rag.ingest(
                name, data, mime=mime, path=path, collection=collection, doc_id=doc_id
            ),
            name=f"ingest-{doc_id}",
        )
        return await asyncio.to_thread(self.app.storage.get_document, doc_id)

    async def get_document(self, did: str):
        doc = await asyncio.to_thread(self.app.storage.get_document, did)
        if doc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return doc

    async def get_document_chunks(self, did: str):
        doc = await asyncio.to_thread(self.app.storage.get_document, did)
        if doc is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        chunks = await asyncio.to_thread(self.app.storage.get_document_chunks, did)
        return {"items": chunks}

    async def delete_document(self, did: str):
        await asyncio.to_thread(self.app.storage.delete_document, did)
        return {"ok": True}

    async def rag_search(self, body: dict):
        query = (body.get("query") or "").strip()
        if not query:
            return JSONResponse({"error": "query required"}, status_code=400)
        top_k = int(body.get("top_k", 0) or 0) or None
        results = await self.app.rag.search(query, top_k=top_k)
        return {"items": results, "count": len(results)}

    # -------------------------------------------------------------- WebSocket

    async def chat_ws(self, ws: WebSocket) -> None:
        await ws.accept()
        state = {"task": None}
        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")
                if msg_type == "open":
                    await self._ws_open(ws, data)
                elif msg_type == "send":
                    task = state["task"]
                    if task and not task.done():
                        await ws.send_json({
                            "type": "error",
                            "message": "Une génération est déjà en cours.",
                        })
                        continue
                    state["task"] = asyncio.create_task(
                        self._ws_send(ws, state, data),
                        name="ws-chat-send",
                    )
                elif msg_type == "stop":
                    task = state["task"]
                    if task and not task.done():
                        task.cancel()
        except WebSocketDisconnect:
            logger.info("client déconnecté")
        finally:
            task = state["task"]
            if task and not task.done():
                task.cancel()

    async def _ws_open(self, ws: WebSocket, data: dict) -> None:
        conv_id = data.get("conversation_id")
        if conv_id:
            conv = await asyncio.to_thread(self.app.storage.get_conversation, conv_id)
            if conv:
                msgs = await asyncio.to_thread(
                    self.app.storage.list_messages, conv_id
                )
                await ws.send_json({
                    "type": "history",
                    "conversation_id": conv_id,
                    "title": conv["title"],
                    "messages": msgs,
                })
                return
        await ws.send_json({
            "type": "history",
            "conversation_id": None,
            "messages": [],
        })

    async def _ws_send(self, ws: WebSocket, state: dict, data: dict) -> None:
        store = self.app.storage
        content = (data.get("content") or "").strip()
        if not content:
            await ws.send_json({"type": "error", "message": "Message vide."})
            return

        provider_name, model = self.app.router.resolve(
            data.get("provider"), data.get("model")
        )
        conv_id = data.get("conversation_id")
        conv = None
        if conv_id:
            conv = await asyncio.to_thread(store.get_conversation, conv_id)

        parts: list[str] = []
        try:
            if conv is None:
                conv = await asyncio.to_thread(
                    store.create_conversation, model, provider_name
                )
                conv_id = conv["id"]

            await asyncio.to_thread(store.add_message, conv_id, "user", content)
            await self.app.events.publish(
                "message.created",
                {"conversation_id": conv_id, "role": "user", "content": content},
            )

            if conv["title"] == "Nouvelle conversation":
                title = content[:40] + ("…" if len(content) > 40 else "")
                await asyncio.to_thread(store.rename_conversation, conv_id, title)
                conv["title"] = title

            history = await asyncio.to_thread(store.list_messages, conv_id)

            sources: list[dict] = []
            rag_context = None
            if data.get("use_rag") and self.app.rag:
                rag_results = await self.app.rag.search(content, top_k=3)
                rag_context = self.app.rag.format_context(rag_results)
                sources = rag_results

            messages = self._build_messages(history, rag_context)

            await ws.send_json({
                "type": "start",
                "conversation_id": conv_id,
                "title": conv["title"],
                "provider": provider_name,
                "model": model,
            })

            provider = self.app.router.get(provider_name)
            temperature = self.app.settings.get("llm.temperature")
            max_tokens = self.app.settings.get("llm.max_tokens")
            async for delta in provider.stream_chat(
                messages,
                model=model,
                temperature=temperature if temperature is not None else None,
                max_tokens=max_tokens if max_tokens is not None else None,
            ):
                if not delta:
                    continue
                parts.append(delta)
                await ws.send_json({"type": "delta", "content": delta})

            text = "".join(parts)
            if text.strip():
                msg = await asyncio.to_thread(
                    store.add_message, conv_id, "assistant", text,
                    estimate_tokens(text),
                )
                await self.app.events.publish(
                    "message.created",
                    {"conversation_id": conv_id, "role": "assistant", "content": text},
                )
                await ws.send_json({
                    "type": "done",
                    "conversation_id": conv_id,
                    "message_id": msg["id"],
                    "sources": sources,
                })
            else:
                await ws.send_json({"type": "error", "message": "Réponse vide."})
        except asyncio.CancelledError:
            text = "".join(parts)
            if text.strip():
                await asyncio.to_thread(
                    store.add_message, conv_id, "assistant", text,
                    estimate_tokens(text),
                )
            await ws.send_json({"type": "stopped", "conversation_id": conv_id})
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat stream failed")
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:  # noqa: BLE001
                pass

    def _build_messages(self, history: list[dict],
                        rag_context: str | None = None) -> list[ChatMessage]:
        system_prompt = str(self.app.settings.get("llm.system_prompt", ""))
        max_messages = int(self.app.settings.get("chat.max_history_messages", 20))
        msgs: list[ChatMessage] = []
        if system_prompt:
            msgs.append(ChatMessage(role="system", content=system_prompt))
        tail = history[-max_messages:]
        for row in tail:
            if row["role"] in ("user", "assistant"):
                msgs.append(ChatMessage(role=row["role"], content=row["content"]))
        if rag_context:
            # contexte RAG inséré juste avant la dernière question de l'utilisateur
            if msgs and msgs[-1].role == "user":
                msgs.insert(-1, ChatMessage(role="system", content=rag_context))
            else:
                msgs.append(ChatMessage(role="system", content=rag_context))
        return msgs


# ------------------------------------------------------------------ helpers


def _flatten(data: dict, prefix: str = "") -> list[tuple[str, object]]:
    out: list[tuple[str, object]] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.extend(_flatten(value, path))
        else:
            out.append((path, value))
    return out


def _mask_keys(data: dict, secret: str) -> None:
    for key, value in list(data.items()):
        if key == secret and isinstance(value, str) and value:
            data[key] = "••• (configuré)"
        elif isinstance(value, dict):
            _mask_keys(value, secret)
