"""Bus d'événements asynchrone, découplé et typé.

NURU Lite repose sur la publication/écoute d'événements pour relier les
modules (Core, RAG, Memory, plugins) sans couplage direct.

Règles :
- `publish` ne bloque jamais l'appelant : les handlers sont planifiés en tâches.
- Un handler en échec n'empêche jamais les autres handlers de tourner.
- Aucune connaissance des émetteurs/récepteurs n'est requise.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("nuru.events")

Handler = Callable[["Event"], Awaitable[None]]


@dataclass(slots=True)
class Event:
    topic: str
    payload: dict
    ts: float = field(default_factory=time.time)


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._tasks: set[asyncio.Task] = set()

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    def on(self, topic: str) -> Callable[[Handler], Handler]:
        def decorator(fn: Handler) -> Handler:
            self.subscribe(topic, fn)
            return fn

        return decorator

    async def publish(self, topic: str, payload: dict | None = None) -> None:
        event = Event(topic=topic, payload=payload or {})
        for handler in list(self._subs.get(topic, ())):
            task = asyncio.create_task(self._run(handler, event), name=f"evt:{topic}")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, handler: Handler, event: Event) -> None:
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — l'isolation est le contrat du bus
            logger.exception("handler failed for topic=%s", event.topic)

    async def wait_for_handlers(self) -> None:
        """Attend la fin de tous les handlers en cours (utile au shutdown)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
