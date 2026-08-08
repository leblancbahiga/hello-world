"""Stockage SQLite (WAL) : connexion, migrations et DAO minimal.

Choix : sqlite3 stdlib, un seul fichier, FTS5 pour la recherche texte à venir
(P1). Pas d'ORM — des requêtes simples et explicites.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("nuru.storage")

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Nouvelle conversation',
    model       TEXT,
    provider    TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    tokens          INTEGER,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""

_MIGRATIONS: list[str] = [_SCHEMA_V1]

_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    collection  TEXT NOT NULL DEFAULT 'default',
    name        TEXT NOT NULL,
    path        TEXT,
    mime        TEXT,
    size        INTEGER,
    hash        TEXT,
    status      TEXT NOT NULL DEFAULT 'pending',
    error       TEXT,
    chunks      INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    tokens      INTEGER NOT NULL,
    embedding   BLOB,
    page        INTEGER,
    meta        TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

_MIGRATIONS = [_SCHEMA_V1, _SCHEMA_V2]


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


class Storage:
    """Connexion SQLite unique, thread-safe, avec migrations par version."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA cache_size=-512")  # ~512 Ko de cache
            self._migrate()
        logger.info("storage ready at %s", self.path)

    def close(self) -> None:
        if self._conn is not None:
            with self._lock:
                self._conn.close()
            self._conn = None

    def _migrate(self) -> None:
        conn = self._conn
        assert conn is not None
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for idx in range(version, len(_MIGRATIONS)):
            conn.executescript(_MIGRATIONS[idx])
            conn.execute(f"PRAGMA user_version={idx + 1}")
            conn.commit()
            logger.info("migration applied: v%d", idx + 1)

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        conn = self._conn
        assert conn is not None
        with self._lock:
            cur = conn.execute(sql, params)
            conn.commit()
            return cur

    # ------------------------------------------------------------- conversations

    def create_conversation(self, model: str | None = None,
                            provider: str | None = None,
                            title: str | None = None) -> dict:
        conv_id = uuid.uuid4().hex[:16]
        now = time.time()
        self._execute(
            "INSERT INTO conversations (id, title, model, provider, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, title or "Nouvelle conversation", model, provider, now, now),
        )
        return self.get_conversation(conv_id)

    def get_conversation(self, conv_id: str) -> dict | None:
        cur = self._execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        return _row_to_dict(cur.fetchone())

    def list_conversations(self, limit: int = 100) -> list[dict]:
        cur = self._execute(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def rename_conversation(self, conv_id: str, title: str) -> None:
        self._execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, time.time(), conv_id),
        )

    def touch_conversation(self, conv_id: str) -> None:
        self._execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (time.time(), conv_id),
        )

    def delete_conversation(self, conv_id: str) -> None:
        self._execute("DELETE FROM conversations WHERE id = ?", (conv_id,))

    # ----------------------------------------------------------------- messages

    def add_message(self, conv_id: str, role: str, content: str,
                    tokens: int | None = None) -> dict:
        msg_id = uuid.uuid4().hex[:16]
        now = time.time()
        self._execute(
            "INSERT INTO messages (id, conversation_id, role, content, tokens, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, tokens, now),
        )
        self._execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )
        return {"id": msg_id, "conversation_id": conv_id, "role": role,
                "content": content, "tokens": tokens, "created_at": now}

    def list_messages(self, conv_id: str, limit: int = 200) -> list[dict]:
        cur = self._execute(
            "SELECT * FROM messages WHERE conversation_id = ?"
            " ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (conv_id, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    # ---------------------------------------------------------------- documents

    def upsert_document(self, doc: dict) -> dict:
        now = time.time()
        self._execute(
            "INSERT INTO documents"
            " (id, collection, name, path, mime, size, hash, status, error, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " collection=excluded.collection, name=excluded.name, path=excluded.path,"
            " mime=excluded.mime, size=excluded.size, hash=excluded.hash,"
            " status=excluded.status, updated_at=excluded.updated_at",
            (doc["id"], doc.get("collection", "default"), doc["name"], doc.get("path"),
             doc.get("mime"), doc.get("size"), doc.get("hash"), doc.get("status", "pending"),
             now, now),
        )
        return self.get_document(doc["id"])

    def get_document(self, doc_id: str) -> dict | None:
        cur = self._execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        return _row_to_dict(cur.fetchone())

    def find_document_by_hash(self, file_hash: str) -> dict | None:
        cur = self._execute(
            "SELECT * FROM documents WHERE hash = ? AND status = 'ready' LIMIT 1",
            (file_hash,),
        )
        return _row_to_dict(cur.fetchone())

    def find_document_by_name(self, name: str, collection: str = "default") -> dict | None:
        cur = self._execute(
            "SELECT * FROM documents WHERE name = ? AND collection = ? LIMIT 1",
            (name, collection),
        )
        return _row_to_dict(cur.fetchone())

    def list_documents(self, collection: str | None = None, limit: int = 200) -> list[dict]:
        if collection:
            cur = self._execute(
                "SELECT * FROM documents WHERE collection = ? ORDER BY updated_at DESC LIMIT ?",
                (collection, limit),
            )
        else:
            cur = self._execute(
                "SELECT * FROM documents ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def update_document_status(self, doc_id: str, status: str,
                               error: str | None = None, chunks: int | None = None) -> None:
        self._execute(
            "UPDATE documents SET status = ?, error = ?, updated_at = ?,"
            " chunks = COALESCE(?, chunks)"
            " WHERE id = ?",
            (status, error, time.time(), chunks, doc_id),
        )

    def delete_document(self, doc_id: str) -> None:
        self._execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # -------------------------------------------------------------------- chunks

    def add_chunks(self, chunks: list[dict]) -> None:
        conn = self._conn
        assert conn is not None
        now = time.time()
        with self._lock:
            conn.executemany(
                "INSERT INTO chunks"
                " (id, document_id, chunk_index, content, tokens, embedding, page, meta, created_at)"
                " VALUES (:id, :document_id, :chunk_index, :content, :tokens, :embedding, :page, :meta, :created_at)",
                [
                    {
                        "id": chunk["id"],
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "tokens": chunk["tokens"],
                        "embedding": chunk.get("embedding"),
                        "page": chunk.get("page"),
                        "meta": chunk.get("meta"),
                        "created_at": now,
                    }
                    for chunk in chunks
                ],
            )
            conn.commit()

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        cur = self._execute(
            "SELECT id, chunk_index, content, page, meta FROM chunks"
            " WHERE document_id = ? ORDER BY chunk_index ASC",
            (doc_id,),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def iter_chunk_vectors(self) -> list[dict]:
        """Toutes les embeddings, pour la recherche vectorielle en RAM."""
        cur = self._execute(
            "SELECT c.id, c.document_id, c.chunk_index, c.content, c.page, c.embedding"
            " FROM chunks c WHERE c.embedding IS NOT NULL"
        )
        return [_row_to_dict(r) for r in cur.fetchall()]

    def fts_search(self, query: str, limit: int = 50) -> list[dict]:
        """Recherche BM25 via FTS5, retourne les chunks correspondants."""
        cur = self._execute(
            "SELECT c.id, c.document_id, c.chunk_index, c.content, c.page, c.meta,"
            " bm25(chunks_fts) AS score"
            " FROM chunks_fts"
            " JOIN chunks c ON c.rowid = chunks_fts.rowid"
            " WHERE chunks_fts MATCH ?"
            " ORDER BY score LIMIT ?",
            (query, limit),
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
