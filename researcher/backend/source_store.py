"Content-addressed source store with SQLite backend and TTL caching."

from __future__ import annotations
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiosqlite

from .config import get_cache_config
from .models import Source, SourceType

logger = logging.getLogger(__name__)

_CREATE_SOURCES = """
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    title TEXT,
    text TEXT,
    url TEXT,
    type TEXT,
    origin TEXT,
    retrieved_at REAL,
    expires_at REAL
);
"""

_CREATE_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS source_embeddings (
    source_id TEXT PRIMARY KEY,
    embedding BLOB
);
"""

_CREATE_IDX = "CREATE INDEX IF NOT EXISTS idx_expires ON sources(expires_at);"


def _db_path() -> Path:
    cfg = get_cache_config()
    path = Path(cfg.get("db_path", "source_store.db"))
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    return path


async def init_store() -> None:
    db = _db_path()
    async with aiosqlite.connect(db) as conn:
        await conn.execute(_CREATE_SOURCES)
        await conn.execute(_CREATE_EMBEDDINGS)
        await conn.execute(_CREATE_IDX)
        await conn.commit()
    logger.info(f"Source store initialised at {db}")


def _content_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


async def upsert_source(source: Source) -> str:
    """Store source; returns its content-addressed id."""
    cfg = get_cache_config()
    ttl_seconds = cfg.get("ttl_days", 30) * 86400
    now = time.time()
    expires = now + ttl_seconds

    src_id = _content_hash(source.text or source.title)
    db = _db_path()
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            """
            INSERT INTO sources (id, title, text, url, type, origin, retrieved_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                origin = excluded.origin,
                expires_at = excluded.expires_at
            """,
            (
                src_id,
                source.title,
                source.text,
                source.url,
                source.type.value,
                json.dumps(source.origin),
                now,
                expires,
            ),
        )
        await conn.commit()
    return src_id


async def get_source(src_id: str) -> Optional[Source]:
    db = _db_path()
    now = time.time()
    async with aiosqlite.connect(db) as conn:
        async with conn.execute(
            "SELECT id, title, text, url, type, origin, retrieved_at FROM sources WHERE id=? AND expires_at>?",
            (src_id, now),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    return Source(
        id=row[0],
        title=row[1],
        text=row[2],
        url=row[3],
        type=SourceType(row[4]),
        origin=json.loads(row[5]),
        retrieved_at=row[6],
    )


async def deduplicate(sources: List[Source]) -> Tuple[List[Source], Set[str]]:
    """Dedup by content hash; merge origins for duplicates. Returns (unique_list, all_ids)."""
    seen: Dict[str, Source] = {}
    for src in sources:
        h = _content_hash(src.text or src.title)
        if h in seen:
            existing = seen[h]
            merged_origin = list(set(existing.origin + src.origin))
            seen[h] = existing.model_copy(update={"origin": merged_origin})
        else:
            seen[h] = src.model_copy(update={"id": h})

    unique = list(seen.values())
    return unique, set(seen.keys())


async def check_db_size_mb() -> float:
    db = _db_path()
    if db.exists():
        return os.path.getsize(db) / (1024 * 1024)
    return 0.0


async def evict_expired() -> int:
    db = _db_path()
    now = time.time()
    async with aiosqlite.connect(db) as conn:
        cur = await conn.execute("DELETE FROM sources WHERE expires_at<?", (now,))
        deleted = cur.rowcount
        await conn.commit()
    if deleted:
        logger.info(f"Evicted {deleted} expired sources")
    return deleted


async def save_embedding(source_id: str, embedding: List[float]) -> None:
    import struct

    blob = struct.pack(f"{len(embedding)}f", *embedding)
    db = _db_path()
    async with aiosqlite.connect(db) as conn:
        await conn.execute(
            "INSERT OR REPLACE INTO source_embeddings (source_id, embedding) VALUES (?, ?)",
            (source_id, blob),
        )
        await conn.commit()


async def load_embedding(source_id: str) -> Optional[List[float]]:
    import struct

    db = _db_path()
    async with aiosqlite.connect(db) as conn:
        async with conn.execute(
            "SELECT embedding FROM source_embeddings WHERE source_id=?", (source_id,)
        ) as cur:
            row = await cur.fetchone()
    if not row:
        return None
    blob = row[0]
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))
