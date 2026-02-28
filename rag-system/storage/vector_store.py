"""
Vector store backends for persisting and querying chunk embeddings.

Vector store comparison
────────────────────────
ChromaDB (default)
  Local, embedded, zero-infra. Persists to disk via DuckDB+Parquet.
  Supports metadata filtering. Good up to ~1M vectors on a single node.
  Caveat: no horizontal scaling; single writer. Use for ≤1M chunk corpora.

pgvector (PostgreSQL extension)
  SQL-native vector storage; ACID, joins, standard tooling.
  IVFFlat index: fast approximate search. HNSW index: better recall (v0.5+).
  Best when: you already run Postgres and want one fewer service.
  Caveat: must tune index lists/probes parameters for your dataset size.

Pinecone
  Fully managed, scales to billions of vectors, multi-tenant namespaces.
  Supports metadata filtering, hybrid search natively.
  Best when: large corpus, multi-tenant SaaS, don't want ops burden.

Weaviate
  Open-source, supports hybrid (BM25 + vector), GraphQL API.
  Good for: knowledge graphs, cross-referencing object types.

Qdrant
  High-performance Rust-based store; excellent filtering, payload indexing.
  On-premise or managed cloud. Strong HNSW + quantization support.

Milvus / Zilliz
  Enterprise-grade, columnar storage, GPU indexing.
  Best when: >100M vectors, need GPU-accelerated search.

Redis (with RediSearch / RedisVSS)
  In-memory; lowest latency. Use when sub-millisecond search is required.
  Caveat: cost of RAM at scale; persistence is secondary.

Production indexing tips
────────────────────────
1. Use HNSW index for best recall/latency trade-off (>100K vectors).
2. IVFFlat is faster to build but lower recall; tune nlist = sqrt(N).
3. Store rich metadata (source, page, date, doc_id) — used for filtering.
4. Separate collections per tenant/namespace to avoid cross-contamination.
5. Build indexes offline (batch ingest) before switching traffic.
6. Monitor p99 query latency; alert at >200 ms for interactive RAG.
7. Run periodic compaction (Chroma / Milvus) to reclaim disk space.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from config import VectorStoreType, settings
from models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):

    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> None:
        """Store chunks with their embeddings. Idempotent on chunk_id."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        """Return top-k chunks by cosine similarity to query_embedding."""
        ...

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        ...

    @abstractmethod
    def delete_by_source(self, source: str) -> None:
        """Remove all chunks originating from a given source file."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of stored chunks."""
        ...

    def upsert_and_query(
        self, chunks: list[Chunk], query_embedding: list[float], top_k: int = 10
    ) -> list[ScoredChunk]:
        self.upsert(chunks)
        return self.query(query_embedding, top_k=top_k)


# ---------------------------------------------------------------------------
# In-Memory (testing / CI)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(BaseVectorStore):
    """
    Brute-force cosine search over an in-memory list.
    O(N) per query — only for testing.
    """

    def __init__(self):
        self._chunks: dict[str, Chunk] = {}

    def upsert(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            if c.embedding is None:
                raise ValueError(f"Chunk {c.chunk_id} has no embedding")
            self._chunks[c.chunk_id] = c

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        import numpy as np

        q = np.array(query_embedding)
        scores: list[tuple[float, Chunk]] = []

        for chunk in self._chunks.values():
            if filter:
                if not all(chunk.metadata.get(k) == v for k, v in filter.items()):
                    continue
            emb = np.array(chunk.embedding)
            # Cosine similarity (embeddings assumed normalized)
            score = float(np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-9))
            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [ScoredChunk(chunk=c, score=s, retrieval_method="dense") for s, c in scores[:top_k]]

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

    def delete_by_source(self, source: str) -> None:
        to_del = [cid for cid, c in self._chunks.items() if c.source == source]
        for cid in to_del:
            del self._chunks[cid]

    def count(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

class ChromaVectorStore(BaseVectorStore):
    """
    Local persistent vector store backed by ChromaDB.

    ChromaDB uses DuckDB+Parquet for persistence and an HNSW index (hnswlib).
    Supports metadata filtering with $eq/$in/$gte operators.

    Production scaling limits:
    - ~1M vectors before query latency degrades (single node).
    - Single writer; reads can be concurrent.
    - Upgrade path: swap to Qdrant/Weaviate for >1M vectors.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_dim: int | None = None,
    ):
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError("Install chromadb: pip install chromadb")

        persist_dir = str(persist_dir or settings.chroma_persist_dir)
        collection_name = collection_name or settings.chroma_collection_name

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized: dir=%s collection=%s count=%d",
            persist_dir, collection_name, self._collection.count(),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = []

        for c in chunks:
            # Chroma metadata values must be str/int/float/bool
            meta = {
                k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in c.metadata.items()
            }
            meta["doc_id"] = c.doc_id
            meta["source"] = c.source
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks to ChromaDB", len(chunks))

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        where = filter or None
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        scored: list[ScoredChunk] = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            content = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            # Chroma returns cosine distance [0, 2]; convert to similarity
            distance = results["distances"][0][i]
            score = 1.0 - (distance / 2.0)

            chunk = Chunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=meta.pop("doc_id", ""),
                source=meta.pop("source", ""),
                metadata=meta,
                embedding=results["embeddings"][0][i] if results.get("embeddings") else None,
            )
            scored.append(ScoredChunk(chunk=chunk, score=score, retrieval_method="dense"))

        return scored

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# pgvector (PostgreSQL)
# ---------------------------------------------------------------------------

class PgVectorStore(BaseVectorStore):
    """
    PostgreSQL + pgvector extension for vector storage.

    Setup:
        CREATE EXTENSION IF NOT EXISTS vector;
        (The table is auto-created on first use via SQLAlchemy.)

    Index options:
        IVFFlat: CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
                 WITH (lists = 100);  -- tune lists = sqrt(row_count)
        HNSW:    CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
                 WITH (m = 16, ef_construction = 64);  -- better recall, slower build

    Production caveat: run VACUUM ANALYZE after bulk inserts to update planner stats.
    """

    def __init__(self, connection_string: str | None = None, embedding_dim: int | None = None):
        try:
            from sqlalchemy import create_engine, text, Column, String, JSON, Integer
            from sqlalchemy.orm import sessionmaker, DeclarativeBase
            from pgvector.sqlalchemy import Vector
        except ImportError:
            raise ImportError("Install: pip install pgvector psycopg[binary] sqlalchemy")

        conn_str = connection_string or settings.pgvector_connection_string
        dim = embedding_dim or settings.embedding_dim

        self._engine = create_engine(conn_str, pool_pre_ping=True)
        self._Session = sessionmaker(bind=self._engine)
        self._dim = dim
        self._ensure_table()

    def _ensure_table(self) -> None:
        from sqlalchemy import text
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    JSONB DEFAULT '{{}}',
                    embedding   vector({self._dim})
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS rag_chunks_source_idx ON rag_chunks (source)"
            ))
            conn.commit()
        logger.info("pgvector table ensured (dim=%d)", self._dim)

    def upsert(self, chunks: list[Chunk]) -> None:
        from sqlalchemy import text
        with self._Session() as session:
            for c in chunks:
                session.execute(text("""
                    INSERT INTO rag_chunks (chunk_id, doc_id, source, content, metadata, embedding)
                    VALUES (:chunk_id, :doc_id, :source, :content, :metadata, :embedding)
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET content=EXCLUDED.content,
                            embedding=EXCLUDED.embedding,
                            metadata=EXCLUDED.metadata
                """), {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "source": c.source,
                    "content": c.content,
                    "metadata": c.metadata,
                    "embedding": c.embedding,
                })
            session.commit()

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        from sqlalchemy import text

        where_clause = ""
        params: dict[str, Any] = {"embedding": str(query_embedding), "limit": top_k}
        if filter:
            conditions = []
            for i, (k, v) in enumerate(filter.items()):
                param_name = f"filter_{i}"
                conditions.append(f"metadata->>'{k}' = :{param_name}")
                params[param_name] = str(v)
            where_clause = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT chunk_id, doc_id, source, content, metadata,
                   1 - (embedding <=> :embedding::vector) AS score
            FROM rag_chunks
            {where_clause}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :limit
        """
        with self._Session() as session:
            rows = session.execute(text(sql), params).fetchall()

        return [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    source=row.source,
                    content=row.content,
                    metadata=dict(row.metadata or {}),
                ),
                score=float(row.score),
                retrieval_method="dense",
            )
            for row in rows
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        from sqlalchemy import text
        with self._Session() as session:
            session.execute(text("DELETE FROM rag_chunks WHERE chunk_id = ANY(:ids)"), {"ids": chunk_ids})
            session.commit()

    def delete_by_source(self, source: str) -> None:
        from sqlalchemy import text
        with self._Session() as session:
            session.execute(text("DELETE FROM rag_chunks WHERE source = :source"), {"source": source})
            session.commit()

    def count(self) -> int:
        from sqlalchemy import text
        with self._Session() as session:
            return session.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store(store_type: VectorStoreType | None = None) -> BaseVectorStore:
    stype = store_type or settings.vector_store_type

    if stype == VectorStoreType.CHROMA:
        return ChromaVectorStore()
    elif stype == VectorStoreType.PGVECTOR:
        return PgVectorStore()
    elif stype == VectorStoreType.IN_MEMORY:
        return InMemoryVectorStore()
    elif stype == VectorStoreType.PINECONE:
        raise NotImplementedError(
            "Pinecone backend: install pinecone-client and implement PineconeVectorStore. "
            "See: https://docs.pinecone.io/reference/python-sdk"
        )
    else:
        raise ValueError(f"Unknown vector store type: {stype}")
