"Vector store backends for persisting and querying chunk embeddings."

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from config import VectorStoreType, settings
from models import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


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

            score = float(
                np.dot(q, emb) / (np.linalg.norm(q) * np.linalg.norm(emb) + 1e-9)
            )
            scores.append((score, chunk))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            ScoredChunk(chunk=c, score=s, retrieval_method="dense")
            for s, c in scores[:top_k]
        ]

    def delete(self, chunk_ids: list[str]) -> None:
        for cid in chunk_ids:
            self._chunks.pop(cid, None)

    def delete_by_source(self, source: str) -> None:
        to_del = [cid for cid, c in self._chunks.items() if c.source == source]
        for cid in to_del:
            del self._chunks[cid]

    def count(self) -> int:
        return len(self._chunks)


class ChromaVectorStore(BaseVectorStore):
    "Local persistent vector store backed by ChromaDB."

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
            persist_dir,
            collection_name,
            self._collection.count(),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        embeddings = [c.embedding for c in chunks]
        metadatas = []

        for c in chunks:
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

            distance = results["distances"][0][i]
            score = 1.0 - (distance / 2.0)

            chunk = Chunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=meta.pop("doc_id", ""),
                source=meta.pop("source", ""),
                metadata=meta,
                embedding=results["embeddings"][0][i]
                if results.get("embeddings")
                else None,
            )
            scored.append(
                ScoredChunk(chunk=chunk, score=score, retrieval_method="dense")
            )

        return scored

    def delete(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)

    def delete_by_source(self, source: str) -> None:
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        return self._collection.count()


class PgVectorStore(BaseVectorStore):
    "PostgreSQL + pgvector extension for vector storage."

    def __init__(
        self, connection_string: str | None = None, embedding_dim: int | None = None
    ):
        try:
            from sqlalchemy import create_engine, text, Column, String, JSON, Integer
            from sqlalchemy.orm import sessionmaker, DeclarativeBase
            from pgvector.sqlalchemy import Vector
        except ImportError:
            raise ImportError(
                "Install: pip install pgvector psycopg[binary] sqlalchemy"
            )

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
            conn.execute(
                text(f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    doc_id      TEXT NOT NULL,
                    source      TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    metadata    JSONB DEFAULT '{{}}',
                    embedding   vector({self._dim})
                )
            """)
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS rag_chunks_source_idx ON rag_chunks (source)"
                )
            )
            conn.commit()
        logger.info("pgvector table ensured (dim=%d)", self._dim)

    def upsert(self, chunks: list[Chunk]) -> None:
        from sqlalchemy import text

        with self._Session() as session:
            for c in chunks:
                session.execute(
                    text("""
                    INSERT INTO rag_chunks (chunk_id, doc_id, source, content, metadata, embedding)
                    VALUES (:chunk_id, :doc_id, :source, :content, :metadata, :embedding)
                    ON CONFLICT (chunk_id) DO UPDATE
                        SET content=EXCLUDED.content,
                            embedding=EXCLUDED.embedding,
                            metadata=EXCLUDED.metadata
                """),
                    {
                        "chunk_id": c.chunk_id,
                        "doc_id": c.doc_id,
                        "source": c.source,
                        "content": c.content,
                        "metadata": c.metadata,
                        "embedding": c.embedding,
                    },
                )
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
            session.execute(
                text("DELETE FROM rag_chunks WHERE chunk_id = ANY(:ids)"),
                {"ids": chunk_ids},
            )
            session.commit()

    def delete_by_source(self, source: str) -> None:
        from sqlalchemy import text

        with self._Session() as session:
            session.execute(
                text("DELETE FROM rag_chunks WHERE source = :source"),
                {"source": source},
            )
            session.commit()

    def count(self) -> int:
        from sqlalchemy import text

        with self._Session() as session:
            return session.execute(text("SELECT COUNT(*) FROM rag_chunks")).scalar()


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
