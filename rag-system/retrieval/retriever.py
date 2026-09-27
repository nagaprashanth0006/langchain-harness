"Retrieval strategies for the RAG pipeline."

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Optional

from config import RetrieverType, settings
from embedding.embedders import BaseEmbedder, get_embedder
from models import Chunk, ScoredChunk
from storage.vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        """Retrieve top-k chunks for a query."""
        ...


class DenseRetriever(BaseRetriever):
    """Pure semantic retrieval via embedding similarity."""

    def __init__(self, vector_store: BaseVectorStore, embedder: BaseEmbedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        k = top_k or settings.top_k_retrieval
        query_emb = self.embedder.embed_query(query)
        results = self.vector_store.query(query_emb, top_k=k, filter=filter)
        for r in results:
            r.retrieval_method = "dense"
        logger.debug(
            "Dense retrieval: %d results for query='%.60s'", len(results), query
        )
        return results


class BM25Retriever(BaseRetriever):
    "Keyword-based BM25 retrieval over an in-memory corpus."

    def __init__(self, chunks: list[Chunk] | None = None):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

        self._BM25Okapi = BM25Okapi
        self._chunks: list[Chunk] = []
        self._index = None

        if chunks:
            self.index(chunks)

    def index(self, chunks: list[Chunk]) -> None:
        """Build or rebuild the BM25 index from a list of chunks."""
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized = [self._tokenize(c.content) for c in chunks]
        self._index = BM25Okapi(tokenized)
        logger.info("BM25 index built with %d documents", len(chunks))

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        """Incrementally add chunks and rebuild index (BM25Okapi is immutable)."""
        self.index(self._chunks + new_chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenizer. Replace with NLTK for better perf."""
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        if self._index is None or not self._chunks:
            logger.warning("BM25 index is empty; did you call index()?")
            return []

        k = top_k or settings.top_k_retrieval
        tokenized_query = self._tokenize(query)
        scores = self._index.get_scores(tokenized_query)

        valid_indices = range(len(self._chunks))
        if filter:
            valid_indices = [
                i
                for i in valid_indices
                if all(
                    self._chunks[i].metadata.get(fk) == fv for fk, fv in filter.items()
                )
            ]

        scored_valid = sorted(
            [(i, float(scores[i])) for i in valid_indices],
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        results = [
            ScoredChunk(chunk=self._chunks[i], score=s, retrieval_method="sparse")
            for i, s in scored_valid
            if s > 0
        ]
        logger.debug(
            "BM25 retrieval: %d results for query='%.60s'", len(results), query
        )
        return results


class HybridRetriever(BaseRetriever):
    "Combines dense + sparse results via Reciprocal Rank Fusion (RRF)."

    RRF_K = 60

    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        alpha: float | None = None,
    ):
        self.dense = dense
        self.sparse = sparse
        self.alpha = alpha if alpha is not None else settings.hybrid_alpha

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        k = top_k or settings.top_k_retrieval
        fetch_k = k * 2

        dense_results = self.dense.retrieve(query, top_k=fetch_k, filter=filter)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k, filter=filter)

        fused = self._rrf(dense_results, sparse_results)
        logger.debug(
            "Hybrid retrieval: dense=%d sparse=%d fused=%d",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return fused[:k]

    def _rrf(
        self,
        dense_results: list[ScoredChunk],
        sparse_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        """Reciprocal Rank Fusion over two result lists."""
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_map: dict[str, Chunk] = {}

        for rank, sc in enumerate(dense_results):
            cid = sc.chunk.chunk_id
            rrf_scores[cid] += self.alpha * (1.0 / (self.RRF_K + rank + 1))
            chunk_map[cid] = sc.chunk

        for rank, sc in enumerate(sparse_results):
            cid = sc.chunk.chunk_id
            rrf_scores[cid] += (1.0 - self.alpha) * (1.0 / (self.RRF_K + rank + 1))
            chunk_map[cid] = sc.chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [
            ScoredChunk(
                chunk=chunk_map[cid], score=rrf_scores[cid], retrieval_method="hybrid"
            )
            for cid in sorted_ids
        ]


class HyDERetriever(BaseRetriever):
    "Hypothetical Document Embeddings (Gao et al. 2022)."

    def __init__(self, dense: DenseRetriever, generator_fn):
        """
        Args:
            dense: A DenseRetriever instance.
            generator_fn: Callable(query: str) -> str that generates a hypothesis.
        """
        self.dense = dense
        self.generator_fn = generator_fn

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[ScoredChunk]:
        hypothesis = self.generator_fn(query)
        logger.debug("HyDE hypothesis: %.100s", hypothesis)

        return self.dense.retrieve(hypothesis, top_k=top_k, filter=filter)


def get_retriever(
    retriever_type: RetrieverType | None = None,
    vector_store: BaseVectorStore | None = None,
    embedder: BaseEmbedder | None = None,
    bm25_chunks: list[Chunk] | None = None,
) -> BaseRetriever:
    """
    Build and return the configured retriever.

    Args:
        bm25_chunks: Required for sparse/hybrid — chunks to build BM25 index from.
    """
    rtype = retriever_type or settings.retriever_type
    vs = vector_store or get_vector_store()
    emb = embedder or get_embedder()

    if rtype == RetrieverType.DENSE:
        return DenseRetriever(vs, emb)

    elif rtype == RetrieverType.SPARSE:
        retriever = BM25Retriever()
        if bm25_chunks:
            retriever.index(bm25_chunks)
        return retriever

    elif rtype == RetrieverType.HYBRID:
        dense = DenseRetriever(vs, emb)
        sparse = BM25Retriever(bm25_chunks or [])
        return HybridRetriever(dense, sparse)

    else:
        raise ValueError(f"Unknown retriever type: {rtype}")
