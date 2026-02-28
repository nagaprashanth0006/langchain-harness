"""
Retrieval strategies for the RAG pipeline.

Dense retrieval
───────────────
Embeds the query and does approximate nearest-neighbor search in the vector
store. Captures semantic meaning; good at paraphrase matching.
Weakness: fails on exact keyword queries ("error code 0x80070005"), rare terms,
and acronyms the embedding model hasn't seen.

Sparse retrieval (BM25)
───────────────────────
Classic IR algorithm scoring term frequency × inverse document frequency.
Exact keyword match; handles rare terms, IDs, and codes perfectly.
Weakness: no semantic understanding; misses paraphrases.

Hybrid retrieval (Reciprocal Rank Fusion — default)
──────────────────────────────────────────────────────
Combines dense + sparse results using RRF:
  score(d, queries) = Σ  1 / (k + rank_i(d))
                       i
where k=60 is a smoothing constant (Cormack et al. 2009).
RRF is robust to score scale differences; no calibration needed.
Consistently outperforms both pure methods by 5–15 % NDCG in benchmarks.

Alternative fusion strategies
──────────────────────────────
• Linear combination: α·dense_score + (1-α)·sparse_score
  Requires calibrating α; sensitive to score distribution differences.
• CombMNZ: (dense+sparse) × count(systems that retrieved doc)
  Simple, effective, used in TREC evaluations.
• LambdaMart / RankLib: learned ranking; requires labeled training data.

Query expansion
───────────────
For sparse retrieval, query expansion improves recall:
• HyDE (Hypothetical Document Embeddings): generate a hypothetical answer,
  embed it, use that embedding for dense retrieval. Works well for open-QA.
• PRF (Pseudo Relevance Feedback): expand query with terms from top-k docs.
• Synonym expansion via WordNet or domain ontology.

Metadata filtering
──────────────────
Always expose metadata filter params at the API level:
  e.g., filter={"file_name": "annual_report_2024.pdf", "page": {"$gte": 10}}
This lets users scope retrieval without scanning the entire index.
"""

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


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dense retriever
# ---------------------------------------------------------------------------

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
        logger.debug("Dense retrieval: %d results for query='%.60s'", len(results), query)
        return results


# ---------------------------------------------------------------------------
# Sparse retriever (BM25)
# ---------------------------------------------------------------------------

class BM25Retriever(BaseRetriever):
    """
    Keyword-based BM25 retrieval over an in-memory corpus.

    Production caveat: this implementation holds the entire corpus in RAM.
    For large corpora, use Elasticsearch / OpenSearch which maintain a
    persistent inverted index. They also support more advanced features:
    - Fuzzy matching (edit distance)
    - Phrase queries
    - Field boosting
    - Numeric range filters on metadata

    Alternative sparse approaches:
    - SPLADE: learned sparse model; bridges gap between BM25 and dense.
      Produces sparse embedding vectors in vocab space; impressive recall.
    - uniCOIL: contextualized single-token importance weighting.
    - Pinecone Sparse-Dense: native hybrid support in managed service.
    """

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

        # Apply metadata filter manually (since BM25 has no native filter)
        valid_indices = range(len(self._chunks))
        if filter:
            valid_indices = [
                i for i in valid_indices
                if all(self._chunks[i].metadata.get(fk) == fv for fk, fv in filter.items())
            ]

        # Sort by BM25 score
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
        logger.debug("BM25 retrieval: %d results for query='%.60s'", len(results), query)
        return results


# ---------------------------------------------------------------------------
# Hybrid retriever with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Combines dense + sparse results via Reciprocal Rank Fusion (RRF).

    RRF formula:
        score(d) = Σ  1 / (k + rank_i(d))
                    i ∈ {dense, sparse}

    k=60 (Cormack 2009 default) smooths rank differences.

    Why RRF works:
    - Scale-invariant: no need to normalize cosine vs BM25 scores.
    - Rank-based: outlier scores from one system don't dominate.
    - Simple: no learned parameters (unlike linear interpolation).

    Production upgrade path:
    - Pinecone, Weaviate, Elasticsearch all support native hybrid search.
    - ElasticSearch kNN + BM25 with RRF: can be done server-side without
      fetching all results to Python.
    """

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
        fetch_k = k * 2  # Fetch more to have headroom for fusion

        dense_results = self.dense.retrieve(query, top_k=fetch_k, filter=filter)
        sparse_results = self.sparse.retrieve(query, top_k=fetch_k, filter=filter)

        fused = self._rrf(dense_results, sparse_results)
        logger.debug(
            "Hybrid retrieval: dense=%d sparse=%d fused=%d",
            len(dense_results), len(sparse_results), len(fused),
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
            ScoredChunk(chunk=chunk_map[cid], score=rrf_scores[cid], retrieval_method="hybrid")
            for cid in sorted_ids
        ]


# ---------------------------------------------------------------------------
# HyDE retriever (Hypothetical Document Embeddings)
# ---------------------------------------------------------------------------

class HyDERetriever(BaseRetriever):
    """
    Hypothetical Document Embeddings (Gao et al. 2022).

    Instead of embedding the query directly, generate a hypothetical answer
    to the query, then embed that answer. The hypothesis is semantically
    closer to target documents than the query itself.

    Production caveat:
    - Adds one LLM call per query (~50–200 ms extra latency).
    - Can hallucinate; the hypothesis is used ONLY for retrieval, not shown to user.
    - Works best for open-domain QA; less benefit for keyword-heavy queries.
    """

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
        # Retrieve using hypothesis embedding
        return self.dense.retrieve(hypothesis, top_k=top_k, filter=filter)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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
