"""
Embedding backends for the RAG system.

Embedding model comparison
──────────────────────────
OpenAI text-embedding-3-small  (default)
  dim=1536, ~62M tokens/$1, strong performance, closed-source.
  When to use: best quality/cost for English-heavy English production apps.

OpenAI text-embedding-3-large
  dim=3072, ~6× cost vs small, ~5 % better MTEB score.
  When to use: high-stakes retrieval (legal, medical) where quality > cost.

BAAI/bge-small-en-v1.5  (HuggingFace, default)
  dim=384, free, runs on CPU, ~33 ms/batch on M1.
  When to use: dev, privacy-sensitive docs, on-prem, budget-conscious.

BAAI/bge-large-en-v1.5
  dim=1024, free, 4× slower than small, top MTEB English rank.
  When to use: highest quality free embeddings.

intfloat/e5-large-v2
  Excellent for asymmetric retrieval (query vs passage).
  Requires "query: " prefix for queries, "passage: " for docs.

Cohere embed-english-v3.0
  dim=1024, strong multilingual, supports compression (int8).
  When to use: multilingual corpora, or if already using Cohere reranker.

Matryoshka Representation Learning (MRL)
  OpenAI 3-small/large support truncated dimensions (e.g., 512 instead of 1536).
  Use dimensions= parameter to reduce storage/cost with minimal quality loss.

Production tips
───────────────
1. Embed queries with the SAME model used for document indexing.
2. Batch embed documents; avoid per-chunk API calls (rate limits + latency).
3. Cache embeddings on disk: re-embedding the same text is wasteful.
4. Normalize embeddings to unit length for cosine → dot product optimization.
5. Monitor embedding drift: if you change models, re-embed the entire index.
6. For multilingual: use LaBSE or multilingual-e5-large.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import EmbedderType, settings
from models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Disk cache (LRU-style via joblib / pickle)
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Simple disk cache for embeddings keyed by (model_name, text_hash).
    In production, replace with Redis for shared caching across workers.
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, model: str, text: str) -> Path:
        h = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
        return self.cache_dir / f"{h}.pkl"

    def get(self, model: str, text: str) -> Optional[list[float]]:
        key = self._key(model, text)
        if key.exists():
            with open(key, "rb") as f:
                return pickle.load(f)
        return None

    def set(self, model: str, text: str, embedding: list[float]) -> None:
        key = self._key(model, text)
        with open(key, "wb") as f:
            pickle.dump(embedding, f)

    def get_batch(self, model: str, texts: list[str]) -> tuple[dict[int, list[float]], list[int]]:
        """Returns cached results and indices of cache misses."""
        cached: dict[int, list[float]] = {}
        misses: list[int] = []
        for i, text in enumerate(texts):
            result = self.get(model, text)
            if result is not None:
                cached[i] = result
            else:
                misses.append(i)
        return cached, misses


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    model_name: str = "unknown"

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache = EmbeddingCache(cache_dir or settings.embedding_cache_dir)

    @abstractmethod
    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts without caching."""
        ...

    def embed(self, text: str) -> list[float]:
        """Embed a single text string with caching."""
        cached = self.cache.get(self.model_name, text)
        if cached is not None:
            return cached
        result = self._embed_raw([text])[0]
        self.cache.set(self.model_name, text, result)
        return result

    def embed_batch(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """
        Embed a list of texts with caching and batching.

        Returns embeddings in the same order as input texts.
        """
        bs = batch_size or settings.embedding_batch_size
        cached_map, miss_indices = self.cache.get_batch(self.model_name, texts)

        # Embed only cache misses in batches
        miss_texts = [texts[i] for i in miss_indices]
        new_embeddings: list[list[float]] = []

        for start in range(0, len(miss_texts), bs):
            batch = miss_texts[start : start + bs]
            logger.debug("Embedding batch %d–%d / %d", start, start + len(batch), len(miss_texts))
            new_embeddings.extend(self._embed_raw(batch))

        # Store new embeddings in cache
        for idx, emb in zip(miss_indices, new_embeddings):
            self.cache.set(self.model_name, texts[idx], emb)
            cached_map[idx] = emb

        return [cached_map[i] for i in range(len(texts))]

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Embed all chunks in-place and return them."""
        texts = [c.content for c in chunks]
        embeddings = self.embed_batch(texts)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query. Override in subclasses that require query prefixes
        (e.g., E5 requires "query: " prefix for asymmetric retrieval).
        """
        return self.embed(query)


# ---------------------------------------------------------------------------
# OpenAI Embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's embedding API.
    Supports Matryoshka dimensions via `dimensions` parameter.

    Production tip: Set OPENAI_API_KEY via env; never hardcode.
    Rate limits: tier-1 = 1M tokens/min. Use exponential backoff on 429.
    """

    def __init__(self, model: str | None = None, dimensions: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model or settings.openai_embedding_model
        self.dimensions = dimensions

        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self.model_name, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# HuggingFace / SentenceTransformers Embedder
# ---------------------------------------------------------------------------

class HuggingFaceEmbedder(BaseEmbedder):
    """
    Local embedding via sentence-transformers. Runs on CPU or GPU.
    Zero API cost; ideal for privacy-sensitive workloads.

    Production tips:
    - Use ONNX runtime for 2–4× CPU speedup: optimum[onnxruntime].
    - Quantize with int8 for ~4× model size reduction (minor quality loss).
    - For GPU: set device="cuda" and use batch_size=128+.
    """

    def __init__(self, model: str | None = None, device: str = "cpu", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model or settings.huggingface_embedding_model

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=device)
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        logger.info("Loaded HuggingFace embedder: %s on %s", self.model_name, device)

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,  # cosine sim → dot product
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """E5 models require 'query: ' prefix; BGE uses instruction."""
        if "e5" in self.model_name.lower():
            query = f"query: {query}"
        elif "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        return self.embed(query)


# ---------------------------------------------------------------------------
# Cohere Embedder
# ---------------------------------------------------------------------------

class CohereEmbedder(BaseEmbedder):
    """
    Cohere embedding API. Supports input_type for asymmetric retrieval.

    Production advantage: supports int8 compression (75 % size reduction).
    Multilingual model available: embed-multilingual-v3.0.
    """

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model or settings.cohere_embedding_model

        try:
            import cohere
            self._client = cohere.Client(settings.cohere_api_key)
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self.model_name,
            input_type="search_document",
        )
        return response.embeddings

    def embed_query(self, query: str) -> list[float]:
        """Cohere uses input_type to distinguish query vs document."""
        cached = self.cache.get(f"{self.model_name}:query", query)
        if cached is not None:
            return cached

        import cohere
        client = cohere.Client(settings.cohere_api_key)
        response = client.embed(
            texts=[query],
            model=self.model_name,
            input_type="search_query",
        )
        emb = response.embeddings[0]
        self.cache.set(f"{self.model_name}:query", query, emb)
        return emb


# ---------------------------------------------------------------------------
# Ollama Embedder  (100 % local, zero API cost)
# ---------------------------------------------------------------------------

class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding via the Ollama daemon using the native ``ollama`` Python SDK.

    Prerequisites
    ─────────────
    1. Install Ollama:  https://ollama.com/download
    2. Pull an embedding model:
         ollama pull nomic-embed-text     # dim=768  ← default
         ollama pull mxbai-embed-large    # dim=1024  stronger
         ollama pull all-minilm           # dim=384   fast/tiny
         ollama pull bge-m3               # dim=1024  multilingual
    3. Set EMBEDDING_DIM in .env to match the model's output dimension.

    How it works
    ────────────
    Ollama exposes ``POST /api/embed`` which accepts a list of strings and
    returns a list of float vectors.  The Python SDK wraps this cleanly.
    No GPU is required — CPU inference works, just slower.

    Dimension map (update EMBEDDING_DIM accordingly)
    ────────────────────────────────────────────────
    nomic-embed-text   →  768
    mxbai-embed-large  →  1024
    all-minilm         →  384
    bge-m3             →  1024
    snowflake-arctic-embed → 1024

    Production tips
    ───────────────
    • Run the Ollama daemon on a GPU node and point OLLAMA_BASE_URL at it for
      throughput matching cloud APIs at zero per-token cost.
    • Set OLLAMA_KEEP_ALIVE="24h" for always-on servers to avoid cold-start
      latency (model stays loaded in VRAM).
    • Set OLLAMA_KEEP_ALIVE="0" in memory-constrained environments to unload
      immediately after each request.
    • Ollama natively batches embed requests — send the full chunk list in one
      call rather than looping (already handled here via _embed_raw).
    """

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model or settings.ollama_embed_model
        self._base_url = settings.ollama_base_url
        self._timeout = settings.ollama_timeout

        try:
            import ollama as _ollama
            self._client = _ollama.Client(
                host=self._base_url,
                timeout=self._timeout,
            )
        except ImportError:
            raise ImportError(
                "Install the Ollama Python SDK:  pip install ollama\n"
                "Then start the daemon:           ollama serve\n"
                "And pull a model:                ollama pull nomic-embed-text"
            )

        logger.info(
            "OllamaEmbedder initialized: model=%s base_url=%s",
            self.model_name, self._base_url,
        )

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts in a single Ollama API call.

        The Ollama /api/embed endpoint accepts ``input`` as either a single
        string or a list of strings; the SDK handles both cases.
        """
        response = self._client.embed(
            model=self.model_name,
            input=texts,
        )
        # SDK ≥0.3: response.embeddings is list[list[float]]
        embeddings = response.embeddings
        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Query embedding.  Ollama embedding models are symmetric (no prefix
        tricks needed), so we just delegate to the standard embed path.
        Subclass and override if you pull an asymmetric model like bge-m3.
        """
        return self.embed(query)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(embedder_type: EmbedderType | None = None) -> BaseEmbedder:
    etype = embedder_type or settings.embedder_type

    if etype == EmbedderType.OPENAI:
        return OpenAIEmbedder()
    elif etype == EmbedderType.HUGGINGFACE:
        return HuggingFaceEmbedder()
    elif etype == EmbedderType.COHERE:
        return CohereEmbedder()
    elif etype == EmbedderType.OLLAMA:
        return OllamaEmbedder()
    else:
        raise ValueError(f"Unknown embedder type: {etype}")
