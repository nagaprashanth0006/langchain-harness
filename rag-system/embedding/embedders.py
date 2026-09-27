"Embedding backends for the RAG system."

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

    def get_batch(
        self, model: str, texts: list[str]
    ) -> tuple[dict[int, list[float]], list[int]]:
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

    def embed_batch(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """
        Embed a list of texts with caching and batching.

        Returns embeddings in the same order as input texts.
        """
        bs = batch_size or settings.embedding_batch_size
        cached_map, miss_indices = self.cache.get_batch(self.model_name, texts)

        miss_texts = [texts[i] for i in miss_indices]
        new_embeddings: list[list[float]] = []

        for start in range(0, len(miss_texts), bs):
            batch = miss_texts[start : start + bs]
            logger.debug(
                "Embedding batch %d–%d / %d", start, start + len(batch), len(miss_texts)
            )
            new_embeddings.extend(self._embed_raw(batch))

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


class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's embedding API.
    Supports Matryoshka dimensions via `dimensions` parameter.

    Production tip: Set OPENAI_API_KEY via env; never hardcode.
    Rate limits: tier-1 = 1M tokens/min. Use exponential backoff on 429.
    """

    def __init__(
        self, model: str | None = None, dimensions: int | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model or settings.openai_embedding_model
        self.dimensions = dimensions

        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        kwargs: dict = {"model": self.model_name, "input": texts}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]


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
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            )

        logger.info("Loaded HuggingFace embedder: %s on %s", self.model_name, device)

    def _embed_raw(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
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


class OllamaEmbedder(BaseEmbedder):
    "Local embedding via the Ollama daemon using the native ``ollama`` Python SDK."

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
            self.model_name,
            self._base_url,
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
