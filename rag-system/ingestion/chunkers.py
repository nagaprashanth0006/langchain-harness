"Chunking strategies for the RAG ingestion pipeline."

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import Iterator

import nltk
import tiktoken

from config import ChunkerType, settings
from models import Chunk, Document

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=4)
def _get_encoder(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    "Return a cached tiktoken encoder."
    return tiktoken.get_encoding(encoding_name)


def _token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens using the cached OpenAI tokenizer (cl100k == gpt-4 / embeddings)."""
    return len(_get_encoder(encoding_name).encode(text))


def _ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


class BaseChunker(ABC):
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]: ...

    def _make_chunk(
        self, text: str, doc: Document, index: int, extra_meta: dict | None = None
    ) -> Chunk:
        meta = {
            **doc.metadata,
            "chunk_index": index,
            "chunk_size_chars": len(text),
            "chunk_tokens": _token_count(text),
        }
        if extra_meta:
            meta.update(extra_meta)
        return Chunk(
            content=text,
            doc_id=doc.doc_id,
            source=doc.source,
            metadata=meta,
        )


class RecursiveChunker(BaseChunker):
    """
    Splits on paragraph → sentence → word boundaries in priority order.
    Based on LangChain's RecursiveCharacterTextSplitter logic; vendored here
    to avoid a heavy LangChain dependency (only needs langchain-text-splitters).

    Production alternative: use LangChain's version directly if you already
    have the dependency. They maintain edge-case fixes actively.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def split(self, document: Document) -> list[Chunk]:

        char_limit = self.chunk_size * 4
        overlap_chars = self.chunk_overlap * 4

        texts = self._split_text(document.content, char_limit, overlap_chars)
        return [self._make_chunk(t, document, i) for i, t in enumerate(texts)]

    def _split_text(self, text: str, limit: int, overlap: int) -> list[str]:
        """Recursively split using the separator hierarchy."""
        for sep in self._SEPARATORS:
            if sep == "" or sep in text:
                splits = text.split(sep) if sep else list(text)
                return self._merge_splits(splits, sep, limit, overlap)
        return [text]

    def _merge_splits(
        self, splits: list[str], sep: str, limit: int, overlap: int
    ) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for split in splits:
            split_len = len(split) + len(sep)
            if current_len + split_len > limit and current_parts:
                chunk_text = sep.join(current_parts).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) + len(sep) > overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(sep)
                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(split)
            current_len += split_len

        if current_parts:
            chunk_text = sep.join(current_parts).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks


class SemanticChunker(BaseChunker):
    """
    Groups consecutive sentences by embedding similarity.
    A new chunk starts when cosine similarity drops below the threshold.

    Production caveat: embeds ALL sentences during ingestion — 2–5× slower
    than RecursiveChunker. Cache embeddings if you re-ingest frequently.

    Alternative: Use LangChain's SemanticChunker (same idea, more options).
    """

    def __init__(
        self, chunk_size: int, chunk_overlap: int, breakpoint_threshold: float = 0.4
    ):
        super().__init__(chunk_size, chunk_overlap)
        self.breakpoint_threshold = breakpoint_threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return self._model

    def split(self, document: Document) -> list[Chunk]:
        import numpy as np

        _ensure_nltk_punkt()
        sentences = nltk.sent_tokenize(document.content)
        if len(sentences) <= 1:
            return [self._make_chunk(document.content, document, 0)]

        model = self._get_model()
        embeddings = model.encode(sentences, normalize_embeddings=True)

        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        split_indices = [
            i + 1
            for i, sim in enumerate(similarities)
            if sim < (1 - self.breakpoint_threshold)
        ]

        chunks: list[Chunk] = []
        prev = 0
        for idx in [*split_indices, len(sentences)]:
            group = " ".join(sentences[prev:idx]).strip()
            if group:
                chunks.append(self._make_chunk(group, document, len(chunks)))
            prev = idx

        return chunks


class SlidingWindowChunker(BaseChunker):
    """
    Produces fixed-size chunks with overlap by sliding a window character by
    character (actually word by word for cleaner boundaries).

    Best for: legal contracts, dense technical specs where any passage may
    be queried.
    Caveat: high redundancy → larger index, more embedding API cost.
    """

    def split(self, document: Document) -> list[Chunk]:
        words = document.content.split()
        char_limit = self.chunk_size * 4
        overlap_words = max(1, self.chunk_overlap * 4 // 6)

        chunks: list[Chunk] = []
        start = 0
        while start < len(words):
            end = start
            current_chars = 0
            while (
                end < len(words) and current_chars + len(words[end]) + 1 <= char_limit
            ):
                current_chars += len(words[end]) + 1
                end += 1

            text = " ".join(words[start:end]).strip()
            if text:
                chunks.append(self._make_chunk(text, document, len(chunks)))

            if end >= len(words):
                break
            start = max(start + 1, end - overlap_words)

        return chunks


class SentenceChunker(BaseChunker):
    """
    Groups sentences until the token limit is reached, then starts a new chunk.
    Preserves sentence boundaries perfectly.

    Best for: FAQ documents, short factual paragraphs.
    Caveat: chunk sizes vary widely; very short answers may produce tiny chunks.
    """

    def split(self, document: Document) -> list[Chunk]:
        _ensure_nltk_punkt()
        sentences = nltk.sent_tokenize(document.content)

        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _token_count(sent)
            if current_tokens + sent_tokens > self.chunk_size and current:
                chunks.append(
                    self._make_chunk(" ".join(current), document, len(chunks))
                )

                overlap_sents: list[str] = []
                overlap_tokens = 0
                for s in reversed(current):
                    t = _token_count(s)
                    if overlap_tokens + t > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                current = overlap_sents
                current_tokens = overlap_tokens

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(" ".join(current), document, len(chunks)))

        return chunks


def get_chunker(
    chunker_type: ChunkerType | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> BaseChunker:
    """
    Return the configured chunker.
    Falls back to settings values for any un-provided parameter.
    """
    ctype = chunker_type or settings.chunker_type
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap

    if ctype == ChunkerType.RECURSIVE:
        return RecursiveChunker(size, overlap)
    elif ctype == ChunkerType.SEMANTIC:
        return SemanticChunker(size, overlap)
    elif ctype == ChunkerType.SLIDING_WINDOW:
        return SlidingWindowChunker(size, overlap)
    elif ctype == ChunkerType.SENTENCE:
        return SentenceChunker(size, overlap)
    else:
        raise ValueError(f"Unknown chunker type: {ctype}")


def chunk_documents(
    documents: list[Document], chunker: BaseChunker | None = None
) -> list[Chunk]:
    """Convenience: chunk a list of documents and return all chunks."""
    if chunker is None:
        chunker = get_chunker()

    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunker.split(doc)
        all_chunks.extend(chunks)
        logger.debug("Chunked '%s' into %d chunks", doc.source, len(chunks))

    logger.info(
        "Total chunks produced: %d from %d documents", len(all_chunks), len(documents)
    )
    return all_chunks
