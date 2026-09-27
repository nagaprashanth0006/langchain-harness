"Shared domain models for the RAG pipeline."

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Document:
    """A raw document loaded from disk / upload."""

    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    title: Optional[str] = None
    page_number: Optional[int] = None
    total_pages: Optional[int] = None


@dataclass
class Chunk:
    """A text chunk derived from a Document, ready for embedding."""

    content: str
    doc_id: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    embedding: Optional[list[float]] = None


@dataclass
class ScoredChunk:
    """A chunk returned from retrieval with a relevance score."""

    chunk: Chunk
    score: float
    retrieval_method: str = "dense"

    def __lt__(self, other: "ScoredChunk") -> bool:
        return self.score < other.score


@dataclass
class RAGResponse:
    """Final answer produced by the RAG pipeline."""

    answer: str
    query: str
    sources: list[ScoredChunk]
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def source_documents(self) -> list[str]:
        """Unique source file paths used to produce the answer."""
        seen: set[str] = set()
        out: list[str] = []
        for sc in self.sources:
            if sc.chunk.source not in seen:
                seen.add(sc.chunk.source)
                out.append(sc.chunk.source)
        return out
