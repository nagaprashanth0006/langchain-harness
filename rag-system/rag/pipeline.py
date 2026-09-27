"The full RAG pipeline: ingestion + query, end-to-end."

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from config import settings
from embedding.embedders import BaseEmbedder, get_embedder
from generation.generator import BaseGenerator, get_generator
from generation.prompts import RAGPromptTemplate
from ingestion.chunkers import BaseChunker, chunk_documents, get_chunker
from ingestion.loaders import load_document, load_documents, load_from_bytes
from models import Chunk, Document, RAGResponse, ScoredChunk
from retrieval.reranker import BaseReranker, get_reranker
from retrieval.retriever import (
    BaseRetriever,
    BM25Retriever,
    HybridRetriever,
    get_retriever,
)
from storage.vector_store import BaseVectorStore, get_vector_store

logger = logging.getLogger(__name__)


class RAGPipeline:
    "Orchestrates the full RAG pipeline."

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        generator: BaseGenerator,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

        self._bm25: Optional[BM25Retriever] = None
        if isinstance(retriever, HybridRetriever):
            self._bm25 = retriever.sparse
        elif hasattr(retriever, "_bm25"):
            self._bm25 = retriever._bm25

        self._all_chunks: list[Chunk] = []

    @classmethod
    def from_settings(cls) -> "RAGPipeline":
        """
        Construct a fully configured pipeline from environment/settings.
        This is the standard production entry point.
        """
        chunker = get_chunker()
        embedder = get_embedder()
        vector_store = get_vector_store()
        reranker = get_reranker()

        retriever = get_retriever(
            vector_store=vector_store,
            embedder=embedder,
        )
        generator = get_generator()

        pipeline = cls(
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            generator=generator,
        )
        logger.info(
            "RAGPipeline initialized: chunker=%s embedder=%s store=%s "
            "retriever=%s reranker=%s generator=%s",
            settings.chunker_type,
            settings.embedder_type,
            settings.vector_store_type,
            settings.retriever_type,
            settings.reranker_type,
            settings.generator_type,
        )
        return pipeline

    def _ingest_documents(self, documents: list[Document]) -> list[Chunk]:
        """
        Core ingestion: chunk → embed → store.
        Returns the list of produced chunks.
        """
        if not documents:
            return []

        t0 = time.perf_counter()

        chunks = chunk_documents(documents, self.chunker)
        logger.info("Produced %d chunks from %d documents", len(chunks), len(documents))

        chunks = self.embedder.embed_chunks(chunks)
        logger.info("Embedded %d chunks", len(chunks))

        self.vector_store.upsert(chunks)

        self._all_chunks.extend(chunks)
        if self._bm25 is not None:
            self._bm25.add_chunks(chunks)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingestion complete: %d chunks in %.2fs (%.1f chunks/s)",
            len(chunks),
            elapsed,
            len(chunks) / elapsed,
        )
        return chunks

    def ingest_file(
        self,
        path: Union[str, Path],
        max_mb: int | None = None,
    ) -> list[Chunk]:
        """Ingest a single file from disk."""
        documents = load_document(path, max_mb=max_mb or settings.max_file_size_mb)
        return self._ingest_documents(documents)

    def ingest_files(
        self,
        paths: list[Union[str, Path]],
        max_mb: int | None = None,
    ) -> list[Chunk]:
        """Ingest multiple files. Failures are logged and skipped."""
        documents = load_documents(paths, max_mb=max_mb or settings.max_file_size_mb)
        return self._ingest_documents(documents)

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        max_mb: int | None = None,
    ) -> list[Chunk]:
        """Ingest raw bytes (e.g., from an HTTP upload)."""
        documents = load_from_bytes(
            content, filename, max_mb=max_mb or settings.max_file_size_mb
        )
        return self._ingest_documents(documents)

    def ingest_text(self, text: str, source: str = "inline") -> list[Chunk]:
        """Ingest raw text directly (for testing or API ingestion)."""
        doc = Document(content=text, source=source, metadata={"source": source})
        return self._ingest_documents([doc])

    def delete_source(self, source: str) -> None:
        """
        Remove all chunks from a given source file.
        Also rebuilds the BM25 index without those chunks.
        """
        self.vector_store.delete_by_source(source)
        self._all_chunks = [c for c in self._all_chunks if c.source != source]
        if self._bm25 is not None and self._all_chunks:
            self._bm25.index(self._all_chunks)
        logger.info("Deleted source from index: %s", source)

    def query(
        self,
        query: str,
        top_k_retrieval: int | None = None,
        top_k_final: int | None = None,
        filter: Optional[dict[str, Any]] = None,
        template: Optional[RAGPromptTemplate] = None,
    ) -> RAGResponse:
        "Full RAG query: retrieve → rerank → generate."
        t0 = time.perf_counter()

        retrieval_k = top_k_retrieval or settings.top_k_retrieval
        candidates = self.retriever.retrieve(query, top_k=retrieval_k, filter=filter)
        t_retrieve = time.perf_counter()

        if not candidates:
            logger.warning("No candidates retrieved for query: '%.60s'", query)
            return RAGResponse(
                answer="I could not find any relevant information in the documents.",
                query=query,
                sources=[],
                model=self.generator.model_name,
            )

        final_k = top_k_final or settings.top_k_final
        context_chunks = self.reranker.rerank(query, candidates, top_k=final_k)
        t_rerank = time.perf_counter()

        response = self.generator.generate(query, context_chunks, template)
        t_generate = time.perf_counter()

        logger.info(
            "RAG query: retrieve=%.2fs rerank=%.2fs generate=%.2fs total=%.2fs "
            "candidates=%d final=%d tokens=%d",
            t_retrieve - t0,
            t_rerank - t_retrieve,
            t_generate - t_rerank,
            t_generate - t0,
            len(candidates),
            len(context_chunks),
            response.total_tokens,
        )
        return response

    def query_stream(
        self,
        query: str,
        top_k_retrieval: int | None = None,
        top_k_final: int | None = None,
        filter: Optional[dict[str, Any]] = None,
        template: Optional[RAGPromptTemplate] = None,
    ) -> Iterator[str]:
        """
        Streaming RAG query. Retrieval and reranking are synchronous;
        generation is streamed token by token.

        Usage:
            for token in pipeline.query_stream("What are the risks?"):
                print(token, end="", flush=True)
        """
        retrieval_k = top_k_retrieval or settings.top_k_retrieval
        candidates = self.retriever.retrieve(query, top_k=retrieval_k, filter=filter)

        if not candidates:
            yield "I could not find any relevant information in the documents."
            return

        final_k = top_k_final or settings.top_k_final
        context_chunks = self.reranker.rerank(query, candidates, top_k=final_k)

        yield from self.generator.generate_stream(query, context_chunks, template)

    def stats(self) -> dict[str, Any]:
        """Return current pipeline stats."""
        return {
            "vector_store_count": self.vector_store.count(),
            "bm25_index_size": len(self._all_chunks),
            "chunker": settings.chunker_type,
            "embedder": settings.embedder_type,
            "vector_store": settings.vector_store_type,
            "retriever": settings.retriever_type,
            "reranker": settings.reranker_type,
            "generator": settings.generator_type,
            "generator_model": self.generator.model_name,
        }
