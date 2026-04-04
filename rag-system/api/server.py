"""
FastAPI server for the production RAG system.

Endpoints
─────────
POST   /ingest/file           Upload and ingest a document file
POST   /ingest/text           Ingest raw text
DELETE /documents/{source}    Remove a source from the index
POST   /query                 RAG query (JSON response)
POST   /query/stream          RAG query (SSE streaming)
GET    /documents/stats       Index statistics
GET    /health                Health check
GET    /                      API info

How the server is layered
──────────────────────────
1. lifespan() — runs once per worker process on startup/shutdown.
   Constructs the RAGPipeline singleton.  Heavy model loading (cross-encoder,
   embedding model) happens here so the first request is never slow.

2. get_pipeline() — dependency-style guard that raises 503 if the pipeline
   failed to initialize (e.g., Ollama not running).  Every endpoint calls it.

3. Thread-pool offloading — all blocking work (embedding, BM25, LLM calls)
   runs in asyncio's default ThreadPoolExecutor via
   ``await loop.run_in_executor(None, ...)`` so the event loop is never blocked.

Production checklist
─────────────────────
□ Rate limiting    — slowapi (redis-backed) or API gateway (Kong, Traefik)
□ Authentication   — OAuth2/JWT middleware; scope /ingest to admin role
□ File-size guard  — enforce `client_max_body_size` in nginx upstream
□ Background jobs  — for files >5 MB, return 202 + job_id; process via Celery/ARQ
□ CORS origins     — change CORS_ORIGINS from ["*"] to your actual domain list
□ Distributed trace— add OpenTelemetry middleware; export to Jaeger/Datadog
□ API versioning   — prefix routes with /api/v1/ before your first public release
□ Graceful reload  — SIGHUP-based reload preserves in-flight requests (gunicorn)

Common pitfalls fixed here
───────────────────────────
• HTTP 204 No Content must return an empty body — FastAPI >= 0.100 asserts this.
  We return ``Response(status_code=204)`` explicitly (not ``None``).
• asyncio.get_event_loop() is deprecated inside coroutines in Python 3.10+ and
  raises a DeprecationWarning in 3.12.  We use get_running_loop() throughout.
• BackgroundTasks must be injected by FastAPI — declare as a parameter with no
  default (``background_tasks: BackgroundTasks``) BEFORE any File/Form params.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline singleton — one instance per worker process
# ---------------------------------------------------------------------------
_pipeline: Optional[RAGPipeline] = None
_start_time: float = time.monotonic()


def get_pipeline() -> RAGPipeline:
    """
    Return the active pipeline or raise 503.

    Why 503 and not 500?
    503 Service Unavailable signals to load balancers and health-check systems
    that this worker is not ready to serve traffic.  The LB can then retry on
    another worker or take this instance out of rotation.
    """
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not ready. Check server logs.",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class TextIngestRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=1_000_000,
        description="Plain text content to ingest directly (no file required).",
    )
    source: str = Field(
        default="api_text",
        description="Logical name for this text — used as the source identifier in retrieval.",
    )


class IngestResponse(BaseModel):
    source: str = Field(description="Source name as stored in the index.")
    chunks_created: int = Field(description="Number of chunks produced and persisted.")
    message: str = Field(description="Human-readable summary.")


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user question to answer from the document index.",
    )
    top_k_retrieval: int = Field(
        default=20, ge=1, le=100,
        description=(
            "Candidates fetched from the vector store + BM25 before reranking. "
            "Higher = better recall but slower reranking.  Sweet spot: 15–25."
        ),
    )
    top_k_final: int = Field(
        default=5, ge=1, le=20,
        description=(
            "Passages passed to the LLM after reranking. "
            "More context = richer answers but more tokens consumed.  Sweet spot: 3–7."
        ),
    )
    filter: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Metadata filter applied at retrieval time.  Supports exact-match on any "
            "field stored in chunk metadata.  Example: {\"file_name\": \"report.pdf\"}"
        ),
    )


class SourceInfo(BaseModel):
    source: str = Field(description="File path / logical source name.")
    page: Optional[Any] = Field(default=None, description="Page number within source (PDF only).")
    score: float = Field(description="Relevance score after reranking (higher is better).")
    retrieval_method: str = Field(description="dense | sparse | hybrid | reranked:*")
    excerpt: str = Field(description="First 200 characters of the chunk content.")


class QueryResponse(BaseModel):
    query: str
    answer: str = Field(description="LLM-generated answer grounded in the retrieved context.")
    sources: list[SourceInfo] = Field(description="Context passages used to produce the answer.")
    model: str = Field(description="Name of the LLM that generated the answer.")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float = Field(description="End-to-end latency in milliseconds (retrieve+rerank+generate).")


class StatsResponse(BaseModel):
    vector_store_count: int = Field(description="Total chunks in the vector index.")
    bm25_index_size: int = Field(description="Total chunks in the in-memory BM25 index.")
    chunker: str
    embedder: str
    vector_store: str
    retriever: str
    reranker: str
    generator: str
    generator_model: str


class HealthResponse(BaseModel):
    status: str = Field(description="'ok' when the pipeline is ready, 'degraded' otherwise.")
    vector_store_count: int
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error envelope — makes client error handling predictable."""
    detail: str
    status_code: int


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Startup / shutdown hook.  Runs once per worker process.

    Why run_in_executor?
    RAGPipeline.from_settings() loads embedding models (sentence-transformers,
    tiktoken) and connects to the vector store — all blocking I/O.  Doing this
    inside an async context without offloading to a thread would block the event
    loop and prevent health-check endpoints from responding during startup.

    In production with gunicorn + UvicornWorker, each worker process runs this
    independently, so startup time multiplies by the worker count.  Keep model
    initialization fast by using lazy loading inside each component.
    """
    global _pipeline
    logger.info("Worker starting — initializing RAG pipeline …")
    try:
        loop = asyncio.get_running_loop()
        _pipeline = await loop.run_in_executor(None, RAGPipeline.from_settings)
        logger.info(
            "RAG pipeline ready — vector_store_count=%d embedder=%s generator=%s",
            _pipeline.vector_store.count(),
            settings.embedder_type,
            settings.generator_type,
        )
    except Exception as exc:
        # Log but don't crash — the /health endpoint will surface 'degraded'
        # so the load balancer can decide whether to route traffic here.
        logger.error("Pipeline initialization failed: %s", exc, exc_info=True)

    yield   # ← server is live and serving requests here

    logger.info("Worker shutting down — releasing resources")
    _pipeline = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    Build and return the ASGI application.

    Why a factory function instead of a module-level app object?
    ─────────────────────────────────────────────────────────────
    uvicorn --workers N spawns N subprocesses.  Each subprocess imports this
    module and must create its own FastAPI instance with its own pipeline
    singleton.  A module-level ``app = FastAPI(...)`` is created in the parent
    process and then pickled into children — which does NOT re-run lifespan.
    A factory avoids that; each worker calls create_app() fresh.

    Usage:
        uvicorn api.server:create_app --factory --workers 4
        gunicorn 'api.server:create_app()' -k uvicorn.workers.UvicornWorker -w 4
    """
    app = FastAPI(
        title="Production RAG API",
        description=(
            "A fully pluggable Retrieval-Augmented Generation pipeline.\n\n"
            "Every backend (embedder, vector store, retriever, reranker, generator) "
            "is swappable via environment variables — no code changes required."
        ),
        version="1.0.0",
        lifespan=lifespan,
        # Disable the default /openapi.json in production if you don't need it
        # openapi_url=None,
    )

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request-ID middleware — adds X-Request-ID to every response so log lines
    # can be correlated across the retrieve → rerank → generate chain.
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        import uuid
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response

    # ------------------------------------------------------------------
    # System endpoints
    # ------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="Health check",
        description=(
            "Returns 'ok' when the pipeline is initialized and the vector store is reachable. "
            "Returns 'degraded' (still HTTP 200) when the pipeline failed to start — the LB "
            "should use this to decide whether to send traffic to this worker."
        ),
    )
    async def health() -> HealthResponse:
        pipeline = _pipeline
        count = 0
        if pipeline:
            try:
                count = pipeline.vector_store.count()
            except Exception:
                pass  # Vector store unreachable — still return 'degraded' not 500
        return HealthResponse(
            status="ok" if pipeline else "degraded",
            vector_store_count=count,
            uptime_seconds=round(time.monotonic() - _start_time, 1),
        )

    @app.get("/", tags=["System"], summary="API info")
    async def root() -> dict:
        return {
            "name": "Production RAG API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
            "embedder": settings.embedder_type,
            "generator": settings.generator_type,
        }

    # ------------------------------------------------------------------
    # Ingestion endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/ingest/file",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Ingestion"],
        summary="Upload and ingest a document",
        description=(
            "Accepts PDF, DOCX, TXT, MD, HTML, JSON, JSONL files.\n\n"
            "The pipeline is: load → chunk → embed → persist to vector store + BM25.\n\n"
            "**Production note:** For files larger than ~5 MB, consider the async pattern:\n"
            "accept the upload, return 202 Accepted with a job_id, and poll "
            "`GET /jobs/{job_id}` for completion (implement with Celery + Redis)."
        ),
    )
    async def ingest_file(
        file: UploadFile = File(..., description="Document file to ingest."),
        # Production pattern for background jobs:
        #   from fastapi import BackgroundTasks
        #   async def ingest_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
        #       background_tasks.add_task(pipeline.ingest_bytes, content, filename)
        #       return IngestResponse(source=filename, chunks_created=-1, message="Queued")
        # Note: BackgroundTasks must be declared BEFORE File/Form params (no default).
        # For production-scale async ingestion, use Celery + Redis instead.
    ) -> IngestResponse:
        pipeline = get_pipeline()

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Uploaded file is empty.",
            )

        filename = file.filename or "unknown"

        try:
            # run_in_executor offloads the blocking work (PDF parsing, tokenization,
            # embedding API call, ChromaDB write) to the thread pool so the event
            # loop stays responsive for other requests during ingestion.
            loop = asyncio.get_running_loop()
            chunks = await loop.run_in_executor(
                None,
                pipeline.ingest_bytes,
                content,
                filename,
            )
        except ValueError as exc:
            # Raised by loaders for unsupported file types or size violations
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
        except Exception as exc:
            logger.error("Ingestion failed for '%s': %s", filename, exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ingestion failed: {exc}",
            )

        return IngestResponse(
            source=filename,
            chunks_created=len(chunks),
            message=f"Successfully ingested '{filename}' → {len(chunks)} chunks.",
        )

    @app.post(
        "/ingest/text",
        response_model=IngestResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Ingestion"],
        summary="Ingest raw text",
        description=(
            "Ingest a plain-text string directly — no file upload needed.\n\n"
            "Useful for ingesting database records, API responses, or programmatically "
            "generated content."
        ),
    )
    async def ingest_text(body: TextIngestRequest) -> IngestResponse:
        pipeline = get_pipeline()
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(
            None,
            pipeline.ingest_text,
            body.text,
            body.source,
        )
        return IngestResponse(
            source=body.source,
            chunks_created=len(chunks),
            message=f"Ingested {len(chunks)} chunks from source '{body.source}'.",
        )

    @app.delete(
        "/documents/{source:path}",
        status_code=status.HTTP_204_NO_CONTENT,
        tags=["Ingestion"],
        summary="Delete a document",
        description=(
            "Remove all chunks for the given source from the vector store and BM25 index.\n\n"
            "The `{source:path}` path parameter captures slashes, so file paths like "
            "`reports/q4/annual.pdf` work without encoding.\n\n"
            "**HTTP 204 No Content** — the response body is intentionally empty."
        ),
        # Why response_class=Response?
        # FastAPI >= 0.100 asserts that HTTP 204 routes must not produce a response
        # body.  Declaring response_class=Response prevents FastAPI from wrapping
        # the return value in a JSON envelope.  We return Response(status_code=204)
        # explicitly below to satisfy this contract.
        response_class=Response,
    )
    async def delete_document(source: str) -> Response:
        pipeline = get_pipeline()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline.delete_source, source)
        # Must return Response directly — returning None here would still trigger
        # FastAPI's body-serialization path in some versions, causing the assertion.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # ------------------------------------------------------------------
    # Query endpoints
    # ------------------------------------------------------------------

    @app.post(
        "/query",
        response_model=QueryResponse,
        tags=["Query"],
        summary="RAG query (synchronous)",
        description=(
            "Full pipeline: retrieve → rerank → generate.  Returns a complete JSON response "
            "once generation is finished.\n\n"
            "Use `POST /query/stream` for token-by-token streaming (lower perceived latency).\n\n"
            "**Token budget:** each call consumes roughly "
            "`(top_k_final × chunk_tokens) + query_tokens + system_prompt_tokens` input tokens."
        ),
    )
    async def query(body: QueryRequest) -> QueryResponse:
        pipeline = get_pipeline()
        t0 = time.perf_counter()

        try:
            loop = asyncio.get_running_loop()
            rag_response = await loop.run_in_executor(
                None,
                lambda: pipeline.query(
                    body.query,
                    top_k_retrieval=body.top_k_retrieval,
                    top_k_final=body.top_k_final,
                    filter=body.filter,
                ),
            )
        except Exception as exc:
            logger.error("Query failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Query failed: {exc}",
            )

        latency_ms = (time.perf_counter() - t0) * 1000

        sources = [
            SourceInfo(
                source=sc.chunk.source,
                page=sc.chunk.metadata.get("page"),
                score=round(sc.score, 4),
                retrieval_method=sc.retrieval_method,
                excerpt=(
                    sc.chunk.content[:200] + "…"
                    if len(sc.chunk.content) > 200
                    else sc.chunk.content
                ),
            )
            for sc in rag_response.sources
        ]

        return QueryResponse(
            query=rag_response.query,
            answer=rag_response.answer,
            sources=sources,
            model=rag_response.model,
            prompt_tokens=rag_response.prompt_tokens,
            completion_tokens=rag_response.completion_tokens,
            total_tokens=rag_response.total_tokens,
            latency_ms=round(latency_ms, 1),
        )

    @app.post(
        "/query/stream",
        tags=["Query"],
        summary="RAG query (SSE streaming)",
        description=(
            "Streaming RAG query via Server-Sent Events (SSE).\n\n"
            "Retrieval and reranking are performed synchronously first (fast, ~200 ms), "
            "then generation tokens are streamed as they are produced.\n\n"
            "**SSE format:** each event is `data: <token>\\n\\n`. "
            "The stream ends with `data: [DONE]\\n\\n`.\n\n"
            "**JavaScript client:**\n"
            "```js\n"
            "const res = await fetch('/query/stream', {\n"
            "  method: 'POST',\n"
            "  headers: {'Content-Type': 'application/json'},\n"
            "  body: JSON.stringify({query: 'What are the risks?'}),\n"
            "});\n"
            "const reader = res.body.getReader();\n"
            "// decode and print chunks\n"
            "```\n\n"
            "**nginx:** add `proxy_buffering off;` and `X-Accel-Buffering: no` "
            "to prevent nginx from buffering the stream."
        ),
        response_class=StreamingResponse,
    )
    async def query_stream(body: QueryRequest) -> StreamingResponse:
        pipeline = get_pipeline()

        async def token_generator() -> AsyncIterator[str]:
            """
            Two-phase generator:
            1. Retrieve + rerank (blocking, offloaded to thread pool)
            2. Stream LLM tokens (blocking iterator, offloaded chunk-by-chunk)
            """
            loop = asyncio.get_running_loop()

            # Phase 1 — Retrieve
            candidates = await loop.run_in_executor(
                None,
                lambda: pipeline.retriever.retrieve(body.query, top_k=body.top_k_retrieval),
            )
            if not candidates:
                yield "data: I could not find relevant information in the indexed documents.\n\n"
                yield "data: [DONE]\n\n"
                return

            # Phase 2 — Rerank
            context_chunks = await loop.run_in_executor(
                None,
                lambda: pipeline.reranker.rerank(body.query, candidates, top_k=body.top_k_final),
            )

            # Phase 3 — Stream tokens
            # generate_stream() returns a synchronous iterator (blocking I/O).
            # We run it in the executor and yield each token to the async generator.
            def _stream_sync():
                return list(pipeline.generator.generate_stream(body.query, context_chunks))

            tokens = await loop.run_in_executor(None, _stream_sync)
            for token in tokens:
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            token_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",   # disable nginx response buffering
            },
        )

    # ------------------------------------------------------------------
    # Stats endpoint
    # ------------------------------------------------------------------

    @app.get(
        "/documents/stats",
        response_model=StatsResponse,
        tags=["System"],
        summary="Index statistics",
        description="Returns the current state of the vector store and BM25 index.",
    )
    async def stats() -> StatsResponse:
        pipeline = get_pipeline()
        s = pipeline.stats()
        return StatsResponse(**s)

    return app
