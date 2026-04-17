# Production RAG System

A fully pluggable, production-grade Retrieval-Augmented Generation pipeline. Every stage is swappable via environment variables — no code changes needed to switch from ChromaDB to pgvector, BM25 to hybrid, or Claude to GPT-4o.

---

## Quick Start

```bash
cd rag-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in ANTHROPIC_API_KEY (minimum required)

# Start the API server
python api/server.py
# → http://localhost:8000/docs

# Or use the pipeline directly in Python
python - <<'EOF'
from rag.pipeline import RAGPipeline
pipeline = RAGPipeline.from_settings()
pipeline.ingest_file("my_document.pdf")
response = pipeline.query("What is this document about?")
print(response.answer)
EOF
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI Server                        │
│  POST /ingest/file  POST /query  POST /query/stream  GET /health │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    RAGPipeline.from_settings()
                           │
         ┌─────────────────┼─────────────────────┐
         │                 │                       │
   INGESTION PATH    QUERY PATH              UTILS
         │                 │
   ┌─────▼──────┐   ┌──────▼───────┐
   │  Loaders   │   │  Retriever   │
   │  PDF/DOCX  │   │  Dense+BM25  │
   │  HTML/TXT  │   │  Hybrid RRF  │
   └─────┬──────┘   └──────┬───────┘
         │                 │
   ┌─────▼──────┐   ┌──────▼───────┐
   │  Chunkers  │   │   Reranker   │
   │  Recursive │   │ CrossEncoder │
   │  Semantic  │   │    Cohere    │
   └─────┬──────┘   └──────┬───────┘
         │                 │
   ┌─────▼──────┐   ┌──────▼───────┐
   │  Embedders │   │  Generator   │
   │  OpenAI    │   │   Claude     │
   │  HuggFace  │   │   OpenAI     │
   └─────┬──────┘   └──────────────┘
         │
   ┌─────▼──────┐
   │VectorStore │
   │  ChromaDB  │
   │  pgvector  │
   └────────────┘
```

---

## Production Guide: Each Stage Explained

### Stage 1 — Document Loading

**What it does:** Reads raw files (PDF, DOCX, HTML, TXT, JSON) into `Document` objects, preserving metadata (page number, source, title).

**Production caveats:**
- **PDF text order is not guaranteed.** Multi-column PDFs extracted with pypdf may produce garbled text. Always test on your actual corpus, not just "clean" samples.
- **Scanned PDFs have zero extractable text.** You need OCR (Azure Document Intelligence, AWS Textract, Tesseract). Plan for this before deployment if any % of your corpus is scanned.
- **Encoding issues are common** in legacy text files. Always detect encoding with `chardet` rather than assuming UTF-8.
- **Large files are a DoS vector.** Set `max_file_size_mb` and enforce it at the network layer (nginx `client_max_body_size`) too, not just in application code.

**Alternatives by format:**

| Format | This system | Better for tables | Best quality/layout |
|--------|-------------|-------------------|---------------------|
| PDF | pypdf | pdfplumber | Azure Doc Intel |
| HTML | BeautifulSoup | trafilatura | Jina Reader |
| DOCX | python-docx | unstructured.io | unstructured.io |
| Scanned | ❌ | ❌ | Textract / Tesseract |

**When to use unstructured.io:** When your PDFs have complex layouts (multi-column, forms, mixed text+tables). It's the best open-source option but requires significant deps or a Docker container.

---

### Stage 2 — Chunking

**What it does:** Splits documents into smaller text units (chunks) that fit within the embedding model's context window and are granular enough for precise retrieval.

**The fundamental tension:** Small chunks → precise retrieval, insufficient context. Large chunks → rich context, imprecise retrieval.

**Config:** `CHUNK_SIZE=512 CHUNK_OVERLAP=64 CHUNKER_TYPE=recursive`

**Strategy comparison:**

| Strategy | Best for | Caveat |
|----------|----------|--------|
| `recursive` (default) | General documents | May split mid-topic |
| `semantic` | Dense narrative text | Slow (embeds at ingest time) |
| `sliding_window` | Legal/technical specs | High redundancy, higher cost |
| `sentence` | FAQ, short factual docs | Tiny chunks, noisy retrieval |

**Production caveats:**
- **Token counting vs character counting:** Always count tokens, not characters. A 512-character chunk can be 50 or 200 tokens depending on the language and content.
- **Chunk overlap is mandatory for cross-boundary context.** Without overlap, answers that span chunk boundaries are invisible to the retriever. Rule of thumb: overlap ≈ 12% of chunk_size.
- **Preserve structure in metadata.** Store `page_number`, `section_header`, `document_title` in chunk metadata. This powers metadata filtering during retrieval.
- **Multi-language:** Recursive/sentence chunkers perform poorly on CJK (Chinese, Japanese, Korean). Use language-aware tokenizers.
- **Code chunks:** Split code at function/class boundaries, not arbitrary character counts. Use tree-sitter or AST-based splitters.

**Advanced: Parent-Child Chunking**
Index small chunks (128 tokens) for retrieval precision, but return the parent chunk (512 tokens) to the LLM for context richness. LangChain's `ParentDocumentRetriever` implements this pattern.

---

### Stage 3 — Embedding

**What it does:** Converts text chunks into dense vectors (floating-point arrays) that encode semantic meaning. Similar texts cluster together in vector space.

**Config:** `EMBEDDER_TYPE=openai OPENAI_EMBEDDING_MODEL=text-embedding-3-small`

**Model comparison:**

| Model | Dim | Quality | Cost | Privacy |
|-------|-----|---------|------|---------|
| `text-embedding-3-small` | 1536 | ★★★★☆ | $0.02/1M tokens | Cloud |
| `text-embedding-3-large` | 3072 | ★★★★★ | $0.13/1M tokens | Cloud |
| `BAAI/bge-small-en-v1.5` | 384 | ★★★☆☆ | Free | Local |
| `BAAI/bge-large-en-v1.5` | 1024 | ★★★★☆ | Free | Local |
| `intfloat/e5-large-v2` | 1024 | ★★★★☆ | Free | Local |
| `cohere embed-english-v3.0` | 1024 | ★★★★☆ | $0.10/1M | Cloud |

**Production caveats:**
- **Embedding model lock-in.** Once you embed 1M chunks with model A, switching to model B requires re-embedding everything. Version your embedding model in chunk metadata.
- **Query vs document asymmetry.** Some models (E5, BGE) require different prefixes for queries vs documents. Using the wrong prefix drops recall by 10–30%.
- **Batch size matters.** Embedding chunks one-by-one is 50–100× slower than batching 32 at a time. Always batch.
- **Disk cache reduces cost.** Re-embedding the same text is wasteful. The built-in disk cache uses SHA-256 hashing — production should use Redis for shared cache across workers.
- **Matryoshka Representation Learning (MRL):** OpenAI 3-small/large support truncated dimensions. Use `dimensions=512` for 66% storage reduction with <3% quality loss.
- **Cost estimate:** 1M chunks × 512 tokens avg = ~500M tokens = $10 with text-embedding-3-small.

---

### Stage 4 — Vector Storage

**What it does:** Stores chunk embeddings in a database optimized for approximate nearest-neighbor (ANN) search. Returns top-k most similar chunks for a query embedding.

**Config:** `VECTOR_STORE_TYPE=chroma`

**Store comparison:**

| Store | Scale | Ops burden | Best for |
|-------|-------|------------|----------|
| ChromaDB | <1M vectors | None | Dev, small prod |
| pgvector | <10M vectors | Low (if using Postgres) | Existing Postgres users |
| Qdrant | <100M vectors | Medium | High-perf production |
| Weaviate | <100M vectors | Medium | GraphQL, hybrid search |
| Pinecone | Billions | None (managed) | Large-scale SaaS |
| Milvus | Billions | High | Enterprise GPU workloads |

**Index types:**

| Index | Build time | Query speed | Recall | Use when |
|-------|-----------|-------------|--------|----------|
| Brute-force | O(1) | O(N) | 100% | N < 10K |
| IVFFlat | Fast | Fast | 90–95% | 10K–1M |
| HNSW | Slow | Fastest | 95–99% | >100K |
| ScaNN | Slow | Fastest | 99%+ | Google-scale |

**Production caveats:**
- **HNSW parameters matter.** `m=16 ef_construction=200` is a good start. Higher values = better recall but slower build and more RAM.
- **Metadata filtering is essential.** Without it, every query scans the entire index. Index frequently filtered fields (`source`, `date`, `category`).
- **Single-writer limitation (ChromaDB).** Chroma doesn't support concurrent writes. Use a queue (Celery/ARQ) to serialize ingestion in multi-worker deployments.
- **pgvector HNSW** (added in v0.5) is production-ready. Run `VACUUM ANALYZE` after bulk inserts. Set `ivfflat.probes=10` (default=1) for better recall.
- **Backup strategy:** For Chroma, copy the persist directory. For pgvector, use pg_dump. Test restore quarterly.

---

### Stage 5 — Retrieval

**What it does:** Given a query, finds the most relevant chunks in the index.

**Config:** `RETRIEVER_TYPE=hybrid HYBRID_ALPHA=0.5 TOP_K_RETRIEVAL=20`

**Strategy comparison:**

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| Dense only | Semantic paraphrase, concept matching | Fails on rare terms, IDs, codes |
| Sparse only (BM25) | Exact keywords, rare terms | No semantic understanding |
| Hybrid (RRF) | Best of both worlds | More complex, ~2× latency |

**Reciprocal Rank Fusion (RRF):**
```
score(d) = Σ  1 / (k + rank_i(d))   where k=60
            i ∈ {dense, sparse}
```
RRF is parameter-free, scale-invariant, and consistently outperforms both single methods by 5–15% NDCG. Use it as your default.

**Production caveats:**
- **BM25 is in-memory here.** For production with >1M chunks, use Elasticsearch or OpenSearch which maintain a persistent inverted index on disk.
- **top_k_retrieval should be 3–5× top_k_final.** Retrieve 20, rerank to 5. This gives the reranker enough candidates to find the best ones.
- **Query expansion improves sparse recall.** Expand queries with synonyms or use HyDE (generate a hypothetical answer and embed that instead of the query).
- **Metadata filtering is a first-class feature.** Let users filter by document, date range, category. This dramatically improves precision for multi-document indexes.

**Advanced techniques:**
- **HyDE:** Generate a hypothetical answer → embed it → retrieve. Better for open-domain QA where queries are very different from document style.
- **Multi-query:** Generate 3–5 query variations → retrieve for each → deduplicate + fuse. Improves recall by ~15%.
- **SPLADE:** Learned sparse model that outperforms BM25 while remaining interpretable. Drop-in replacement for BM25 in many architectures.

---

### Stage 6 — Reranking

**What it does:** Takes the top-k candidates from retrieval and re-scores them with a more powerful model, sorting the best ones to the top.

**Why it's critical:** Retrieval optimizes for recall (find anything relevant). Reranking optimizes for precision (put the most relevant first). Without reranking, context quality suffers.

**Config:** `RERANKER_TYPE=cross_encoder CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`

**Reranker comparison:**

| Reranker | Quality | Latency | Cost | Privacy |
|----------|---------|---------|------|---------|
| Cross-encoder (local) | ★★★★☆ | 50–200ms CPU | Free | Local |
| Cross-encoder (GPU) | ★★★★☆ | 10–30ms | GPU cost | Local |
| Cohere Rerank v3 | ★★★★★ | 100–300ms | API cost | Cloud |
| FlashRank (ONNX) | ★★★★☆ | 5–15ms CPU | Free | Local |
| LLM listwise | ★★★★★ | 1–3s | LLM cost | Depends |
| None | N/A | 0ms | Free | N/A |

**Production caveats:**
- **Cross-encoder bottleneck.** With 20 candidates × ~50ms each = 1s on CPU. Parallelize with ThreadPoolExecutor or use GPU/FlashRank.
- **Cohere multilingual reranker** (`rerank-multilingual-v3.0`) is the easiest path to multilingual RAG. Local cross-encoders require separate multilingual models.
- **Score thresholding.** Cross-encoder scores are calibrated; you can set a minimum score threshold (e.g., 0.1) to avoid injecting irrelevant context.
- **Reranking k=20→5 is the sweet spot.** Beyond k=50 input candidates, quality gains plateau while latency increases linearly.

---

### Stage 7 — Generation (Augmented Generation)

**What it does:** Feeds the reranked context passages + query to an LLM to produce a grounded, cited answer.

**Config:** `GENERATOR_TYPE=anthropic ANTHROPIC_MODEL=claude-sonnet-4-6`

**Model comparison:**

| Model | Context | Quality | Speed | Cost/1M tokens |
|-------|---------|---------|-------|----------------|
| Claude Sonnet 4.6 | 200K | ★★★★★ | Fast | ~$3 in / $15 out |
| Claude Haiku 4.5 | 200K | ★★★★☆ | Fastest | ~$0.25/$1.25 |
| GPT-4o | 128K | ★★★★★ | Fast | ~$2.5/$10 |
| GPT-4o-mini | 128K | ★★★☆☆ | Fastest | ~$0.15/$0.60 |
| LLaMA 3.1 70B | 128K | ★★★★☆ | Medium | Self-hosted |

**Production caveats:**
- **Faithfulness vs fluency.** Without strong grounding instructions, LLMs add information from training data. Use explicit system prompt: "Answer ONLY from the provided context."
- **Prompt caching saves 50% cost.** The system prompt is identical across all requests. Claude and OpenAI both support caching it (different APIs, same concept).
- **Temperature matters.** Use `temperature=0.0–0.1` for factual RAG. Higher temperatures introduce creative drift — dangerous for grounded answers.
- **Context window isn't free.** At 5 chunks × 512 tokens each = 2560 tokens of context + system prompt + answer ≈ 4000 tokens per query. Budget accordingly.
- **Streaming is UX-critical.** Users perceive streaming as faster even when total latency is equal. Implement SSE (Server-Sent Events) for web interfaces.
- **Citation verification.** After generation, optionally verify that each [Source N] citation actually appears in the retrieved chunks (LLMs sometimes hallucinate citations).

**Advanced generation patterns:**
- **MapReduce:** For very long documents, first summarize each chunk separately (Map), then synthesize the summaries (Reduce). Avoids context window limits.
- **Self-RAG:** LLM decides when to retrieve (using special tokens), critiques its own answer, and retrieves again if needed. State-of-the-art for accuracy.
- **Iterative retrieval (ITER-RETGEN):** Generate draft answer → extract sub-questions → retrieve again → refine answer. Excellent for multi-hop questions.

---

## Evaluation

Never deploy changes to retrieval or generation without measuring quality.

### RAGAS (Recommended)
```bash
pip install ragas
```
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Requires a golden dataset: {question, answer, contexts, ground_truth}
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
```

| Metric | Measures | Target |
|--------|----------|--------|
| Faithfulness | Answer grounded in context? | > 0.85 |
| Answer Relevancy | Answer addresses the query? | > 0.85 |
| Context Precision | Retrieved docs relevant? | > 0.75 |
| Context Recall | All relevant docs found? | > 0.75 |

### Build a Golden Dataset
1. Sample 100–200 real user queries from logs (or generate with GPT-4)
2. Have domain experts write ground-truth answers
3. Run RAGAS before and after every change to the pipeline
4. Store results in a spreadsheet with the pipeline config (git SHA + env vars)

---

## Production Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Scaling Checklist
- [ ] Move BM25 to Elasticsearch for concurrent writes
- [ ] Move ChromaDB to Qdrant/Weaviate for horizontal scaling
- [ ] Add Redis cache for embeddings (shared across workers)
- [ ] Add Redis cache for query→answer (for repeated identical queries)
- [ ] Move ingestion to background queue (Celery + Redis)
- [ ] Add rate limiting per API key (slowapi)
- [ ] Add request tracing (OpenTelemetry → Jaeger/Datadog)
- [ ] Add embedding + LLM latency dashboards (Prometheus + Grafana)
- [ ] Set up alerts: p99 latency > 5s, error rate > 1%, vector store count drops

### Cost Optimization
1. **Cache aggressively:** Embeddings (disk/Redis), LLM responses (Redis with TTL)
2. **Use smaller models for routing:** Use Haiku to classify query type, then route complex queries to Sonnet
3. **Chunk compression:** LLMLingua compresses retrieved context by 3–20× with minimal quality loss
4. **Batch ingestion overnight:** Embedding APIs have off-peak discounts on some providers
5. **Monitor cost/query:** Track (prompt_tokens × input_price) + (completion_tokens × output_price) per request

---

## File Structure

```
rag-system/
├── config.py                   # All settings (env-driven)
├── models.py                   # Shared dataclasses (Document, Chunk, RAGResponse)
├── requirements.txt
├── .env.example
│
├── ingestion/
│   ├── loaders.py              # PDF, DOCX, HTML, TXT, JSON loaders
│   └── chunkers.py             # Recursive, Semantic, SlidingWindow, Sentence
│
├── embedding/
│   └── embedders.py            # OpenAI, HuggingFace, Cohere + disk cache
│
├── storage/
│   └── vector_store.py         # ChromaDB, pgvector, InMemory
│
├── retrieval/
│   ├── retriever.py            # Dense, BM25, Hybrid (RRF), HyDE
│   └── reranker.py             # CrossEncoder, Cohere, LLM listwise, None
│
├── generation/
│   ├── prompts.py              # RAG prompt templates
│   └── generator.py            # Claude (Anthropic), GPT (OpenAI) + streaming
│
├── rag/
│   └── pipeline.py             # Full pipeline orchestration
│
└── api/
    └── server.py               # FastAPI: /ingest /query /query/stream /health
```

---

## API Reference

```bash
# Ingest a file
curl -X POST http://localhost:8000/ingest/file \
  -F "file=@report.pdf"

# Ingest text
curl -X POST http://localhost:8000/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world document...", "source": "my_text"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the revenue?", "top_k_retrieval": 20, "top_k_final": 5}'

# Query with metadata filter
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "risks", "filter": {"file_name": "annual_report.pdf"}}'

# Stream
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the key findings"}'

# Stats
curl http://localhost:8000/documents/stats

# Health
curl http://localhost:8000/health

# Delete a source
curl -X DELETE http://localhost:8000/documents/annual_report.pdf
```
