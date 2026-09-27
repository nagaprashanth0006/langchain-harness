# RAG system

A configurable pipeline for loading documents, splitting text, embedding chunks, retrieving matches and generating answers. Each stage has its own module so different implementations can be compared.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

```powershell
Copy-Item .env.example .env
ollama pull nomic-embed-text
ollama pull llama3.2
python main.py --reload
```

Open http://localhost:8000/docs. Use the upload endpoint to ingest a document, then the query endpoint to ask a question. The default example configuration uses local Ollama models, Chroma and hybrid retrieval. Reranking is disabled in the example to avoid downloading another model on the first run.

The server entry point is `main.py`. `api/server.py` supplies the app factory and routes; running it directly does not start a server.

## Change the pipeline

Edit `.env` to choose a chunker, embedder, store, retriever, reranker or generator. `config.py` lists the settings and `rag/pipeline.py` connects the stages. Provider keys are only needed when you select that provider.

The flow is:

`document → chunks → embeddings → store → retrieval → reranking → answer`

Hybrid retrieval combines vector search with BM25. Cross-encoder reranking requires a model download. Optional stores may need extra packages and a running service. Stored data lives under `.data/`.

Install dependencies with `pip install -r requirements.txt`. If you need to import the modules outside this directory, also run `pip install -e .`.

This is a reference implementation. It has not been load-tested or validated for a production deployment. Evaluate answers on your own documents before relying on them. Use a single worker for the local demo, which keeps an in-process sparse index.
