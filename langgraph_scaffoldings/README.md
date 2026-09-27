# LangGraph scaffolding

Document upload and chat using Streamlit, Chroma and Ollama. The graph examples sit alongside the UI so the individual pieces are easy to inspect.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

```powershell
ollama pull llama3.1:8b
ollama pull nomic-embed-text
python main.py
```

Open http://localhost:8501, upload a text or PDF file, then ask a question about it. `python main.py` launches `streamlit_app.py`. You can also run `streamlit run streamlit_app.py` directly.

Copy `.env.example` to `.env` to change the Ollama address. The default is `http://localhost:11434`. Documents are indexed in `vector-store/`; deleting that folder means uploading them again.

`rag_chat.py` handles answers, `vector_store_operations.py` handles ingestion and retrieval, and `main_graph.py` defines the graph tools. `new_app.py` is an alternate UI experiment.

After indexing a knowledge base in the `default` collection, try the graph directly with `python main_graph.py "What does the knowledge base cover?"`.

## Docker

```powershell
docker compose up --build -d
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

The UI uses port 8501. Ollama is exposed on host port 11474. The default Compose setup runs without a GPU reservation; add one if your Docker setup supports NVIDIA GPUs. Stop it with `docker compose down`.
