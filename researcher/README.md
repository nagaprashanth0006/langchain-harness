# Research assistant

A local-first research app with two modes. Normal mode retrieves sources and writes an answer. Research mode uses a LangGraph loop to check claims, look for gaps and review the answer again.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Install Node.js for the frontend. Copy `.env.example` to `.env` if you want optional provider keys, then pull the models selected in `config.yaml`:

```powershell
ollama pull qwen2.5:7b
ollama pull gemma3:12b
ollama pull phi4:14b
ollama pull nomic-embed-text
python run.py
```

The launcher installs frontend packages with `npm ci` when needed, then starts the API and UI. Open http://localhost:5173. The API docs are at http://localhost:8000/docs. Press Ctrl+C to stop both processes.

These models need enough local memory. Change the classifier, synthesis and judge models in `config.yaml` to ones your machine can run. The fallback models listed there must also be pulled if you want to use them.

For backend-only use:

```powershell
python run.py --no-frontend
```

For separate development terminals, use `python -m uvicorn backend.main:app --reload` from this folder and `npm run dev` from `frontend/`.

`config.yaml` controls models, eligible sources, review depth and stopping conditions. Optional keys belong in `.env`. Source records are cached in `source_store.db`, which is recreated locally. Search requires internet access even when inference runs locally.

The UI shows sources and progress so you can inspect the answer. Agreement between judges is a heuristic; check the cited evidence for claims that matter.
