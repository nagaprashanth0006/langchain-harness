# Agent exercises

Small LangGraph examples for tool calling, search, memory and retrieval. Start with the simple agent before trying the larger scripts.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

```powershell
Copy-Item .env.example .env
ollama pull llama3
python first-entrypoint.py
```

`first-entrypoint.py` uses a calculator, a small dictionary-based knowledge tool and a Python tool. It runs a sample question and prints the answer. The Python tool executes model-generated code, so run examples in an environment you can discard.

Other entry points:

- `python langgraph_agents/agent-with-tools-memory.py` — tools with in-memory checkpoints; pull `llama3.1:8b` first.
- `python langgraph_agents/prac1.py` — a small state graph with arithmetic tools.
- `python lean_with_tools.py` — a minimal tool loop; pull `llama3.1:8b-instruct-q4_K_M` first.
- `python pluggable-tools.py --help` — options for the larger example.

Search examples need `TAVILY_API_KEY` in `.env`. Retrieval examples need `nomic-embed-text` in Ollama and documents in `data/`. Vector stores and checkpoints are created locally. Model names vary between examples; check the selected script before running it.
