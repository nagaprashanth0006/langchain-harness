# Agentic AI examples

Four small scripts showing how tool calls turn into routing, task orchestration and a review loop. Ollama runs the models; an MCP server provides search and calculation tools.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

```powershell
Copy-Item .env.example .env
ollama pull llama3.2
python 1_agent_with_tools.py
```

Set `TAVILY_API_KEY` in `.env` for web search. Calculation works without it. Each script starts its own MCP server.

Run the examples in order:

1. `1_agent_with_tools.py` — one agent calling tools.
2. `2_multi_agent_router.py` — route a question to a specialist.
3. `3_multi_agent_orchestrator.py` — split a question into tasks and combine the results.
4. `4_multi_agent_loop.py` — evaluate the answer and try again.

The prompts are in the scripts. Change them to try a different task. The calculator evaluates model-provided expressions; keep this a local learning example.
