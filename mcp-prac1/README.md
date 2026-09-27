# Log-analysis tools

An early HTTP tool-server experiment with a Streamlit UI. Despite the folder name, this server uses a custom `/call` endpoint; it does not implement the MCP protocol. The protocol-based example is in `agentic-ai-learning`.

Use a separate environment for this project. From this folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`.

Start Ollama and pull `mistral`. Start the logging lab if you want sample Loki data.

In one terminal:

```powershell
python run.py server
```

In a second terminal, with the same environment activated:

```powershell
python run.py ui
```

The tool server uses http://localhost:8000 and the UI uses http://localhost:8501. Enter the Loki address and a query in the UI. The defaults expect Loki at http://localhost:3100.

The server provides log fetching, local inference and a generated quote. Unrelated dividend-scraping experiments have been removed from this copy.
