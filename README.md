# LangChain harness

Experiments with LangChain, LangGraph, RAG, local models and tool calling. Some are small examples to understand a pattern. Others put those pieces into an app.

## Projects

| Folder | What it covers | Start here |
| --- | --- | --- |
| [langgraph_scaffoldings](langgraph_scaffoldings/README.md) | Document chat, vector stores and graph tools | `python main.py` |
| [agentic-exercise](agentic-exercise/README.md) | Tools, memory and graph examples | `python first-entrypoint.py` |
| [agentic-ai-learning](agentic-ai-learning/README.md) | MCP tools, routing and multi-agent loops | `python 1_agent_with_tools.py` |
| [rag-system](rag-system/README.md) | Configurable ingestion, retrieval and generation | `python main.py --reload` |
| [researcher](researcher/README.md) | Search, source grounding and repeated review | `python run.py` |
| [logging-agent](logging-agent/README.md) | Local services, logs and correlation | Docker Compose |
| [mcp-prac1](mcp-prac1/README.md) | HTTP tools and a log-analysis UI | `python run.py server` |
| [ml_scratch](ml_scratch/README.md) | Vector similarity and distance plots | Individual Python scripts |
| [nn-visualize-activation-function](nn-visualize-activation-function/README.md) | Neural-network training and activation functions | Vite UI or Tkinter app |

Run commands from the project folder, after following its setup instructions. Python projects use separate environments because their dependency versions differ. Python 3.11 is a starting point; the scaffolding package declares Python 3.12 or newer. The frontend projects use Node.js; use Node 22.12 or newer for the neural-network UI.

Ollama must be running for local inference. Pull the models listed in each README before starting. API projects share port 8000 and Vite projects share port 5173, so run one at a time or change the ports.

These are learning projects and prototypes. The RAG and research apps need model services to answer queries. Generated databases, logs, environments and installed packages stay out of Git. Copy `.env.example` where provided and keep your keys in `.env`.
