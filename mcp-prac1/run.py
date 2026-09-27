"""Run the HTTP tool server or the log-analysis UI."""

import argparse
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=["server", "ui"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    command = (
        ["uvicorn", "mcp.mcp_server:app", "--host", "127.0.0.1", "--port", "8000"]
        if args.service == "server"
        else ["streamlit", "run", "app.py"]
    )
    raise SystemExit(subprocess.call([sys.executable, "-m", *command], cwd=root))
