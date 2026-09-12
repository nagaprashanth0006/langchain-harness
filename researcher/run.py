"""Convenience launcher: starts backend + optionally opens frontend dev server."""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent


def start_backend():
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT,
    )


def start_frontend():
    fe = ROOT / "frontend"
    if not (fe / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=fe, check=True)
    return subprocess.Popen(["npm", "run", "dev"], cwd=fe)


if __name__ == "__main__":
    print("Starting Research Assistant...")
    backend = start_backend()
    time.sleep(2)

    frontend = None
    if "--no-frontend" not in sys.argv:
        frontend = start_frontend()
        time.sleep(3)
        webbrowser.open("http://localhost:5173")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        if frontend:
            frontend.terminate()
