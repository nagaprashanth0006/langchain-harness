"""Start the research API and frontend."""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-frontend", action="store_true")
    args = parser.parse_args()
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not args.no_frontend and not npm:
        parser.error("Install Node.js or use --no-frontend.")
    if not args.no_frontend and not (ROOT / "frontend/node_modules").exists():
        subprocess.run([npm, "ci"], cwd=ROOT / "frontend", check=True)
    children = []
    try:
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "backend.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                cwd=ROOT,
            )
        )
        if not args.no_frontend:
            children.append(
                subprocess.Popen([npm, "run", "dev"], cwd=ROOT / "frontend")
            )
        print("API: http://localhost:8000/docs")
        if not args.no_frontend:
            print("UI: http://localhost:5173")
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        return next(
            (child.returncode for child in children if child.returncode is not None), 0
        )
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        capture_output=True,
                    )
                else:
                    child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
