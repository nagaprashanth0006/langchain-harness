"""Open the document chat UI."""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(root / "streamlit_app.py"),
                *sys.argv[1:],
            ],
            cwd=root,
        )
    )
