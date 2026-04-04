"""
Canonical entry point for the production RAG system.

Installation (run once)
────────────────────────
    cd rag-system
    pip install -e .            # installs the project as an editable package
    cp .env.example .env        # configure your settings

Running the server
───────────────────
Development (single worker, auto-reload on code changes):
    python main.py --reload

Production (multiple workers, no reload):
    python main.py --workers 4

Via uvicorn directly (full control over all options):
    uvicorn api.server:create_app --factory --host 0.0.0.0 --port 8000 --workers 4

Via gunicorn (recommended for production — handles worker recycling, SIGTERM):
    gunicorn 'api.server:create_app()' \\
        -k uvicorn.workers.UvicornWorker \\
        -w 4 \\
        --timeout 120 \\
        --graceful-timeout 30

Why ``--factory`` / factory form?
───────────────────────────────────
uvicorn's multi-worker mode (--workers N) forks N subprocesses.  Each subprocess
must import the module and create its own FastAPI + pipeline instance.  If we
put ``app = create_app()`` at module level:

  • In the parent process, create_app() runs once (fine).
  • Each child imports the module again, running create_app() AGAIN.
  • BUT — the module-level call in the parent also triggers lifespan(), which
    opens connections that cannot be shared across fork boundaries (file handles,
    socket connections to ChromaDB, etc.).

Using the factory form (``--factory`` flag or passing the callable as a string)
defers create_app() to each child process, after forking, so every worker gets
its own clean pipeline instance.  This is the correct production pattern.

Why logging is configured before create_app()?
───────────────────────────────────────────────
create_app() registers routes and returns immediately — it does not start the
server or trigger lifespan.  However, importing modules like ``config`` can
trigger module-level log calls.  Configuring logging first ensures those calls
land in the correct format/handler from the start.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so all modules are importable.
# This is a fallback guard; after ``pip install -e .`` it is not needed.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import logging

import uvicorn


# ---------------------------------------------------------------------------
# Logging — must be configured BEFORE importing local modules that log at
# import time, and before create_app() which may trigger pipeline imports.
# ---------------------------------------------------------------------------

def configure_logging(log_level: str = "INFO", json: bool = False) -> None:
    """
    Set up structured logging for the entire application.

    In production, use JSON format so log aggregators (Datadog, Splunk,
    CloudWatch) can parse and index fields automatically.

    In development, use plain text for readability.

    Why not use uvicorn's --log-config?
    uvicorn's log config overrides only uvicorn's own loggers.  Configuring
    the root logger here ensures all application loggers (ingestion, retrieval,
    generation, etc.) use the same format and level.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    if json:
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler()
            handler.setFormatter(
                jsonlogger.JsonFormatter(
                    "%(asctime)s %(name)s %(levelname)s %(message)s"
                )
            )
            logging.basicConfig(level=level, handlers=[handler], force=True)
            return
        except ImportError:
            pass  # Fall through to plain-text

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,  # Override any earlier basicConfig calls
    )


# ---------------------------------------------------------------------------
# Build the ASGI app
# ---------------------------------------------------------------------------
# Logging is configured first so that any log calls during module import
# (e.g., pydantic-settings loading .env) use the correct handlers.
#
# We read log settings from environment directly rather than importing
# ``settings`` here, because importing ``config`` triggers pydantic-settings
# which reads .env — and we want logging ready before that happens.

import os as _os
configure_logging(
    log_level=_os.environ.get("LOG_LEVEL", "INFO"),
    json=_os.environ.get("LOG_JSON", "true").lower() in ("true", "1", "yes"),
)

# Now safe to import local modules
from config import settings          # noqa: E402  (after sys.path + logging setup)
from api.server import create_app    # noqa: E402

# ---------------------------------------------------------------------------
# Module-level app object
# ---------------------------------------------------------------------------
# Exposed as ``main:app`` so uvicorn can import it with:
#   uvicorn main:app --host 0.0.0.0 --port 8000
#
# For multi-worker production, prefer the factory form to avoid
# fork-before-connect issues (see module docstring):
#   uvicorn api.server:create_app --factory --workers 4
# ---------------------------------------------------------------------------
app = create_app()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Production RAG API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=settings.api_host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.api_port, help="Bind port")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes.  Use 1 for development. "
            "For production, set to (2 × CPU cores) + 1.  "
            "Incompatible with --reload."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on source changes.  Development only; forces --workers 1.",
    )
    args = parser.parse_args()

    if args.reload and args.workers > 1:
        logging.warning("--reload forces --workers 1; ignoring --workers %d", args.workers)

    uvicorn.run(
        # Use the string form so uvicorn can re-import in each worker process.
        # If we pass the ``app`` object directly, it is pickled — which breaks
        # connections held by the app (ChromaDB file handle, model objects, etc.).
        "main:app",
        host=args.host,
        port=args.port,
        workers=1 if args.reload else args.workers,
        reload=args.reload,
        # Pass log_level to uvicorn's access logger; application logs are
        # already handled by configure_logging() above.
        log_level=settings.log_level.lower(),
    )
