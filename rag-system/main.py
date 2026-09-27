"Canonical entry point for the RAG system."

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import logging

import uvicorn


def configure_logging(log_level: str = "INFO", json: bool = False) -> None:
    "Set up structured logging for the entire application."
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
            pass

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


import os as _os

configure_logging(
    log_level=_os.environ.get("LOG_LEVEL", "INFO"),
    json=_os.environ.get("LOG_JSON", "true").lower() in ("true", "1", "yes"),
)


from config import settings  # noqa: E402  (after sys.path + logging setup)
from api.server import create_app  # noqa: E402


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG API server",
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
        logging.warning(
            "--reload forces --workers 1; ignoring --workers %d", args.workers
        )

    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=1 if args.reload else args.workers,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )
