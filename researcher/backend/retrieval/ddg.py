"""DuckDuckGo search provider."""
from __future__ import annotations
import asyncio
import logging
from typing import List

from ..models import Source, SourceType

logger = logging.getLogger(__name__)


async def ddg_search(query: str, max_results: int = 10) -> List[Source]:
    """Run a DDG text search and return Source stubs (no full text yet)."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        def _sync_search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))

        results = await asyncio.to_thread(_sync_search)
        sources = []
        for r in results:
            src = Source(
                origin=["ddg"],
                title=r.get("title", ""),
                text=r.get("body", ""),
                url=r.get("href", ""),
                type=SourceType.WEB,
            )
            sources.append(src)
        logger.debug(f"DDG '{query}': {len(sources)} results")
        return sources
    except Exception as e:
        logger.warning(f"DDG search failed: {e}")
        return []
