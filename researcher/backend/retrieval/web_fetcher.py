"""Fetch and extract clean text from web pages."""
from __future__ import annotations
import asyncio
import logging
from typing import List, Optional

import httpx

from ..models import Source

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ResearchAssistant/1.0)",
    "Accept": "text/html,application/xhtml+xml",
}
_TIMEOUT = 15
_MAX_CONTENT_LEN = 50_000


async def fetch_page(url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers=_HEADERS,
        ) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            ct = resp.headers.get("content-type", "")
            if "html" not in ct and "text" not in ct:
                return None
            return resp.text[:_MAX_CONTENT_LEN]
    except Exception as e:
        logger.debug(f"Fetch failed {url}: {e}")
        return None


def _extract_text(html: str, url: str = "") -> Optional[str]:
    try:
        import trafilatura
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
        return text
    except ImportError:
        pass

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:_MAX_CONTENT_LEN]
    except Exception:
        return None


async def enrich_source(source: Source) -> Source:
    if not source.url or len(source.text) > 500:
        return source
    html = await fetch_page(source.url)
    if not html:
        return source
    text = _extract_text(html, source.url)
    if text and len(text) > len(source.text):
        source = source.model_copy(update={"text": text})
    return source


async def enrich_sources(sources: List[Source], max_concurrent: int = 5) -> List[Source]:
    sem = asyncio.Semaphore(max_concurrent)

    async def _enrich(src: Source) -> Source:
        async with sem:
            return await enrich_source(src)

    tasks = [_enrich(s) for s in sources]
    return list(await asyncio.gather(*tasks, return_exceptions=False))
