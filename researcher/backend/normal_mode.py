"""Normal mode: fast single-pass DDG → fetch → synthesise."""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List

from .capability_resolver import ResolvedTopology
from .models import RouterOutput, Source
from .retrieval.ddg import ddg_search
from .retrieval.web_fetcher import enrich_sources
from .source_store import deduplicate, upsert_source
from .synthesis import synthesize_streaming

logger = logging.getLogger(__name__)


def _make_queries(query: str) -> List[str]:
    return [query, f"{query} overview explanation"]


async def run_normal(
    query: str,
    router_output: RouterOutput,
    topology: ResolvedTopology,
    event_queue: asyncio.Queue,
) -> Dict[str, Any]:
    start = time.time()

    await event_queue.put({
        "type": "progress",
        "step": "search",
        "message": "Searching the web...",
    })

    queries = _make_queries(query)
    all_sources: List[Source] = []
    for q in queries:
        results = await ddg_search(q, max_results=8)
        all_sources.extend(results)

    await event_queue.put({
        "type": "progress",
        "step": "dedup",
        "message": f"Found {len(all_sources)} results, deduplicating...",
    })

    deduped, _ = await deduplicate(all_sources)

    to_enrich = [s for s in deduped if not s.text or len(s.text) < 200][:5]
    if to_enrich:
        await event_queue.put({
            "type": "progress",
            "step": "enrich",
            "message": f"Fetching full text for {len(to_enrich)} pages...",
        })
        enriched = await enrich_sources(to_enrich, max_concurrent=4)
        for orig, enr in zip(to_enrich, enriched):
            idx = next((i for i, s in enumerate(deduped) if s.url == orig.url), None)
            if idx is not None:
                deduped[idx] = enr

    for src in deduped:
        await upsert_source(src)

    for src in deduped[:8]:
        await event_queue.put({"type": "source_found", "source": src.to_dict()})

    await event_queue.put({
        "type": "progress",
        "step": "synthesize",
        "message": "Generating answer...",
    })

    synth_backend = topology.synthesis_backend
    await event_queue.put({"type": "synthesis_start"})

    full_answer = []
    async for chunk in synthesize_streaming(
        query=query,
        sources=deduped[:10],
        backend=synth_backend,
        sensitivity=router_output.sensitivity,
    ):
        full_answer.append(chunk)
        await event_queue.put({"type": "synthesis_chunk", "chunk": chunk})

    answer = "".join(full_answer)

    elapsed = time.time() - start
    result = {
        "answer": answer,
        "sources": [s.to_dict() for s in deduped[:10]],
        "elapsed": elapsed,
        "mode": "normal",
        "sensitivity": router_output.sensitivity,
        "category": router_output.category.value,
    }

    await event_queue.put({"type": "result", **result})
    return result
