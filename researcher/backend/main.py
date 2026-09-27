"FastAPI application — entry point for the research assistant backend."

from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from .capability_resolver import ResolvedTopology, probe_capabilities
from .config import get_config, load_config
from .models import OperationMode, QueryRequest, RunTrace
from .normal_mode import run_normal
from .research_graph import run_research
from .router_agent import route_query
from .source_store import evict_expired, init_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_active_queries: Dict[str, asyncio.Queue] = {}
_topology: ResolvedTopology | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _topology
    load_config()
    await init_store()
    await evict_expired()
    try:
        _topology = await probe_capabilities()
        logger.info(f"Capabilities resolved: {_topology.to_dict()}")
    except Exception as e:
        logger.warning(f"Capability probe failed: {e} — will retry on first request")
    yield


app = FastAPI(title="Research Assistant", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.get("/capabilities")
async def capabilities():
    global _topology
    if _topology is None:
        try:
            _topology = await probe_capabilities()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e))
    return _topology.to_dict()


@app.get("/config")
async def get_cfg():
    cfg = get_config()
    safe = {k: v for k, v in cfg.items() if k != "backends"}
    return safe


@app.post("/query")
async def start_query(req: QueryRequest) -> Dict[str, str]:
    query_id = str(uuid.uuid4())[:8]
    queue: asyncio.Queue = asyncio.Queue()
    _active_queries[query_id] = queue

    asyncio.create_task(_run_query(query_id, req, queue))
    return {"query_id": query_id}


@app.get("/stream/{query_id}")
async def stream(query_id: str):
    queue = _active_queries.get(query_id)
    if queue is None:
        raise HTTPException(status_code=404, detail="Query not found")

    async def event_gen():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    yield {"data": json.dumps({"type": "keepalive"})}
                    continue

                if event is None:
                    break
                yield {"data": json.dumps(event)}
                if event.get("type") in ("result", "error"):
                    break
        finally:
            _active_queries.pop(query_id, None)

    return EventSourceResponse(event_gen())


async def _run_query(query_id: str, req: QueryRequest, queue: asyncio.Queue) -> None:
    global _topology
    try:
        if _topology is None:
            await queue.put(
                {
                    "type": "progress",
                    "step": "init",
                    "message": "Probing available AI backends...",
                }
            )
            _topology = await probe_capabilities()

        await queue.put(
            {
                "type": "progress",
                "step": "route",
                "message": "Classifying query...",
            }
        )

        router_output = await route_query(req.query)

        await queue.put(
            {
                "type": "route_complete",
                "category": router_output.category.value,
                "secondary": router_output.secondary_category.value
                if router_output.secondary_category
                else None,
                "sensitivity": router_output.sensitivity,
                "eligible_sources": router_output.eligible_sources,
                "rubric": router_output.judge_rubric,
                "reasoning": router_output.reasoning,
            }
        )

        if not _topology._ollama_reachable:
            synth_model = _topology.synthesis_backend.get("model", "?")
            synth_url = _topology.synthesis_backend.get(
                "base_url", "http://localhost:11434"
            )
            await queue.put(
                {
                    "type": "warning",
                    "message": (
                        f"Ollama is not running at {synth_url}. "
                        f"Retrieval will work but synthesis will fail. "
                        f"To fix: run `ollama serve` then `ollama pull {synth_model}`."
                    ),
                }
            )

        if req.mode == OperationMode.NORMAL:
            await run_normal(req.query, router_output, _topology, queue)
        else:
            start = time.time()
            final_state = await run_research(
                query=req.query,
                depth=req.depth,
                accuracy=req.accuracy,
                router_output=router_output,
                topology=_topology,
                event_queue=queue,
            )
            elapsed = time.time() - start

            iterations_data = [
                {
                    "iteration": it.iteration,
                    "sources_found": it.sources_found,
                    "new_sources": it.new_sources,
                    "quality": it.quality,
                    "gap_list": it.gap_list[:5],
                    "quorum_passed": it.quorum_result.passed
                    if it.quorum_result
                    else False,
                }
                for it in final_state.get("iterations", [])
            ]

            quality_history = final_state.get("quality_history", [])
            final_confidence = quality_history[-1] if quality_history else 0.0

            await queue.put(
                {
                    "type": "result",
                    "answer": final_state.get("answer", ""),
                    "sources": [
                        s.to_dict() for s in final_state.get("sources", [])[:20]
                    ],
                    "iterations": iterations_data,
                    "final_confidence": final_confidence,
                    "stop_reason": final_state.get("stop_reason", ""),
                    "contested_positions": final_state.get("contested_positions", []),
                    "elapsed": elapsed,
                    "mode": "research",
                    "sensitivity": router_output.sensitivity,
                    "category": router_output.category.value,
                }
            )

    except Exception as e:
        logger.exception(f"Query {query_id} failed")
        await queue.put({"type": "error", "message": str(e)})
    finally:
        await queue.put(None)


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
