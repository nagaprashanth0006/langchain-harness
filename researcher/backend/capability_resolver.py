from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx

from .config import get_config, get_judge_candidates

logger = logging.getLogger(__name__)


class ResolvedTopology:
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"
    SEQUENTIAL = "sequential"

    def __init__(
        self,
        mode: str,
        synthesis_backend: Dict[str, Any],
        judge_pool: List[Dict[str, Any]],
        classifier_backend: Dict[str, Any],
        ollama_reachable: bool = False,
    ):
        self.mode = mode
        self.synthesis_backend = synthesis_backend
        self.judge_pool = judge_pool
        self.classifier_backend = classifier_backend
        self._ollama_reachable = ollama_reachable

    def to_dict(self) -> Dict[str, Any]:
        synth_url = self.synthesis_backend.get("base_url", "")
        return {
            "topology": self.mode,
            "synthesis": {
                "model": self.synthesis_backend.get("model"),
                "base_url": synth_url,
                "reachable": self._ollama_reachable,
            },
            "judges": [
                {
                    "id": j.get("id"),
                    "model": j.get("model"),
                    "provider": j.get("provider", "ollama"),
                }
                for j in self.judge_pool
            ],
            "classifier": {
                "model": self.classifier_backend.get("model"),
                "base_url": self.classifier_backend.get("base_url"),
            },
            "ollama_reachable": self._ollama_reachable,
            "setup_hint": (
                None
                if self._ollama_reachable
                else f"Ollama not running at {synth_url}. Run: ollama serve && ollama pull {self.synthesis_backend.get('model', 'qwen2.5:14b')}"
            ),
        }


async def _probe_ollama(base_url: str, timeout: float = 3.0) -> Optional[List[str]]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base_url}/api/tags")
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return None


async def _model_available(
    base_url: str, model: str, available_models: Optional[List[str]]
) -> bool:
    if available_models is None:
        return False
    model_base = model.split(":")[0].lower()
    for m in available_models:
        if m.lower().startswith(model_base):
            return True
    return False


async def _pick_available_model(
    backend: Dict[str, Any], available: Optional[List[str]]
) -> Optional[str]:
    for candidate in [backend.get("model"), backend.get("fallback_model")]:
        if not candidate:
            continue
        if await _model_available(backend.get("base_url", ""), candidate, available):
            return candidate
    return None


async def probe_capabilities() -> ResolvedTopology:
    cfg = get_config()
    backends = cfg.get("backends", {})
    synth_cfg = backends.get("synthesis", {})
    class_cfg = backends.get("classifier", {})

    primary_url = synth_cfg.get("base_url", "http://localhost:11434")
    available_models = await _probe_ollama(primary_url)

    if available_models is None:
        logger.warning(
            f"Ollama not reachable at {primary_url}. "
            "Start Ollama with `ollama serve`, then pull models via `ollama pull <model>`. "
            "Synthesis and classification will fail until a backend is available."
        )

    synth_model = await _pick_available_model(synth_cfg, available_models)
    if synth_model:
        synth_cfg = {**synth_cfg, "model": synth_model}
    elif available_models:
        synth_cfg = {**synth_cfg, "model": available_models[0]}

    class_model = await _pick_available_model(class_cfg, available_models)
    if class_model:
        class_cfg = {**class_cfg, "model": class_model}
    elif available_models:
        class_cfg = {**class_cfg, "model": available_models[0]}

    candidates = get_judge_candidates()
    live_judges: List[Dict[str, Any]] = []

    for judge in candidates:
        provider = judge.get("provider", "ollama")
        if provider == "ollama" or "base_url" in judge:
            judge_url = judge.get("base_url", primary_url)
            if judge_url == primary_url:
                j_available = available_models
            else:
                j_available = await _probe_ollama(judge_url)

            chosen = await _pick_available_model(judge, j_available)
            if chosen:
                live_judges.append({**judge, "model": chosen, "available": True})
        else:
            if judge.get("api_key"):
                live_judges.append({**judge, "available": True})

    if len(live_judges) >= 3:
        local_count = sum(
            1
            for j in live_judges
            if not j.get("provider") or j.get("provider") == "ollama"
        )
        mode = (
            ResolvedTopology.DISTRIBUTED
            if local_count >= 3
            else ResolvedTopology.HYBRID
        )
    elif len(live_judges) >= 1:
        mode = ResolvedTopology.HYBRID
    else:
        if available_models:
            fallback_judge = {
                "id": "fallback-1",
                "base_url": primary_url,
                "model": available_models[0],
                "available": True,
            }
            live_judges.append(fallback_judge)
        mode = ResolvedTopology.SEQUENTIAL

    ollama_ok = available_models is not None
    logger.info(
        f"Capability resolve: topology={mode}, judges={len(live_judges)}, synth={synth_cfg.get('model')}, ollama={ollama_ok}"
    )
    return ResolvedTopology(
        mode=mode,
        synthesis_backend=synth_cfg,
        judge_pool=live_judges,
        classifier_backend=class_cfg,
        ollama_reachable=ollama_ok,
    )
