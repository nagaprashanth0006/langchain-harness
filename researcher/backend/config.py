from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_config: Optional[Dict[str, Any]] = None


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    global _config
    target = path or _CONFIG_PATH
    with open(target, "r") as f:
        _config = yaml.safe_load(f)
    _apply_env_overrides(_config)
    return _config


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _apply_env_overrides(cfg: Dict[str, Any]) -> None:
    backends = cfg.get("backends", {})
    judges = backends.get("judges", {})

    for cloud in judges.get("cloud", []):
        provider = cloud.get("provider", "")
        if provider == "groq":
            key = os.environ.get("GROQ_API_KEY") or cloud.get("api_key")
            if key:
                cloud["api_key"] = key
        elif provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY") or cloud.get("api_key")
            if key:
                cloud["api_key"] = key

    ds = cfg.get("datasources", {})
    if not ds.get("pubmed", {}).get("api_key"):
        ds.setdefault("pubmed", {})["api_key"] = os.environ.get("PUBMED_API_KEY")
    if not ds.get("unpaywall", {}).get("email"):
        ds.setdefault("unpaywall", {})["email"] = os.environ.get(
            "UNPAYWALL_EMAIL"
        ) or ds.get("openalexs", {}).get("email")


def get_slider_params(depth: int, accuracy: int) -> Dict[str, Any]:
    cfg = get_config()
    sm = cfg.get("slider_mappings", {})

    def _get(mapping: dict, key: int) -> dict:
        return mapping.get(key) or mapping.get(str(key)) or {}

    d_params = _get(sm.get("depth", {}), depth)
    a_params = _get(sm.get("accuracy", {}), accuracy)
    return {**d_params, **a_params}


def get_eligible_sources(category: str) -> List[str]:
    cfg = get_config()
    eligibility = cfg.get("datasources", {}).get("eligibility", {})
    return eligibility.get(category, ["ddg"])


def get_synthesis_backend() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("backends", {}).get("synthesis", {})


def get_classifier_backend() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get("backends", {}).get("classifier", {})


def get_judge_candidates() -> List[Dict[str, Any]]:
    cfg = get_config()
    judges = cfg.get("backends", {}).get("judges", {})
    local = judges.get("local", [])
    cloud = [j for j in judges.get("cloud", []) if j.get("api_key")]
    return local + cloud


def get_cache_config() -> Dict[str, Any]:
    cfg = get_config()
    return cfg.get(
        "cache", {"ttl_days": 30, "max_size_mb": 500, "db_path": "source_store.db"}
    )
