"""Answer synthesis and claim extraction."""
from __future__ import annotations
import json
import logging
import re
from typing import Any, AsyncIterator, Dict, List, Optional

from .llm_client import LLMClient
from .models import Claim, ClaimStatus, Source

logger = logging.getLogger(__name__)

_SENSITIVITY_ADDENDUM = (
    "\n\nIMPORTANT: This topic involves personal health, legal, financial, or mental-health "
    "considerations. Frame your response with appropriate uncertainty; recommend consulting "
    "qualified professionals where the stakes are personal."
)

_SYNTHESIS_SYSTEM = """You are a precise research synthesizer. Given a query and a set of source excerpts, produce a well-structured answer that:
1. Directly addresses the query
2. Cites sources using [SOURCE_ID] notation inline
3. Distinguishes established consensus from contested or preliminary evidence
4. Is factually grounded — only claims you can trace to the provided sources
5. Uses clear, accessible language

Format your response in clean markdown with appropriate headings."""

_CLAIM_EXTRACTION_SYSTEM = """Extract distinct factual claims from the provided answer text. Return a JSON array of objects:
[
  {"id": "c1", "text": "claim text", "cited_source_ids": ["source_id_1"]},
  ...
]
Each claim must be a standalone, falsifiable statement. Omit meta-commentary, transitions, and hedges. Include any cited source IDs."""


def _build_synthesis_context(sources: List[Source]) -> str:
    parts = []
    for i, src in enumerate(sources[:20]):
        preview = src.text[:1500] if src.text else "(no text)"
        parts.append(
            f"[{src.id}] {src.title or 'Untitled'} ({src.type.value})\n"
            f"URL: {src.url}\n"
            f"Content: {preview}"
        )
    return "\n\n---\n\n".join(parts)


async def synthesize_answer(
    query: str,
    sources: List[Source],
    backend: Dict[str, Any],
    sensitivity: bool = False,
    failed_claims: Optional[List[str]] = None,
    gap_list: Optional[List[str]] = None,
) -> str:
    client = LLMClient(backend)
    context = _build_synthesis_context(sources)
    system = _SYNTHESIS_SYSTEM
    if sensitivity:
        system += _SENSITIVITY_ADDENDUM

    user_parts = [f"Query: {query}", f"\nSources:\n{context}"]

    if failed_claims:
        user_parts.append(
            f"\nThe following claims from the previous iteration need better support: "
            + "; ".join(failed_claims)
        )
    if gap_list:
        user_parts.append(
            f"\nIdentified gaps to address: " + "; ".join(gap_list)
        )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    try:
        return await client.chat(messages, temperature=0.2, max_tokens=3000)
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return f"Synthesis failed: {e}"


async def synthesize_streaming(
    query: str,
    sources: List[Source],
    backend: Dict[str, Any],
    sensitivity: bool = False,
) -> AsyncIterator[str]:
    client = LLMClient(backend)
    context = _build_synthesis_context(sources)
    system = _SYNTHESIS_SYSTEM
    if sensitivity:
        system += _SENSITIVITY_ADDENDUM

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {query}\n\nSources:\n{context}"},
    ]

    async for chunk in client.chat_stream(messages, temperature=0.2, max_tokens=3000):
        yield chunk


async def extract_claims(answer: str, backend: Dict[str, Any]) -> List[Claim]:
    client = LLMClient(backend)
    messages = [
        {"role": "system", "content": _CLAIM_EXTRACTION_SYSTEM},
        {"role": "user", "content": answer},
    ]

    try:
        raw = await client.chat(messages, temperature=0.0, max_tokens=2000, json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(raw)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and "claims" in data:
            items = data["claims"]
        else:
            items = []

        claims = []
        for item in items:
            claim = Claim(
                id=item.get("id", f"c{len(claims)+1}"),
                text=item.get("text", ""),
                cited_source_ids=item.get("cited_source_ids", []),
                status=ClaimStatus.PENDING,
            )
            claims.append(claim)
        return claims
    except Exception as e:
        logger.warning(f"Claim extraction failed: {e}")
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 30]
        return [Claim(id=f"c{i+1}", text=s) for i, s in enumerate(sentences[:10])]


def verify_citations(answer: str, source_ids: set) -> List[str]:
    """Return list of source IDs cited in answer that don't exist in source_ids."""
    cited = set(re.findall(r"\[([a-f0-9]{8,16})\]", answer))
    return [c for c in cited if c not in source_ids]


def ground_claims(claims: List[Claim], source_ids: set) -> List[Claim]:
    """Remove cited_source_ids that don't exist in the evidence pool."""
    grounded = []
    for claim in claims:
        valid_ids = [sid for sid in claim.cited_source_ids if sid in source_ids]
        grounded.append(claim.model_copy(update={"cited_source_ids": valid_ids}))
    return grounded
