"""Multi-judge quorum panel with structured verdicts."""
from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient
from .models import Claim, JudgeVerdict, QuorumResult, Source

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_CORROBORATION = """You are an expert research judge evaluating an answer for factual accuracy and source corroboration.

Evaluate the provided answer against the evidence. Return a JSON object:
{
  "target": "<claim_id or 'answer'>",
  "score": <0.0-1.0 float>,
  "gaps": ["gap 1", "gap 2"],
  "reasoning": "brief explanation"
}

Scoring rubric:
- 0.9+: All claims strongly corroborated by independent sources
- 0.7-0.9: Most claims corroborated, minor gaps
- 0.5-0.7: Some claims lack corroboration or sources are weak
- <0.5: Major unsupported claims or contradictions

For gaps, be specific: name the uncorroborated claim, weak source, or unanswered sub-question."""

_JUDGE_SYSTEM_REASONING_QUALITY = """You are an expert judge evaluating reasoning quality and perspective balance.

Evaluate whether the response:
1. Reasons logically and coherently
2. Represents major viewpoints fairly
3. Acknowledges complexity and nuance
4. Avoids false dichotomies

Return a JSON object:
{
  "target": "<claim_id or 'answer'>",
  "score": <0.0-1.0 float>,
  "gaps": ["gap 1", "gap 2"],
  "reasoning": "brief explanation"
}

Scoring: 0.9+ excellent reasoning/balance; 0.7-0.9 good with minor issues; 0.5-0.7 moderate issues; <0.5 poor."""

_JUDGE_SYSTEM_SENSITIVITY = """You are an expert judge evaluating responses on sensitive personal topics (health, legal, financial, mental health).

Evaluate whether the response:
1. Is appropriately hedged — acknowledges uncertainty
2. Recommends professional consultation where warranted
3. Balances perspectives without being alarmist or dismissive
4. Would not cause harm to a vulnerable reader

Return a JSON object:
{
  "target": "<claim_id or 'answer'>",
  "score": <0.0-1.0 float>,
  "gaps": ["gap 1", "gap 2"],
  "reasoning": "brief explanation"
}"""

_RUBRIC_TO_SYSTEM = {
    "corroboration": _JUDGE_SYSTEM_CORROBORATION,
    "reasoning_quality": _JUDGE_SYSTEM_REASONING_QUALITY,
    "sensitivity_balanced": _JUDGE_SYSTEM_SENSITIVITY,
}


def _build_judge_prompt(
    query: str,
    answer: str,
    sources: List[Source],
    target: str,
    claim_text: Optional[str] = None,
) -> str:
    sources_summary = "\n".join(
        f"[{s.id}] {s.title} — {s.text[:400]}" for s in sources[:10]
    )
    prompt = f"Query: {query}\n\nAnswer:\n{answer}\n\nEvidence pool:\n{sources_summary}"
    if target != "answer" and claim_text:
        prompt += f"\n\nSpecific claim to evaluate (id={target}): {claim_text}"
    return prompt


async def _run_judge(
    judge_cfg: Dict[str, Any],
    query: str,
    answer: str,
    sources: List[Source],
    rubric: str,
    target: str = "answer",
    claim_text: Optional[str] = None,
) -> Optional[JudgeVerdict]:
    system = _RUBRIC_TO_SYSTEM.get(rubric, _JUDGE_SYSTEM_CORROBORATION)
    client = LLMClient(judge_cfg)
    prompt = _build_judge_prompt(query, answer, sources, target, claim_text)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    try:
        raw = await client.chat(messages, temperature=0.1, max_tokens=1024, json_mode=True)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(raw)
        return JudgeVerdict(
            judge_id=judge_cfg.get("id", "unknown"),
            model=judge_cfg.get("model", "unknown"),
            target=target,
            score=float(data.get("score", 0.5)),
            gaps=data.get("gaps", []),
            rubric=rubric,
            reasoning=data.get("reasoning", ""),
        )
    except Exception as e:
        logger.warning(f"Judge {judge_cfg.get('id')} failed: {e}")
        return None


async def run_quorum_whole_answer(
    query: str,
    answer: str,
    sources: List[Source],
    judge_pool: List[Dict[str, Any]],
    rubric: str,
    threshold: float = 0.75,
    min_judges: int = 3,
) -> QuorumResult:
    active_judges = judge_pool[:max(min_judges, len(judge_pool))]
    tasks = [
        _run_judge(j, query, answer, sources, rubric, "answer")
        for j in active_judges
    ]
    verdicts_raw = await asyncio.gather(*tasks, return_exceptions=True)
    verdicts = [v for v in verdicts_raw if isinstance(v, JudgeVerdict)]

    if not verdicts:
        return QuorumResult(
            target="answer",
            passed=False,
            agreement_ratio=0.0,
            composite_quality=0.0,
            contested=False,
            verdicts=[],
            gap_list=["All judges failed — check model availability"],
            mean_score=0.0,
        )

    scores = [v.score for v in verdicts]
    mean_score = sum(scores) / len(scores)
    passing = sum(1 for s in scores if s >= threshold)
    agreement_ratio = passing / len(scores)
    passed = agreement_ratio >= threshold

    gap_list = []
    seen_gaps = set()
    for v in verdicts:
        for g in v.gaps:
            if g not in seen_gaps:
                gap_list.append(g)
                seen_gaps.add(g)

    coverage = min(1.0, len(sources) / 5.0)
    independent_origins = set()
    for src in sources:
        independent_origins.update(src.origin)
    corroboration = min(1.0, len(independent_origins) / 3.0)
    composite = coverage * corroboration * mean_score

    score_variance = max(scores) - min(scores) if len(scores) > 1 else 0.0
    contested = score_variance > 0.4 or (not passed and mean_score > 0.3)

    return QuorumResult(
        target="answer",
        passed=passed,
        agreement_ratio=agreement_ratio,
        composite_quality=composite,
        contested=contested,
        verdicts=verdicts,
        gap_list=gap_list,
        mean_score=mean_score,
    )


async def run_quorum_per_claim(
    query: str,
    answer: str,
    sources: List[Source],
    claims: List[Claim],
    judge_pool: List[Dict[str, Any]],
    rubric: str,
    threshold: float = 0.75,
    min_judges: int = 3,
) -> List[QuorumResult]:
    results = []
    active_judges = judge_pool[:max(min_judges, len(judge_pool))]

    for claim in claims:
        tasks = [
            _run_judge(j, query, answer, sources, rubric, claim.id, claim.text)
            for j in active_judges
        ]
        verdicts_raw = await asyncio.gather(*tasks, return_exceptions=True)
        verdicts = [v for v in verdicts_raw if isinstance(v, JudgeVerdict)]

        if not verdicts:
            results.append(QuorumResult(
                target=claim.id,
                passed=False,
                agreement_ratio=0.0,
                composite_quality=0.0,
                contested=False,
                verdicts=[],
                gap_list=["Judges unavailable"],
                mean_score=0.0,
            ))
            continue

        scores = [v.score for v in verdicts]
        mean_score = sum(scores) / len(scores)
        passing = sum(1 for s in scores if s >= threshold)
        agreement_ratio = passing / len(scores)
        passed = agreement_ratio >= threshold

        gap_list = list({g for v in verdicts for g in v.gaps})
        score_variance = max(scores) - min(scores) if len(scores) > 1 else 0.0
        contested = score_variance > 0.4

        coverage = min(1.0, len(claim.cited_source_ids) / 2.0)
        composite = coverage * mean_score

        results.append(QuorumResult(
            target=claim.id,
            passed=passed,
            agreement_ratio=agreement_ratio,
            composite_quality=composite,
            contested=contested,
            verdicts=verdicts,
            gap_list=gap_list,
            mean_score=mean_score,
        ))

    return results
