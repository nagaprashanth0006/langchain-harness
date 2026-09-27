"LangGraph state machine for research mode."

from __future__ import annotations
import asyncio
import logging
import time
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from .capability_resolver import ResolvedTopology
from .config import get_slider_params
from .models import (
    Claim,
    ClaimStatus,
    IterationResult,
    QuorumResult,
    RouterOutput,
    Source,
)
from .quorum import run_quorum_per_claim, run_quorum_whole_answer
from .retrieval.academic import retrieve_from_academic
from .retrieval.ddg import ddg_search
from .retrieval.web_fetcher import enrich_sources
from .source_store import deduplicate, upsert_source
from .synthesis import extract_claims, ground_claims, synthesize_answer

logger = logging.getLogger(__name__)


class ResearchState(TypedDict):
    query: str
    depth: int
    accuracy: int
    router_output: RouterOutput
    topology: ResolvedTopology

    queries: List[str]
    sources: List[Source]
    source_ids: List[str]
    claims: List[Claim]
    answer: str

    iterations: List[IterationResult]
    quality_history: List[float]
    gap_list: List[str]
    failed_claim_ids: List[str]

    iteration: int
    max_iterations: int
    threshold: float
    judge_count: int
    min_corroborating: int
    granularity: str

    done: bool
    stop_reason: str
    contested_positions: List[Dict[str, Any]]
    events: List[Dict[str, Any]]

    start_time: float


def _emit(state: ResearchState, event_type: str, **data) -> None:
    state["events"].append({"type": event_type, **data})


def _build_initial_queries(
    query: str, router_output: RouterOutput, depth: int
) -> List[str]:
    base_queries = [query]
    if depth >= 3:
        base_queries.append(f"{query} systematic review")
        base_queries.append(f"{query} recent studies")
    if depth >= 4:
        base_queries.append(f"{query} meta-analysis")
    return base_queries[: max(1, depth)]


async def retrieve_node(state: ResearchState) -> Dict[str, Any]:
    router = state["router_output"]
    eligible = router.eligible_sources
    queries = state["queries"]
    existing_ids = set(state["source_ids"])

    _emit(
        state,
        "progress",
        step="retrieve",
        message=f"Searching {len(queries)} queries across {len(eligible)} source types...",
    )

    all_new: List[Source] = []

    if "ddg" in eligible:
        for q in queries[:3]:
            results = await ddg_search(q, max_results=8)
            all_new.extend(results)

    academic_eligible = [s for s in eligible if s != "ddg"]
    if academic_eligible:
        for q in queries[:2]:
            results = await retrieve_from_academic(
                q, academic_eligible, max_per_source=4
            )
            all_new.extend(results)

    if "ddg" in eligible and state["depth"] >= 2:
        ddg_srcs = [s for s in all_new if "ddg" in s.origin]
        to_enrich = [s for s in ddg_srcs if not s.text or len(s.text) < 200][:5]
        if to_enrich:
            _emit(
                state,
                "progress",
                step="enrich",
                message=f"Fetching full text for {len(to_enrich)} pages...",
            )
            enriched = await enrich_sources(to_enrich, max_concurrent=4)
            url_to_enriched = {e.url: e for e in enriched}
            all_new = [url_to_enriched.get(s.url, s) for s in all_new]

    deduped, _ = await deduplicate(all_new)
    new_unique = [s for s in deduped if s.id not in existing_ids]

    for src in new_unique:
        await upsert_source(src)

    _emit(
        state,
        "progress",
        step="dedup",
        message=f"Found {len(new_unique)} new unique sources (total: {len(existing_ids) + len(new_unique)})",
    )

    for src in new_unique:
        _emit(state, "source_found", source=src.to_dict())

    updated_sources = state["sources"] + new_unique
    updated_ids = list(existing_ids | {s.id for s in new_unique})

    return {
        "sources": updated_sources,
        "source_ids": updated_ids,
        "events": state["events"],
    }


async def synthesize_node(state: ResearchState) -> Dict[str, Any]:
    _emit(
        state,
        "progress",
        step="synthesize",
        message="Synthesising answer from evidence...",
    )

    topology = state["topology"]
    router = state["router_output"]
    failed_ids = set(state["failed_claim_ids"])

    failed_texts = (
        [c.text for c in state["claims"] if c.id in failed_ids] if failed_ids else []
    )
    gap_list = state["gap_list"]

    answer = await synthesize_answer(
        query=state["query"],
        sources=state["sources"],
        backend=topology.synthesis_backend,
        sensitivity=router.sensitivity,
        failed_claims=failed_texts or None,
        gap_list=gap_list or None,
    )

    _emit(state, "synthesis_complete", answer=answer)

    claims = await extract_claims(answer, topology.synthesis_backend)
    grounded = ground_claims(claims, set(state["source_ids"]))

    return {"answer": answer, "claims": grounded, "events": state["events"]}


async def judge_node(state: ResearchState) -> Dict[str, Any]:
    topology = state["topology"]
    router = state["router_output"]
    granularity = state["granularity"]
    threshold = state["threshold"]
    judge_count = state["judge_count"]

    active_judges = topology.judge_pool[: max(1, judge_count)]
    if not active_judges:
        active_judges = [topology.synthesis_backend]

    _emit(
        state,
        "progress",
        step="quorum",
        message=f"Running quorum ({len(active_judges)} judges, granularity={granularity})...",
    )

    if granularity == "per_claim" and state["claims"]:
        quorum_results = await run_quorum_per_claim(
            query=state["query"],
            answer=state["answer"],
            sources=state["sources"],
            claims=state["claims"],
            judge_pool=active_judges,
            rubric=router.judge_rubric,
            threshold=threshold,
            min_judges=min(2, len(active_judges)),
        )
        overall_score = (
            sum(r.composite_quality for r in quorum_results) / len(quorum_results)
            if quorum_results
            else 0.0
        )
        all_gaps = list({g for r in quorum_results for g in r.gap_list})
        failed_ids = [
            state["claims"][i].id
            for i, r in enumerate(quorum_results)
            if i < len(state["claims"]) and not r.passed
        ]
        top_result = quorum_results[0] if quorum_results else None
        contested = any(r.contested for r in quorum_results)
    else:
        top_result = await run_quorum_whole_answer(
            query=state["query"],
            answer=state["answer"],
            sources=state["sources"],
            judge_pool=active_judges,
            rubric=router.judge_rubric,
            threshold=threshold,
            min_judges=min(2, len(active_judges)),
        )
        quorum_results = [top_result]
        overall_score = top_result.composite_quality
        all_gaps = top_result.gap_list
        failed_ids = [] if top_result.passed else ["answer"]
        contested = top_result.contested

    for r in quorum_results:
        _emit(
            state,
            "judge_result",
            target=r.target,
            passed=r.passed,
            score=r.mean_score,
            agreement=r.agreement_ratio,
            gaps=r.gap_list[:3],
        )

    prev_sources = state["iterations"][-1].sources_found if state["iterations"] else 0
    iteration_result = IterationResult(
        iteration=state["iteration"],
        queries_used=state["queries"],
        sources_found=len(state["sources"]),
        new_sources=len(state["sources"]) - prev_sources,
        synthesis_excerpt=state["answer"][:300],
        quorum_result=top_result,
        quality=overall_score,
        gap_list=all_gaps,
    )

    contested_positions = []
    if contested and all_gaps:
        contested_positions = [
            {"aspect": g, "support": "partial", "contested": True} for g in all_gaps[:3]
        ]

    _emit(
        state,
        "iteration_complete",
        iteration=state["iteration"],
        quality=overall_score,
        done=False,
        reason="",
    )

    return {
        "iterations": state["iterations"] + [iteration_result],
        "quality_history": state["quality_history"] + [overall_score],
        "gap_list": all_gaps,
        "failed_claim_ids": failed_ids,
        "contested_positions": contested_positions,
        "events": state["events"],
    }


def _check_plateau(
    quality_history: List[float], window: int = 3, delta: float = 0.05
) -> bool:
    if len(quality_history) < window:
        return False
    recent = quality_history[-window:]
    return (max(recent) - min(recent)) < delta


def check_termination_node(state: ResearchState) -> Dict[str, Any]:
    quality_history = state["quality_history"]
    current_quality = quality_history[-1] if quality_history else 0.0
    iteration = state["iteration"]
    max_iter = state["max_iterations"]
    threshold = state["threshold"]
    gap_list = state["gap_list"]

    passed = current_quality >= threshold
    at_ceiling = iteration >= max_iter
    plateau = _check_plateau(quality_history)
    no_gaps = not gap_list

    if passed:
        reason = "quorum_met"
    elif at_ceiling:
        reason = "ceiling_reached"
    elif plateau and iteration >= 3:
        reason = "plateau"
    elif no_gaps and iteration >= 2:
        reason = "no_gaps_to_fill"
    else:
        reason = "continue"

    done = reason != "continue"

    if state["events"]:
        last = state["events"][-1]
        if last.get("type") == "iteration_complete":
            state["events"][-1] = {**last, "done": done, "reason": reason}

    return {
        "done": done,
        "stop_reason": reason,
        "iteration": iteration + 1,
        "events": state["events"],
    }


async def refine_node(state: ResearchState) -> Dict[str, Any]:
    gap_list = state["gap_list"]
    base = state["query"]

    if not gap_list:
        return {"done": True, "stop_reason": "no_gaps", "queries": [base]}

    new_queries = [base]
    for gap in gap_list[:3]:
        new_queries.append(f"{base} {gap}")

    _emit(
        state,
        "progress",
        step="refine",
        message=f"Refining with {len(new_queries)} targeted queries based on {len(gap_list)} gaps...",
    )

    return {"queries": new_queries, "events": state["events"]}


def _should_continue(state: ResearchState) -> str:
    return "done" if state["done"] else "continue"


def create_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("judge", judge_node)
    graph.add_node("check", check_termination_node)
    graph.add_node("refine", refine_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "judge")
    graph.add_edge("judge", "check")
    graph.add_conditional_edges(
        "check",
        _should_continue,
        {
            "done": END,
            "continue": "refine",
        },
    )
    graph.add_edge("refine", "retrieve")

    return graph.compile()


async def run_research(
    query: str,
    depth: int,
    accuracy: int,
    router_output: RouterOutput,
    topology: ResolvedTopology,
    event_queue: asyncio.Queue,
) -> ResearchState:
    params = get_slider_params(depth, accuracy)
    max_iter = params.get("max_iterations", 3)
    threshold = params.get("threshold", 0.75)
    judge_count = params.get("judge_count", 3)
    min_corr = params.get("min_corroborating", 2)
    granularity = params.get("granularity", "auto")

    if granularity == "auto":
        granularity = "per_claim" if depth >= 4 else "whole"

    initial_queries = _build_initial_queries(query, router_output, depth)

    initial_state: ResearchState = {
        "query": query,
        "depth": depth,
        "accuracy": accuracy,
        "router_output": router_output,
        "topology": topology,
        "queries": initial_queries,
        "sources": [],
        "source_ids": [],
        "claims": [],
        "answer": "",
        "iterations": [],
        "quality_history": [],
        "gap_list": [],
        "failed_claim_ids": [],
        "iteration": 1,
        "max_iterations": max_iter,
        "threshold": threshold,
        "judge_count": judge_count,
        "min_corroborating": min_corr,
        "granularity": granularity,
        "done": False,
        "stop_reason": "",
        "contested_positions": [],
        "events": [],
        "start_time": time.time(),
    }

    graph = create_research_graph()
    final_state = await graph.ainvoke(initial_state)

    for event in final_state.get("events", []):
        await event_queue.put(event)

    return final_state
