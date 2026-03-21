"""
Reranking strategies to improve retrieval precision.

Why rerank?
───────────
Initial retrieval (dense / sparse / hybrid) optimizes for recall — it fetches
candidates that are likely relevant. Rerankers optimize for precision — they
score each candidate against the query with a more powerful model, then
re-sort and truncate to the final context window.

Typical pipeline:
  retrieve top-20 (fast, coarse) → rerank → take top-5 (slow, precise)

Reranker comparison
────────────────────
Cross-encoder (default)
  A transformer that jointly encodes [query, document] and outputs a
  relevance score in [0, 1]. Much more powerful than bi-encoder similarity.
  Models: ms-marco-MiniLM-L-6-v2 (fast), ms-marco-MiniLM-L-12-v2 (better),
          cross-encoder/ms-marco-electra-base (best quality).
  Caveat: O(k) inference calls; each call is ~10–50 ms on CPU.
          Use GPU for latency-critical paths.
  Free, local, no API cost.

Cohere Reranker
  API-based; strong multilingual support. Simple to integrate.
  rerank-english-v3.0 / rerank-multilingual-v3.0
  Caveat: adds ~100–300 ms network latency; API cost per call.
  Best when: multilingual corpus or when you can't run GPU locally.

LLM Reranker
  Ask the generation LLM to score or rank retrieved passages.
  Approaches:
  - Pointwise: prompt LLM "Is this passage relevant to the query? Score 1–10"
  - Listwise: prompt LLM "Rank these passages by relevance" (RankGPT paper)
  - Pairwise: compare pairs; O(k²) calls (expensive)
  Caveat: high latency + cost; best for offline batch pipelines.

FlashRank
  Ultra-fast cross-encoder (ONNX quantized). <5 ms per batch on CPU.
  Install: pip install flashrank. Best latency/quality tradeoff for production.

None
  Skip reranking. Acceptable when:
  - top_k_retrieval is already small (≤5) and retrieval quality is high
  - latency budget is tight (< 100 ms end-to-end)
  - BM25-only queries where keyword matching is already very precise

Production tips
───────────────
1. Reranking k=20→5 is the sweet spot. Diminishing returns beyond k=50.
2. Cross-encoder scores are calibrated; BM25/cosine scores are not.
   After reranking, scores in [0,1] can be used as confidence thresholds.
3. Log score distributions over time; a sudden score drop means retrieval quality degraded.
4. For async APIs, run reranking in a thread pool (it's CPU-bound).
5. Consider caching rerank results for repeated queries.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from config import RerankerType, settings
from models import ScoredChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseReranker(ABC):

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        """
        Rerank candidates and return the top-k most relevant.
        Input order does not matter; output is sorted by relevance descending.
        """
        ...


# ---------------------------------------------------------------------------
# No-op reranker
# ---------------------------------------------------------------------------

class NoReranker(BaseReranker):
    """Pass-through: just truncate to top_k without reranking."""

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        k = top_k or settings.top_k_final
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """
    Local cross-encoder using sentence-transformers.

    The cross-encoder sees [CLS] query [SEP] passage [SEP] and outputs a
    relevance logit. It's 10–100× more accurate than cosine similarity but
    also 10–100× slower per candidate.

    Performance guide:
      MiniLM-L6-v2   → 20 ms/query (CPU), excellent baseline
      MiniLM-L12-v2  → 35 ms/query (CPU), +3% NDCG
      electra-base    → 80 ms/query (CPU), +5% NDCG
      Any above + GPU → 3–5× speedup

    Alternative: FlashRank (ONNX quantized, ~5 ms):
      from flashrank import Ranker
      ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    """

    def __init__(self, model: str | None = None):
        self.model_name = model or settings.cross_encoder_model
        self._model = None  # Lazy load

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError("Install sentence-transformers: pip install sentence-transformers")
            self._model = CrossEncoder(self.model_name, max_length=512)
            logger.info("Loaded CrossEncoder: %s", self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        k = top_k or settings.top_k_final
        model = self._get_model()

        pairs = [(query, sc.chunk.content) for sc in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        reranked = [
            ScoredChunk(chunk=sc.chunk, score=float(s), retrieval_method="reranked:cross_encoder")
            for sc, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "CrossEncoder reranked %d → %d; top score=%.3f",
            len(candidates), k, reranked[0].score if reranked else 0,
        )
        return reranked[:k]


# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------

class CohereReranker(BaseReranker):
    """
    API-based reranker from Cohere.

    Advantages over cross-encoder:
    - Managed: no GPU needed
    - Multilingual: rerank-multilingual-v3.0 supports 100+ languages
    - Compression: returns compressed embeddings (reduces storage)

    Rate limits: 10K rerankings/month on free tier; paid = higher.
    Latency: ~100–300 ms (network round-trip).
    """

    def __init__(self, model: str | None = None):
        self.model_name = model or settings.cohere_reranker_model
        try:
            import cohere
            self._client = cohere.Client(settings.cohere_api_key)
        except ImportError:
            raise ImportError("Install cohere: pip install cohere")

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        k = top_k or settings.top_k_final
        docs = [sc.chunk.content for sc in candidates]

        response = self._client.rerank(
            query=query,
            documents=docs,
            model=self.model_name,
            top_n=k,
        )

        reranked: list[ScoredChunk] = []
        for result in response.results:
            sc = candidates[result.index]
            reranked.append(
                ScoredChunk(
                    chunk=sc.chunk,
                    score=result.relevance_score,
                    retrieval_method="reranked:cohere",
                )
            )

        logger.debug("Cohere reranked %d → %d", len(candidates), len(reranked))
        return reranked


# ---------------------------------------------------------------------------
# LLM listwise reranker (RankGPT style)
# ---------------------------------------------------------------------------

class LLMReranker(BaseReranker):
    """
    Ask the generation LLM to rank passages (RankGPT, Sun et al. 2023).

    Listwise approach:
    - Send all candidates to LLM with "rank by relevance" instruction
    - Parse the returned ranked list

    Best for: offline batch pipelines, high-stakes retrieval.
    Cost: 1 LLM call per query (cheap), but latency is ~1–3 s.
    Caveat: LLM may not follow ranking instructions perfectly; add fallback.
    """

    def __init__(self, generator_fn):
        """
        Args:
            generator_fn: Callable(prompt: str) -> str  (e.g., Claude call)
        """
        self.generator_fn = generator_fn

    def rerank(
        self,
        query: str,
        candidates: list[ScoredChunk],
        top_k: int | None = None,
    ) -> list[ScoredChunk]:
        if not candidates:
            return []

        k = top_k or settings.top_k_final
        numbered = "\n\n".join(
            f"[{i+1}] {sc.chunk.content[:400]}" for i, sc in enumerate(candidates)
        )
        prompt = (
            f"Given the query: \"{query}\"\n\n"
            f"Rank the following passages by relevance (most relevant first).\n"
            f"Return ONLY a comma-separated list of passage numbers, e.g.: 3,1,5,2,4\n\n"
            f"{numbered}\n\nRanking:"
        )

        try:
            response = self.generator_fn(prompt)
            ranking = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
            reranked = [candidates[i] for i in ranking if 0 <= i < len(candidates)]
            # Fallback: append any candidates not mentioned
            mentioned = set(ranking)
            for i, sc in enumerate(candidates):
                if i not in mentioned:
                    reranked.append(sc)
        except Exception as exc:
            logger.error("LLM reranker failed: %s — falling back to original order", exc)
            reranked = candidates

        for i, sc in enumerate(reranked):
            sc.score = 1.0 - (i / len(reranked))  # Assign rank-based scores
            sc.retrieval_method = "reranked:llm"

        return reranked[:k]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reranker(
    reranker_type: RerankerType | None = None,
    generator_fn=None,
) -> BaseReranker:
    rtype = reranker_type or settings.reranker_type

    if rtype == RerankerType.NONE:
        return NoReranker()
    elif rtype == RerankerType.CROSS_ENCODER:
        return CrossEncoderReranker()
    elif rtype == RerankerType.COHERE:
        return CohereReranker()
    elif rtype == RerankerType.LLM:
        if generator_fn is None:
            raise ValueError("LLM reranker requires a generator_fn argument")
        return LLMReranker(generator_fn)
    else:
        raise ValueError(f"Unknown reranker type: {rtype}")
