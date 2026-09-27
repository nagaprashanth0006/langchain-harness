"Reranking strategies to improve retrieval precision."

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from config import RerankerType, settings
from models import ScoredChunk

logger = logging.getLogger(__name__)


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


class CrossEncoderReranker(BaseReranker):
    "Local cross-encoder using sentence-transformers."

    def __init__(self, model: str | None = None):
        self.model_name = model or settings.cross_encoder_model
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers"
                )
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
            ScoredChunk(
                chunk=sc.chunk,
                score=float(s),
                retrieval_method="reranked:cross_encoder",
            )
            for sc, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)

        logger.debug(
            "CrossEncoder reranked %d → %d; top score=%.3f",
            len(candidates),
            k,
            reranked[0].score if reranked else 0,
        )
        return reranked[:k]


class CohereReranker(BaseReranker):
    "API-based reranker from Cohere."

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


class LLMReranker(BaseReranker):
    "Ask the generation LLM to rank passages (RankGPT, Sun et al. 2023)."

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
            f"[{i + 1}] {sc.chunk.content[:400]}" for i, sc in enumerate(candidates)
        )
        prompt = (
            f'Given the query: "{query}"\n\n'
            f"Rank the following passages by relevance (most relevant first).\n"
            f"Return ONLY a comma-separated list of passage numbers, e.g.: 3,1,5,2,4\n\n"
            f"{numbered}\n\nRanking:"
        )

        try:
            response = self.generator_fn(prompt)
            ranking = [
                int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()
            ]
            reranked = [candidates[i] for i in ranking if 0 <= i < len(candidates)]

            mentioned = set(ranking)
            for i, sc in enumerate(candidates):
                if i not in mentioned:
                    reranked.append(sc)
        except Exception as exc:
            logger.error(
                "LLM reranker failed: %s — falling back to original order", exc
            )
            reranked = candidates

        for i, sc in enumerate(reranked):
            sc.score = 1.0 - (i / len(reranked))
            sc.retrieval_method = "reranked:llm"

        return reranked[:k]


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
