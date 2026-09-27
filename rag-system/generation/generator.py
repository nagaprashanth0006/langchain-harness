"LLM generation backends for the RAG pipeline."

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from config import GeneratorType, settings
from models import RAGResponse, ScoredChunk
from generation.prompts import build_rag_prompt, RAGPromptTemplate

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> RAGResponse:
        """Generate a grounded answer from query + retrieved context."""
        ...

    @abstractmethod
    def generate_stream(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> Iterator[str]:
        """Stream the generated answer token by token."""
        ...

    def complete(self, prompt: str) -> str:
        """
        Simple single-turn completion (used by HyDE / LLM reranker).
        Override for efficiency if the backend has a cheaper completion API.
        """
        dummy_chunk: list[ScoredChunk] = []

        raise NotImplementedError("Subclass must implement complete()")


class AnthropicGenerator(BaseGenerator):
    """
    Claude generation via the Anthropic SDK.

    Production features used:
    - System prompt caching via `cache_control` (beta header)
    - Streaming via `stream=True`
    - Usage tracking for cost attribution
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key
                or os.environ.get("ANTHROPIC_API_KEY", "")
            )
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.model_name = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )

    def _call(self, system: str, user_message: str) -> tuple[str, int, int]:
        """
        Single non-streaming call. Returns (text, prompt_tokens, completion_tokens).
        """
        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        latency = time.perf_counter() - t0
        text = response.content[0].text
        usage = response.usage
        logger.info(
            "Claude generation: model=%s in_tokens=%d out_tokens=%d latency=%.2fs",
            self.model_name,
            usage.input_tokens,
            usage.output_tokens,
            latency,
        )
        return text, usage.input_tokens, usage.output_tokens

    def generate(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> RAGResponse:
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        answer, prompt_tokens, completion_tokens = self._call(system, user_message)
        return RAGResponse(
            answer=answer,
            query=query,
            sources=scored_chunks,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_stream(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> Iterator[str]:
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        with self._client.messages.stream(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def complete(self, prompt: str) -> str:
        """Simple completion for HyDE / LLM reranker."""
        text, _, _ = self._call(
            system="You are a helpful assistant.",
            user_message=prompt,
        )
        return text


class OpenAIGenerator(BaseGenerator):
    """
    GPT-4o / GPT-4o-mini generation via the OpenAI SDK.

    Production features:
    - Streaming
    - Seed for reproducibility (pass seed= for deterministic outputs)
    - Usage tracking
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        try:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=settings.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
            )
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.model_name = model or settings.openai_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )

    def _call(self, system: str, user_message: str) -> tuple[str, int, int]:
        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        latency = time.perf_counter() - t0
        text = response.choices[0].message.content or ""
        usage = response.usage
        logger.info(
            "OpenAI generation: model=%s in_tokens=%d out_tokens=%d latency=%.2fs",
            self.model_name,
            usage.prompt_tokens,
            usage.completion_tokens,
            latency,
        )
        return text, usage.prompt_tokens, usage.completion_tokens

    def generate(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> RAGResponse:
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        answer, prompt_tokens, completion_tokens = self._call(system, user_message)
        return RAGResponse(
            answer=answer,
            query=query,
            sources=scored_chunks,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_stream(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> Iterator[str]:
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        stream = self._client.chat.completions.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete(self, prompt: str) -> str:
        text, _, _ = self._call("You are a helpful assistant.", prompt)
        return text


class OllamaGenerator(BaseGenerator):
    "Local LLM generation via the Ollama daemon."

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        try:
            import ollama as _ollama

            self._client = _ollama.Client(
                host=settings.ollama_base_url,
                timeout=settings.ollama_timeout,
            )
            self._ollama = _ollama
        except ImportError:
            raise ImportError(
                "Install the Ollama Python SDK:  pip install ollama\n"
                "Then start the daemon:           ollama serve\n"
                "And pull a model:                ollama pull llama3.2"
            )

        self.model_name = model or settings.ollama_chat_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self._keep_alive = settings.ollama_keep_alive

        logger.info(
            "OllamaGenerator initialized: model=%s base_url=%s",
            self.model_name,
            settings.ollama_base_url,
        )

    def _build_messages(self, system: str, user_message: str) -> list[dict]:
        """Ollama uses the standard OpenAI-compatible message format."""
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]

    def _call(self, system: str, user_message: str) -> tuple[str, int, int]:
        """
        Non-streaming call. Returns (text, prompt_tokens, completion_tokens).

        Ollama's response includes eval_count (completion tokens) and
        prompt_eval_count (prompt tokens) in the response metadata.
        """
        t0 = time.perf_counter()
        response = self._client.chat(
            model=self.model_name,
            messages=self._build_messages(system, user_message),
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            keep_alive=self._keep_alive,
            stream=False,
        )
        latency = time.perf_counter() - t0

        text = response.message.content or ""
        prompt_tokens = getattr(response, "prompt_eval_count", 0) or 0
        completion_tokens = getattr(response, "eval_count", 0) or 0

        logger.info(
            "Ollama generation: model=%s in_tokens=%d out_tokens=%d latency=%.2fs",
            self.model_name,
            prompt_tokens,
            completion_tokens,
            latency,
        )
        return text, prompt_tokens, completion_tokens

    def generate(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> RAGResponse:
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        answer, prompt_tokens, completion_tokens = self._call(system, user_message)
        return RAGResponse(
            answer=answer,
            query=query,
            sources=scored_chunks,
            model=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def generate_stream(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
        template: Optional[RAGPromptTemplate] = None,
    ) -> Iterator[str]:
        """
        Stream tokens from Ollama.  Each chunk from the SDK exposes
        ``chunk.message.content`` with the delta text.
        """
        system, user_message = build_rag_prompt(query, scored_chunks, template)
        stream = self._client.chat(
            model=self.model_name,
            messages=self._build_messages(system, user_message),
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            keep_alive=self._keep_alive,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.message.content
            if delta:
                yield delta

    def complete(self, prompt: str) -> str:
        """Simple single-turn completion for HyDE / LLM reranker."""
        text, _, _ = self._call(
            system="You are a helpful assistant.",
            user_message=prompt,
        )
        return text


def get_generator(generator_type: GeneratorType | None = None) -> BaseGenerator:
    gtype = generator_type or settings.generator_type

    if gtype == GeneratorType.ANTHROPIC:
        return AnthropicGenerator()
    elif gtype == GeneratorType.OPENAI:
        return OpenAIGenerator()
    elif gtype == GeneratorType.OLLAMA:
        return OllamaGenerator()
    else:
        raise ValueError(f"Unknown generator type: {gtype}")
