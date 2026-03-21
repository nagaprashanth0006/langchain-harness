"""
LLM generation backends for the RAG pipeline.

Model selection guide
─────────────────────
Claude claude-sonnet-4-6 (default)
  - Best reasoning and instruction-following for RAG
  - 200K context window — fits large retrieved contexts
  - Supports streaming, system prompts, tool use
  - Production: use prompt caching for system prompts (50% cost reduction)

Claude claude-haiku-4-5
  - 5× cheaper, 3× faster than Sonnet
  - Excellent for simple Q&A and summarization
  - Use when latency < 500 ms is required

GPT-4o
  - Comparable quality to Claude Sonnet
  - Supports parallel function calling
  - Use when customers require OpenAI

GPT-4o-mini
  - 10× cheaper than GPT-4o; better than GPT-3.5
  - Good for non-critical summarization tasks

Open-source alternatives
─────────────────────────
• LLaMA 3.1 70B / 405B via Ollama, vLLM, or Together.ai
  - Full data privacy; on-prem deployment
  - vLLM: 2–4× throughput vs naïve serving
• Mistral Large: strong coding + reasoning, EU jurisdiction
• Command R+ (Cohere): built-in RAG mode with citations

Production generation tips
───────────────────────────
1. Use prompt caching: set `cache_control` on the system prompt (Claude) or
   seed (OpenAI). Repeated queries with the same system prompt cost 50% less.
2. Set temperature=0.0–0.1 for factual RAG; higher for creative tasks.
3. Stream responses for interactive UIs; buffer for batch APIs.
4. Add a max_context_chars guard: truncate context if > model's limit.
5. Log (prompt_tokens, completion_tokens, latency) per request for cost tracking.
6. Implement exponential backoff with jitter on 429/503 errors.
7. Consider response validation: check if answer contains expected entities.
"""

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


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

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
        # Abuse the generate path for a simple completion
        raise NotImplementedError("Subclass must implement complete()")


# ---------------------------------------------------------------------------
# Anthropic / Claude generator
# ---------------------------------------------------------------------------

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
                api_key=settings.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            )
            self._anthropic = anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        self.model_name = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.temperature = temperature if temperature is not None else settings.temperature

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
            self.model_name, usage.input_tokens, usage.output_tokens, latency,
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


# ---------------------------------------------------------------------------
# OpenAI generator
# ---------------------------------------------------------------------------

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
        self.temperature = temperature if temperature is not None else settings.temperature

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
            self.model_name, usage.prompt_tokens, usage.completion_tokens, latency,
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


# ---------------------------------------------------------------------------
# Ollama Generator  (100 % local, zero API cost)
# ---------------------------------------------------------------------------

class OllamaGenerator(BaseGenerator):
    """
    Local LLM generation via the Ollama daemon.

    Prerequisites
    ─────────────
    1. Install Ollama:  https://ollama.com/download
    2. Pull a chat model:
         ollama pull llama3.2          # 3B — fast, good for RAG  ← default
         ollama pull llama3.1          # 8B — balanced quality/speed
         ollama pull mistral           # 7B — strong reasoning
         ollama pull mixtral           # 47B MoE — GPT-3.5 class
         ollama pull gemma2            # 9B — Google Gemma
         ollama pull qwen2.5           # 7B — strong multilingual
         ollama pull phi3              # 3.8B — Microsoft, very fast
         ollama pull deepseek-r1       # reasoning-tuned
         ollama pull command-r         # Cohere, built for RAG + citations
    3. Keep OLLAMA_CHAT_MODEL in .env in sync with the pulled model tag.

    Context window guide
    ─────────────────────
    Most 7B models: 8K–32K tokens.  Set MAX_TOKENS to leave headroom.
    llama3.1 and llama3.2 support 128K with flash-attention enabled.
    Run `ollama show llama3.2` to inspect the model's context length.

    Performance guide
    ──────────────────
    CPU only (no GPU):   phi3/llama3.2-3B ≈ 15–30 tok/s on M2, 5–10 tok/s on x86
    Apple Silicon GPU:   llama3.1-8B ≈ 40–80 tok/s on M3 Max
    NVIDIA GPU:          llama3.1-8B ≈ 100–200 tok/s on RTX 4090

    Production tips
    ───────────────
    • Set OLLAMA_KEEP_ALIVE="24h" on dedicated inference nodes so the model
      stays loaded in VRAM and avoids cold-start (~3–10 s for 7B models).
    • For multi-user workloads, run multiple Ollama processes on separate GPUs
      and load-balance with Nginx or Traefik upstream.
    • Ollama natively handles concurrent requests up to OLLAMA_NUM_PARALLEL
      (default=1); set OLLAMA_NUM_PARALLEL=4 for production throughput.
    • For ultra-low latency (<200 ms), prefer llama3.2:3b or phi3:mini.
    • For highest quality local RAG, prefer mistral-nemo or llama3.1:8b.
    • Enable quantized models (Q4_K_M) for 2× speed with ~3% quality loss.
    """

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
        self.temperature = temperature if temperature is not None else settings.temperature
        self._keep_alive = settings.ollama_keep_alive

        logger.info(
            "OllamaGenerator initialized: model=%s base_url=%s",
            self.model_name, settings.ollama_base_url,
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
                "num_predict": self.max_tokens,   # Ollama's name for max_tokens
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
            self.model_name, prompt_tokens, completion_tokens, latency,
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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
