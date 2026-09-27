"Thin wrapper for OpenAI-compatible LLM calls (Ollama, Groq, OpenRouter)."

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, backend: Dict[str, Any]):
        self.base_url = backend.get("base_url", "http://localhost:11434")
        self.model = backend.get("model", "llama3.2")
        self.api_key = backend.get("api_key") or "ollama"
        self.timeout = backend.get("timeout", 60)
        provider = backend.get("provider", "ollama")

        if "ollama" in self.base_url or provider == "ollama":
            self._chat_endpoint = f"{self.base_url}/v1/chat/completions"
            self._embed_endpoint = f"{self.base_url}/api/embeddings"
        else:
            self._chat_endpoint = f"{self.base_url}/chat/completions"
            self._embed_endpoint = None

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    self._chat_endpoint, json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach LLM backend at {self._chat_endpoint}. "
                f"Make sure Ollama is running (`ollama serve`) and model '{self.model}' is pulled (`ollama pull {self.model}`)."
            )
        except httpx.TimeoutException:
            raise TimeoutError(
                f"LLM request timed out after {self.timeout}s (model: {self.model})"
            )

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", self._chat_endpoint, json=payload, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            chunk = line[6:]
                            if chunk.strip() == "[DONE]":
                                break
                            try:
                                obj = json.loads(chunk)
                                delta = obj["choices"][0]["delta"].get("content", "")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError):
                                continue
        except httpx.ConnectError:
            raise ConnectionError(
                f"Cannot reach LLM backend at {self._chat_endpoint}. "
                f"Make sure Ollama is running (`ollama serve`) and model '{self.model}' is pulled (`ollama pull {self.model}`)."
            )

    async def embed(self, text: str) -> Optional[List[float]]:
        if not self._embed_endpoint:
            return None
        payload = {"model": self.model, "prompt": text}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(self._embed_endpoint, json=payload)
                resp.raise_for_status()
                return resp.json().get("embedding")
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
