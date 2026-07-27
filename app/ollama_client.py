"""
Thin async client for the shared Ollama server. See
https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in one or two sentences, focusing on any text, "
    "data, diagrams, or educational content shown."
)


class OllamaUnreachableError(Exception):
    def __init__(self, host: str, cause: Exception | None = None):
        super().__init__(f"Unable to reach Ollama server at {host}")
        self.cause = cause


class OllamaClient:
    def __init__(
        self,
        host: str = settings.ollama_host,
        embed_model: str = settings.embed_model,
        chat_model: str = settings.chat_model,
        vision_model: str = settings.vision_model,
    ):
        self.host = host
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.vision_model = vision_model

    async def is_reachable(self, timeout_s: float = 3.0) -> bool:
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.embed_model, "prompt": text},
                )
            except httpx.HTTPError as cause:
                raise OllamaUnreachableError(self.host, cause) from cause
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama embeddings request failed with status {response.status_code}"
                )
            return response.json()["embedding"]

    async def chat_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Yields incremental text tokens as they arrive."""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.host}/api/chat",
                    json={"model": self.chat_model, "messages": messages, "stream": True},
                ) as response:
                    if response.status_code != 200:
                        raise RuntimeError(
                            f"Ollama chat request failed with status {response.status_code}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        parsed = json.loads(line)
                        token = (parsed.get("message") or {}).get("content", "")
                        if token:
                            yield token
        except httpx.HTTPError as cause:
            raise OllamaUnreachableError(self.host, cause) from cause

    async def caption(self, image_base64: str, prompt: str = DEFAULT_CAPTION_PROMPT) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.vision_model,
                        "messages": [
                            {"role": "user", "content": prompt, "images": [image_base64]}
                        ],
                        "stream": False,
                    },
                )
            except httpx.HTTPError as cause:
                raise OllamaUnreachableError(self.host, cause) from cause
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama caption request failed with status {response.status_code}"
                )
            data = response.json()
            return (data.get("message") or {}).get("content", "").strip()
