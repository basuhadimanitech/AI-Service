"""
Thin async client for the OpenAI API. See
https://platform.openai.com/docs/api-reference
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from app.config import settings

DEFAULT_CAPTION_PROMPT = (
    "Describe this image in one or two sentences, focusing on any text, "
    "data, diagrams, or educational content shown."
)


class LLMUnreachableError(RuntimeError):
    def __init__(self, cause: Exception | None = None):
        super().__init__("Unable to reach the OpenAI API")
        self.cause = cause


class LLMClient:
    def __init__(
        self,
        api_key: str = settings.openai_api_key,
        embed_model: str = settings.embed_model,
        chat_model: str = settings.chat_model,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self.embed_model = embed_model
        self.chat_model = chat_model

    async def is_reachable(self, timeout_s: float = 3.0) -> bool:
        try:
            await self._client.with_options(timeout=timeout_s).models.list()
            return True
        except Exception:
            return False

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(model=self.embed_model, input=text)
        except (APIConnectionError, APIStatusError) as cause:
            raise LLMUnreachableError(cause) from cause
        return response.data[0].embedding

    async def chat_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Yields incremental text tokens as they arrive."""
        try:
            stream = await self._client.chat.completions.create(
                model=self.chat_model, messages=messages, stream=True
            )
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except (APIConnectionError, APIStatusError) as cause:
            raise LLMUnreachableError(cause) from cause

    async def caption(self, image_base64: str, prompt: str = DEFAULT_CAPTION_PROMPT) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
            )
        except (APIConnectionError, APIStatusError) as cause:
            raise LLMUnreachableError(cause) from cause
        return (response.choices[0].message.content or "").strip()
