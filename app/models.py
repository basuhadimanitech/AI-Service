"""
ADOBE CONFIDENTIAL
Copyright 2026 Adobe. All Rights Reserved.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class SlideSignature(CamelModel):
    slide_id: str = Field(alias="slideId")
    content_hash: str = Field(alias="contentHash")


class StaleCheckRequest(CamelModel):
    signatures: list[SlideSignature]


class StaleCheckResponse(CamelModel):
    stale_slide_ids: list[str] = Field(alias="staleSlideIds")


class IndexSlideRequest(CamelModel):
    content_hash: str = Field(alias="contentHash")
    text: str = ""
    # Raw base64 image bytes (no data-URL prefix), one per image on the slide.
    images: list[str] = []


class IndexSlideResponse(CamelModel):
    chunks: int


class ChatMessage(CamelModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AskRequest(CamelModel):
    question: str
    history: list[ChatMessage] = []


class HealthResponse(CamelModel):
    status: Literal["ok", "degraded"]
    ollama_reachable: bool = Field(alias="ollamaReachable")
