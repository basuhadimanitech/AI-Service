"""
AI Q&A service for CaptivateNext runtime: retrieval-augmented Q&A over
course slide content, backed by the OpenAI API instead of a per-learner
local LLM install.
"""

import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from app.chunker import chunk_slide_text
from app.config import settings
from app.models import (
    AskRequest,
    HealthResponse,
    IndexSlideRequest,
    IndexSlideResponse,
    StaleCheckRequest,
    StaleCheckResponse,
)
from app.llm_client import LLMClient, LLMUnreachableError
from app.vector_store import StoredChunk, get_vector_store
import httpx

logger = logging.getLogger("cpai")

app = FastAPI(title="CaptivateNext AI Q&A Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = LLMClient()


@app.exception_handler(LLMUnreachableError)
async def _llm_unreachable_handler(_request, exc: LLMUnreachableError) -> JSONResponse:
    logger.warning("OpenAI API unreachable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    reachable = await llm.is_reachable()
    return HealthResponse(status="ok" if reachable else "degraded", llm_reachable=reachable)

@app.get("/test")
async def test():
     async with httpx.AsyncClient() as client:
         print("hello")
         return client





@app.post("/projects/{project_id}/stale-check", response_model=StaleCheckResponse)
async def stale_check(project_id: str, body: StaleCheckRequest) -> StaleCheckResponse:
    store = get_vector_store(project_id)
    # Slides not present in this signature list no longer exist in the
    # project (e.g. deleted since last preview) - drop their cached chunks.
    store.prune_to_slide_ids([sig.slide_id for sig in body.signatures])
    signatures = {sig.slide_id: sig.content_hash for sig in body.signatures}
    return StaleCheckResponse(stale_slide_ids=store.get_stale_or_missing_slide_ids(signatures))


@app.post("/projects/{project_id}/slides/{slide_id}/index", response_model=IndexSlideResponse)
async def index_slide(project_id: str, slide_id: str, body: IndexSlideRequest) -> IndexSlideResponse:
    captions: list[str] = []
    for image_base64 in body.images:
        try:
            caption = await llm.caption(image_base64)
            if caption:
                captions.append(caption)
        except (LLMUnreachableError, RuntimeError):
            # Skip images that fail to caption rather than failing the
            # whole slide's indexing.
            logger.warning("Caption failed for slide %s in project %s", slide_id, project_id)

    combined_text = "\n\n".join([body.text, *captions]).strip()
    chunks = chunk_slide_text(slide_id, body.content_hash, combined_text)

    stored_chunks: list[StoredChunk] = []
    for chunk in chunks:
        embedding = await llm.embed(chunk.text)
        stored_chunks.append(
            StoredChunk(
                slide_id=chunk.slide_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                embedding=embedding,
            )
        )

    store = get_vector_store(project_id)
    store.upsert_slide_chunks(slide_id, body.content_hash, stored_chunks)
    return IndexSlideResponse(chunks=len(stored_chunks))


@app.post("/projects/{project_id}/ask")
async def ask(project_id: str, body: AskRequest) -> StreamingResponse:
    store = get_vector_store(project_id)
    if store.is_empty:
        raise HTTPException(
            status_code=409, detail="No indexed slide content available to answer questions from."
        )

    query_embedding = await llm.embed(body.question)
    top_chunks = store.search(query_embedding, settings.top_k_chunks)
    context = "\n\n".join(
        f"[{i + 1}] (Slide {chunk.slide_id}) {chunk.text}" for i, chunk in enumerate(top_chunks)
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant answering questions about an e-learning "
                "course, using only the provided slide excerpts as context. If the "
                "answer isn't in the excerpts, say you don't know."
            ),
        },
        *[message.model_dump() for message in body.history],
        {"role": "user", "content": f"Course excerpts:\n{context}\n\nQuestion: {body.question}"},
    ]
    source_slide_ids = list(dict.fromkeys(chunk.slide_id for chunk in top_chunks))

    async def _stream():
        async for token in llm.chat_stream(messages):
            yield json.dumps({"token": token}) + "\n"
        yield json.dumps({"done": True, "sourceSlideIds": source_slide_ids}) + "\n"

    return StreamingResponse(_stream(), media_type="application/x-ndjson")
