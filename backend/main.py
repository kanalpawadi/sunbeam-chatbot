"""FastAPI server for the Sunbeam chatbot.

Dev:   (from the repo root)  uvicorn backend.main:app --reload --port 8000
Prod:  build the frontend (npm run build) and the same server also serves frontend/dist.
"""
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from shared.config import CHATBOT_DIR, GROQ_MODEL, EMBED_MODEL

from .agent import build_agent
from .store import store

log = logging.getLogger("sunbeam")
IST = timezone(timedelta(hours=5, minutes=30))
FRONTEND_DIST = CHATBOT_DIR / "frontend" / "dist"

app = FastAPI(title="Sunbeam Chatbot API")
# Browsers may only call this API from these sites (the GitHub Pages frontend + local dev).
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://kanalpawadi.github.io,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=2)
def agent_for(day: str):
    # Rebuilt once per day so "today's date" in the system prompt stays correct.
    return build_agent()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


TOOL_LABELS = {
    "search_sunbeam": "Searching Sunbeam data",
    "list_courses": "Looking up the course catalog",
}


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))


async def typewriter(text: str):
    """Send a complete reply in small pieces so the UI still renders it progressively."""
    for match in re.finditer(r"\S+\s*|\s+", text):
        yield match.group(0)
        await asyncio.sleep(0.008)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    snap = store.snapshot()
    categories: dict[str, int] = {}
    for c in snap.catalog:
        categories[c["category"]] = categories.get(c["category"], 0) + 1
    return {
        "updated_at": snap.updated_at,
        "chunks": snap.chunk_count,
        "documents": snap.document_count,
        "courses": len(snap.catalog),
        "categories": categories,
        "llm": GROQ_MODEL,
        "embeddings": EMBED_MODEL,
        "ready": snap.collection is not None,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    history = [
        HumanMessage(m.content) if m.role == "user" else AIMessage(m.content)
        for m in req.messages[-12:]
    ]

    async def stream():
        sources: list[dict] = []
        try:
            if store.snapshot().collection is None:
                yield sse({"type": "error", "message": "The knowledge base is not built yet. Run the data pipeline first."})
                return
            agent = agent_for(datetime.now(IST).date().isoformat())
            async for msg, meta in agent.astream({"messages": history}, stream_mode="messages"):
                if isinstance(msg, AIMessageChunk):
                    # Token-level streaming (provider streams deltas).
                    for tc in msg.tool_call_chunks or []:
                        if tc.get("name"):
                            yield sse({"type": "status", "text": TOOL_LABELS.get(tc["name"], "Working")})
                    if not msg.tool_call_chunks:
                        text = text_of(msg)
                        if text:
                            yield sse({"type": "token", "text": text})
                elif isinstance(msg, AIMessage):
                    # Whole-message step (provider returned the full reply at once).
                    for tc in msg.tool_calls or []:
                        yield sse({"type": "status", "text": TOOL_LABELS.get(tc["name"], "Working")})
                    if not msg.tool_calls:
                        async for piece in typewriter(text_of(msg)):
                            yield sse({"type": "token", "text": piece})
                elif isinstance(msg, ToolMessage):
                    for s in msg.artifact or []:
                        if s not in sources:
                            sources.append(s)
                    yield sse({"type": "status", "text": "Writing the answer"})
            if sources:
                yield sse({"type": "sources", "items": sources[:6]})
            yield sse({"type": "done"})
        except Exception as exc:
            log.exception("chat failed")
            message = str(exc)
            if "401" in message or "invalid_api_key" in message.lower() or "API key" in message:
                message = "An API key was rejected. Check GROQ_API_KEY and GOOGLE_API_KEY in .env."
            elif "429" in message or "rate" in message.lower():
                message = "The AI service is rate-limited right now. Please try again in a moment."
            else:
                message = f"Something went wrong while answering: {message[:300]}"
            yield sse({"type": "error", "message": message})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# Serve the built React app (production / single-server mode).
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        file = FRONTEND_DIST / path
        if path and file.is_file():
            return FileResponse(file)
        return FileResponse(FRONTEND_DIST / "index.html")
