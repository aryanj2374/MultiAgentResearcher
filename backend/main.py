from __future__ import annotations

import asyncio
from contextlib import suppress
import json
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import get_settings
from .orchestrator import run_question, run_question_with_progress
from .schemas import AskRequest, RunResponse

# Configure logging to see agent diagnostics
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(title="Multi-Agent Scientific Research Assistant")
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=RunResponse)
async def ask(payload: AskRequest) -> RunResponse:
    try:
        return await run_question(payload.question)
    except Exception as exc:  # pragma: no cover - guardrail
        logging.exception("Failed to process research question")
        raise HTTPException(status_code=500, detail="Failed to process question.") from exc


async def generate_sse_events(question: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for agent progress and final result with keep-alive."""
    iterator = run_question_with_progress(question).__aiter__()
    pending_task: asyncio.Task | None = None

    try:
        while True:
            # Create a task for the next event if we don't have one pending
            if pending_task is None:
                pending_task = asyncio.create_task(iterator.__anext__())

            # Wait for the event with a 15-second timeout
            done, _ = await asyncio.wait({pending_task}, timeout=15.0)

            if done:
                # Event is ready, get the result
                try:
                    event = pending_task.result()
                    yield f"data: {json.dumps(event)}\n\n"
                    pending_task = None  # Clear so we fetch the next event
                except StopAsyncIteration:
                    break
            else:
                # Timeout - send keep-alive but keep the task pending
                yield ": keep-alive\n\n"
    except asyncio.CancelledError:
        raise
    except Exception:
        logging.exception("Unhandled error while streaming research progress")
        yield f"data: {json.dumps({'type': 'error', 'message': 'Research stream failed.'})}\n\n"
    finally:
        if pending_task and not pending_task.done():
            pending_task.cancel()
            with suppress(asyncio.CancelledError):
                await pending_task
        with suppress(Exception):
            await iterator.aclose()


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest):
    return StreamingResponse(
        generate_sse_events(payload.question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
