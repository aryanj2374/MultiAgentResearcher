from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from orchestrator import run_question, run_question_with_progress
from schemas import AskRequest, RunResponse

app = FastAPI(title="Multi-Agent Scientific Research Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=RunResponse)
async def ask(payload: AskRequest) -> RunResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        return await run_question(question)
    except Exception as exc:  # pragma: no cover - guardrail
        raise HTTPException(status_code=500, detail=f"Failed to process question: {exc}") from exc


async def generate_sse_events(question: str) -> AsyncGenerator[str, None]:
    """Generate SSE events for agent progress and final result with keep-alive."""
    iterator = run_question_with_progress(question).__aiter__()
    pending_task: asyncio.Task | None = None
    
    while True:
        try:
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
        except StopAsyncIteration:
            break
        except Exception:
            # If there's an error, clean up and exit
            if pending_task and not pending_task.done():
                pending_task.cancel()
            break


@app.post("/api/ask/stream")
async def ask_stream(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    return StreamingResponse(
        generate_sse_events(question),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

