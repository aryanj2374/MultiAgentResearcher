from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from orchestrator import run_question
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
