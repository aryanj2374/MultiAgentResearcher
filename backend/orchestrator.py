from __future__ import annotations

from typing import Any
from uuid import uuid4

from agents.critic import critique_all
from agents.extractor import extract_all
from agents.referee import verify_synthesis
from agents.retriever import retrieve_papers
from agents.synthesizer import synthesize
from llm import ChatLLM
from schemas import RunResponse
from storage import save_run


async def run_question(question: str) -> RunResponse:
    run_id = str(uuid4())
    logs: dict[str, Any] = {"notes": []}

    llm = ChatLLM()
    if not llm.available:
        logs["notes"].append("LLM not configured; using heuristic fallbacks.")

    # Retrieve papers with improved query handling
    papers, retrieval_meta = await retrieve_papers(question)
    logs["retrieve"] = {
        "question": question,
        "count": len(papers),
        "papers": [p.model_dump() for p in papers],
        "search_metadata": retrieval_meta,
    }
    
    if not papers:
        logs["notes"].append(f"No papers found after {retrieval_meta['total_attempts']} search attempts.")

    extractions = await extract_all(papers, llm)
    logs["extract"] = [e.model_dump() for e in extractions]

    critiques = await critique_all(papers, extractions, llm)
    logs["critique"] = [c.model_dump() for c in critiques]

    # Pass has_papers flag to synthesizer
    synthesis = await synthesize(question, papers, extractions, critiques, llm)
    logs["synthesize"] = {"initial": synthesis.model_dump()}

    verification = verify_synthesis(synthesis, papers, critiques)
    logs["verify"] = {"initial": verification.model_dump()}

    if not verification.passed:
        revised = await synthesize(question, papers, extractions, critiques, llm, issues=verification.issues)
        logs["synthesize"]["revised"] = revised.model_dump()
        verification = verify_synthesis(revised, papers, critiques)
        logs["verify"]["revised"] = verification.model_dump()
        synthesis = revised
        if not verification.passed:
            verification.revised_synthesis = revised

    response = RunResponse(
        run_id=run_id,
        question=question,
        papers=papers,
        extractions=extractions,
        critiques=critiques,
        synthesis=synthesis,
        verification=verification,
        logs=logs,
    )

    try:
        save_run(response)
    except OSError:
        if response.logs is not None:
            response.logs.setdefault("notes", []).append("Failed to persist run logs.")

    return response


async def run_question_with_progress(question: str):
    """Async generator that yields progress events for each agent step."""
    run_id = str(uuid4())
    logs: dict[str, Any] = {"notes": []}

    llm = ChatLLM()
    if not llm.available:
        logs["notes"].append("LLM not configured; using heuristic fallbacks.")

    # Agent names for progress tracking
    agents = ["retriever", "extractor", "critic", "synthesizer", "referee"]

    # Retriever step
    yield {"type": "progress", "agent": "retriever", "status": "running"}
    try:
        papers, retrieval_meta = await retrieve_papers(question)
        logs["retrieve"] = {
            "question": question,
            "count": len(papers),
            "papers": [p.model_dump() for p in papers],
            "search_metadata": retrieval_meta,
        }
        if not papers:
            logs["notes"].append(f"No papers found after {retrieval_meta['total_attempts']} search attempts.")
        yield {"type": "progress", "agent": "retriever", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "retriever", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Retriever failed: {e}"}
        return

    # Extractor step
    yield {"type": "progress", "agent": "extractor", "status": "running"}
    try:
        extractions = await extract_all(papers, llm)
        logs["extract"] = [e.model_dump() for e in extractions]
        yield {"type": "progress", "agent": "extractor", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "extractor", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Extractor failed: {e}"}
        return

    # Critic step
    yield {"type": "progress", "agent": "critic", "status": "running"}
    try:
        critiques = await critique_all(papers, extractions, llm)
        logs["critique"] = [c.model_dump() for c in critiques]
        yield {"type": "progress", "agent": "critic", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "critic", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Critic failed: {e}"}
        return

    # Synthesizer step
    yield {"type": "progress", "agent": "synthesizer", "status": "running"}
    try:
        synthesis = await synthesize(question, papers, extractions, critiques, llm)
        logs["synthesize"] = {"initial": synthesis.model_dump()}
        yield {"type": "progress", "agent": "synthesizer", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "synthesizer", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Synthesizer failed: {e}"}
        return

    # Referee (verification) step
    yield {"type": "progress", "agent": "referee", "status": "running"}
    try:
        verification = verify_synthesis(synthesis, papers, critiques)
        logs["verify"] = {"initial": verification.model_dump()}

        if not verification.passed:
            revised = await synthesize(question, papers, extractions, critiques, llm, issues=verification.issues)
            logs["synthesize"]["revised"] = revised.model_dump()
            verification = verify_synthesis(revised, papers, critiques)
            logs["verify"]["revised"] = verification.model_dump()
            synthesis = revised
            if not verification.passed:
                verification.revised_synthesis = revised

        yield {"type": "progress", "agent": "referee", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "referee", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Referee failed: {e}"}
        return

    response = RunResponse(
        run_id=run_id,
        question=question,
        papers=papers,
        extractions=extractions,
        critiques=critiques,
        synthesis=synthesis,
        verification=verification,
        logs=logs,
    )

    try:
        save_run(response)
    except OSError:
        if response.logs is not None:
            response.logs.setdefault("notes", []).append("Failed to persist run logs.")

    yield {"type": "result", "data": response.model_dump()}
