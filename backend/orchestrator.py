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

    papers = await retrieve_papers(question)
    logs["retrieve"] = {"question": question, "count": len(papers), "papers": [p.model_dump() for p in papers]}

    extractions = await extract_all(papers, llm)
    logs["extract"] = [e.model_dump() for e in extractions]

    critiques = await critique_all(papers, extractions, llm)
    logs["critique"] = [c.model_dump() for c in critiques]

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
