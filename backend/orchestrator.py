from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator
from uuid import uuid4

from agents.critic import critique_all
from agents.extractor import extract_all
from agents.planner import plan_research
from agents.referee import verify_synthesis
from agents.retriever import retrieve_papers
from agents.synthesizer import synthesize
from llm import ChatLLM
from schemas import (
    Critique,
    Paper,
    ResearchPlan,
    RunResponse,
    StudyExtraction,
    SubQuestionResult,
)
from storage import save_run


async def _process_single_question(
    question: str,
    llm: ChatLLM,
    logs: dict[str, Any],
    log_prefix: str = "",
) -> tuple[list[Paper], list[StudyExtraction], list[Critique], dict[str, Any]]:
    """Process a single question through retrieve -> extract -> critique pipeline."""
    # Retrieve
    papers, retrieval_meta = await retrieve_papers(question)
    logs[f"{log_prefix}retrieve"] = {
        "question": question,
        "count": len(papers),
        "papers": [p.model_dump() for p in papers],
        "search_metadata": retrieval_meta,
    }
    
    # Extract
    extractions = await extract_all(papers, llm)
    logs[f"{log_prefix}extract"] = [e.model_dump() for e in extractions]
    
    # Critique
    critiques = await critique_all(papers, extractions, llm)
    logs[f"{log_prefix}critique"] = [c.model_dump() for c in critiques]
    
    return papers, extractions, critiques, logs


async def run_question(question: str) -> RunResponse:
    run_id = str(uuid4())
    logs: dict[str, Any] = {"notes": []}

    llm = ChatLLM()
    if not llm.available:
        logs["notes"].append("LLM not configured; using heuristic fallbacks.")

    # Step 1: Plan the research
    plan = await plan_research(question, llm)
    logs["plan"] = plan.model_dump()

    sub_results: list[SubQuestionResult] | None = None
    
    if plan.strategy == "decompose" and plan.sub_questions:
        # Deep research mode: process sub-questions in parallel
        logs["notes"].append(f"Deep research mode: {len(plan.sub_questions)} sub-questions")
        
        async def process_sub_question(sub_q: str, idx: int):
            sub_logs: dict[str, Any] = {}
            papers, extractions, critiques, _ = await _process_single_question(
                sub_q, llm, sub_logs, log_prefix=""
            )
            logs[f"sub_question_{idx}"] = sub_logs
            return SubQuestionResult(
                sub_question=sub_q,
                papers=papers,
                extractions=extractions,
                critiques=critiques,
            )
        
        # Process all sub-questions in parallel
        sub_results = await asyncio.gather(*[
            process_sub_question(sq, i) 
            for i, sq in enumerate(plan.sub_questions)
        ])
        
        # Merge results from all sub-questions
        all_papers: list[Paper] = []
        all_extractions: list[StudyExtraction] = []
        all_critiques: list[Critique] = []
        seen_paper_ids: set[str] = set()
        
        for result in sub_results:
            for paper in result.papers:
                if paper.paper_id not in seen_paper_ids:
                    all_papers.append(paper)
                    seen_paper_ids.add(paper.paper_id)
            all_extractions.extend(result.extractions)
            all_critiques.extend(result.critiques)
        
        papers = all_papers
        extractions = all_extractions
        critiques = all_critiques
    else:
        # Direct research mode
        papers, extractions, critiques, logs = await _process_single_question(
            question, llm, logs
        )

    # Synthesize with full context
    synthesis = await synthesize(question, papers, extractions, critiques, llm)
    logs["synthesize"] = {"initial": synthesis.model_dump()}

    # Verify
    verification = verify_synthesis(synthesis, papers, critiques)
    logs["verify"] = {"initial": verification.model_dump()}

    if not verification.passed:
        revised = await synthesize(
            question, papers, extractions, critiques, llm, issues=verification.issues
        )
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
        plan=plan,
        sub_results=sub_results,
    )

    try:
        save_run(response)
    except OSError:
        if response.logs is not None:
            response.logs.setdefault("notes", []).append("Failed to persist run logs.")

    return response


async def run_question_with_progress(question: str) -> AsyncGenerator[dict[str, Any], None]:
    """Async generator that yields progress events for each agent step."""
    run_id = str(uuid4())
    logs: dict[str, Any] = {"notes": []}

    llm = ChatLLM()
    if not llm.available:
        logs["notes"].append("LLM not configured; using heuristic fallbacks.")

    # Step 1: Planner
    yield {"type": "progress", "agent": "planner", "status": "running"}
    try:
        plan = await plan_research(question, llm)
        logs["plan"] = plan.model_dump()
        yield {
            "type": "progress", 
            "agent": "planner", 
            "status": "completed",
            "plan": plan.model_dump(),
        }
    except Exception as e:
        yield {"type": "progress", "agent": "planner", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Planner failed: {e}"}
        return

    sub_results: list[SubQuestionResult] | None = None
    
    if plan.strategy == "decompose" and plan.sub_questions:
        # Deep research mode
        logs["notes"].append(f"Deep research mode: {len(plan.sub_questions)} sub-questions")
        
        # Signal start of sub-question processing
        yield {
            "type": "deep_research_start",
            "sub_questions": plan.sub_questions,
        }
        
        async def process_sub_with_progress(sub_q: str, idx: int):
            """Process a sub-question and return result (progress handled separately)."""
            sub_logs: dict[str, Any] = {}
            papers, retrieval_meta = await retrieve_papers(sub_q)
            sub_logs["retrieve"] = {
                "question": sub_q,
                "count": len(papers),
            }
            
            extractions = await extract_all(papers, llm)
            critiques = await critique_all(papers, extractions, llm)
            
            logs[f"sub_question_{idx}"] = sub_logs
            return SubQuestionResult(
                sub_question=sub_q,
                papers=papers,
                extractions=extractions,
                critiques=critiques,
            )
        
        # Process sub-questions and yield progress for each
        sub_results = []
        for idx, sub_q in enumerate(plan.sub_questions):
            yield {
                "type": "sub_question_progress",
                "index": idx,
                "sub_question": sub_q,
                "status": "running",
            }
            try:
                result = await process_sub_with_progress(sub_q, idx)
                sub_results.append(result)
                yield {
                    "type": "sub_question_progress",
                    "index": idx,
                    "sub_question": sub_q,
                    "status": "completed",
                    "papers_found": len(result.papers),
                }
            except Exception as e:
                yield {
                    "type": "sub_question_progress",
                    "index": idx,
                    "sub_question": sub_q,
                    "status": "failed",
                    "message": str(e),
                }
        
        # Merge results
        all_papers: list[Paper] = []
        all_extractions: list[StudyExtraction] = []
        all_critiques: list[Critique] = []
        seen_paper_ids: set[str] = set()
        
        for result in sub_results:
            for paper in result.papers:
                if paper.paper_id not in seen_paper_ids:
                    all_papers.append(paper)
                    seen_paper_ids.add(paper.paper_id)
            all_extractions.extend(result.extractions)
            all_critiques.extend(result.critiques)
        
        papers = all_papers
        extractions = all_extractions
        critiques = all_critiques
        
        # Mark retriever/extractor/critic as completed (they ran per sub-question)
        yield {"type": "progress", "agent": "retriever", "status": "completed"}
        yield {"type": "progress", "agent": "extractor", "status": "completed"}
        yield {"type": "progress", "agent": "critic", "status": "completed"}
        
    else:
        # Direct research mode - standard pipeline
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
                logs["notes"].append(
                    f"No papers found after {retrieval_meta['total_attempts']} search attempts."
                )
            yield {"type": "progress", "agent": "retriever", "status": "completed"}
        except Exception as e:
            yield {"type": "progress", "agent": "retriever", "status": "failed", "message": str(e)}
            yield {"type": "error", "message": f"Retriever failed: {e}"}
            return

        yield {"type": "progress", "agent": "extractor", "status": "running"}
        try:
            extractions = await extract_all(papers, llm)
            logs["extract"] = [e.model_dump() for e in extractions]
            yield {"type": "progress", "agent": "extractor", "status": "completed"}
        except Exception as e:
            yield {"type": "progress", "agent": "extractor", "status": "failed", "message": str(e)}
            yield {"type": "error", "message": f"Extractor failed: {e}"}
            return

        yield {"type": "progress", "agent": "critic", "status": "running"}
        try:
            critiques = await critique_all(papers, extractions, llm)
            logs["critique"] = [c.model_dump() for c in critiques]
            yield {"type": "progress", "agent": "critic", "status": "completed"}
        except Exception as e:
            yield {"type": "progress", "agent": "critic", "status": "failed", "message": str(e)}
            yield {"type": "error", "message": f"Critic failed: {e}"}
            return

    # Synthesizer
    yield {"type": "progress", "agent": "synthesizer", "status": "running"}
    try:
        synthesis = await synthesize(question, papers, extractions, critiques, llm)
        logs["synthesize"] = {"initial": synthesis.model_dump()}
        yield {"type": "progress", "agent": "synthesizer", "status": "completed"}
    except Exception as e:
        yield {"type": "progress", "agent": "synthesizer", "status": "failed", "message": str(e)}
        yield {"type": "error", "message": f"Synthesizer failed: {e}"}
        return

    # Referee
    yield {"type": "progress", "agent": "referee", "status": "running"}
    try:
        verification = verify_synthesis(synthesis, papers, critiques)
        logs["verify"] = {"initial": verification.model_dump()}

        if not verification.passed:
            revised = await synthesize(
                question, papers, extractions, critiques, llm, issues=verification.issues
            )
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
        plan=plan,
        sub_results=sub_results,
    )

    try:
        save_run(response)
    except OSError:
        if response.logs is not None:
            response.logs.setdefault("notes", []).append("Failed to persist run logs.")

    yield {"type": "result", "data": response.model_dump()}
