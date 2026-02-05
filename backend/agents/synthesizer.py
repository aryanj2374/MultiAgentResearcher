from __future__ import annotations

import json
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Critique, Paper, StudyExtraction, Synthesis
from utils import citation_label, safe_json_loads

SYNTH_SYSTEM = """You synthesize evidence into a concise, citation-grounded answer.
Every bullet or paragraph MUST include inline citations like [AuthorYear].
If no papers were found, acknowledge this and provide general guidance.
Return ONLY valid JSON matching the schema."""

NO_EVIDENCE_SYSTEM = """You are a helpful research assistant. The user asked a research question but no academic papers were found.
Provide a helpful response acknowledging the lack of search results and offering suggestions.
Return ONLY valid JSON matching the schema."""


def _build_citation_map(papers: List[Paper]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for paper in papers:
        label = citation_label(paper.authors, paper.year)
        mapping[paper.paper_id] = label
    return mapping


def _no_evidence_synthesis(question: str) -> Synthesis:
    """Generate a helpful synthesis when no papers were found."""
    return Synthesis(
        final_answer=[
            "No academic papers were found for this specific query.",
            "This could be due to the search terms being too specific or the topic being emerging/niche.",
            "Try rephrasing your question with broader or alternative terms.",
            "Consider searching for related concepts or breaking down your question into smaller parts.",
            "You may also want to search directly on Google Scholar or PubMed for more comprehensive results.",
        ],
        evidence_consensus="Unable to synthesize evidence as no papers were retrieved from Semantic Scholar.",
        top_limitations_overall=[
            "No papers found - synthesis is based on general guidance only.",
            "The Semantic Scholar API may have rate limits or connectivity issues.",
            "Some topics may not be well-indexed in the database.",
        ],
        confidence_score=0,
        confidence_rationale=[
            "Confidence is 0 because no academic evidence was retrieved to support any claims.",
            "This response provides search guidance rather than evidence-based conclusions.",
        ],
        citations_used=[],
    )


def _fallback_synthesis(
    question: str,
    papers: List[Paper],
    extractions: List[StudyExtraction],
    critiques: List[Critique],
) -> Synthesis:
    # Handle empty paper list
    if not papers:
        return _no_evidence_synthesis(question)
    
    citation_map = _build_citation_map(papers)
    bullets: List[str] = []
    citations_used: List[str] = []

    for extraction in extractions:
        label = citation_map.get(extraction.paper_id, "Unknownn.d.")
        bullets.append(f"{extraction.claim_summary} [{label}]")
        citations_used.append(extraction.paper_id)

    evidence_consensus = "Evidence is limited to abstract-level summaries and appears mixed."
    evidence_consensus += " [" + ", ".join({citation_map.get(pid, "Unknownn.d.") for pid in citations_used}) + "]"

    limitations = []
    for extraction in extractions:
        limitations.extend(extraction.limitations)
    if not limitations:
        limitations = ["Abstract-only review; full methods not assessed."]
    limitations = list(dict.fromkeys(limitations))[:5]
    limitations = [f"{item} [" + ", ".join({citation_map.get(pid, "Unknownn.d.") for pid in citations_used}) + "]" for item in limitations]

    high_bias = sum(1 for c in critiques if c.risk_of_bias == "high")
    medium_bias = sum(1 for c in critiques if c.risk_of_bias == "medium")
    score = 70 - high_bias * 15 - medium_bias * 8
    score = max(25, min(85, score))

    rationale = []
    if high_bias:
        rationale.append("Several studies show high risk of bias. [" + ", ".join({citation_map.get(pid, "Unknownn.d.") for pid in citations_used}) + "]")
    if medium_bias:
        rationale.append("Moderate bias limits causal confidence. [" + ", ".join({citation_map.get(pid, "Unknownn.d.") for pid in citations_used}) + "]")
    if not rationale:
        rationale.append("Evidence base is limited to abstracts and may omit key methods. [" + ", ".join({citation_map.get(pid, "Unknownn.d.") for pid in citations_used}) + "]")

    return Synthesis(
        final_answer=bullets[:10],
        evidence_consensus=evidence_consensus,
        top_limitations_overall=limitations,
        confidence_score=score,
        confidence_rationale=rationale,
        citations_used=citations_used,
    )



def _build_prompt(
    question: str,
    papers: List[Paper],
    extractions: List[StudyExtraction],
    critiques: List[Critique],
    issues: List[str] | None = None,
) -> str:
    citation_map = _build_citation_map(papers)
    payload = {
        "question": question,
        "citation_map": citation_map,
        "extractions": [e.model_dump() for e in extractions],
        "critiques": [c.model_dump() for c in critiques],
        "requirements": [
            "Use the citation_map values for inline citations like [Smith2020].",
            "Every bullet in final_answer must include at least one citation.",
            "evidence_consensus must include citations.",
            "top_limitations_overall bullets must include citations.",
            "confidence_score is an int 0-100; justify in confidence_rationale bullets with citations.",
            "citations_used must list paper_id values actually cited.",
        ],
    }
    if issues:
        payload["fix_issues"] = issues

    return (
        "Synthesize a concise, citation-grounded answer.\n\n"
        f"INPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return JSON ONLY."
    )


async def synthesize(
    question: str,
    papers: List[Paper],
    extractions: List[StudyExtraction],
    critiques: List[Critique],
    llm: ChatLLM,
    issues: List[str] | None = None,
) -> Synthesis:
    # Always use no-evidence synthesis when no papers found
    if not papers:
        return _no_evidence_synthesis(question)
    
    if not llm.available:
        return _fallback_synthesis(question, papers, extractions, critiques)

    try:
        raw = await llm.chat(
            SYNTH_SYSTEM,
            _build_prompt(question, papers, extractions, critiques, issues),
            max_tokens=900,
            temperature=0.2,
        )
        data = safe_json_loads(raw)
        return Synthesis.model_validate(data)
    except (LLMUnavailableError, LLMRequestError, ValueError):
        return _fallback_synthesis(question, papers, extractions, critiques)

