from __future__ import annotations

import json
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Critique, Paper, StudyExtraction, Synthesis
from utils import citation_label, safe_json_loads

SYNTH_SYSTEM = """You synthesize evidence into a concise, citation-grounded answer.
Every bullet or paragraph MUST include inline citations like [AuthorYear].
Return ONLY valid JSON matching the schema."""


def _build_citation_map(papers: List[Paper]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for paper in papers:
        label = citation_label(paper.authors, paper.year)
        mapping[paper.paper_id] = label
    return mapping


def _fallback_synthesis(
    question: str,
    papers: List[Paper],
    extractions: List[StudyExtraction],
    critiques: List[Critique],
) -> Synthesis:
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
