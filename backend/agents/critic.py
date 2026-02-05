from __future__ import annotations

import asyncio
import json
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Critique, Paper, StudyExtraction
from utils import safe_json_loads

CRITIC_SYSTEM = """You assess study quality and risk of bias from limited metadata.
Return ONLY valid JSON matching the schema. No commentary."""

# Maximum concurrent LLM requests to stay within rate limits
MAX_CONCURRENT_CRITIQUES = 5


def _fallback_critique(extraction: StudyExtraction) -> Critique:
    study_type = extraction.study_type
    if study_type in {"meta_analysis", "systematic_review"}:
        risk = "medium"
        rationale = ["Review-level evidence but abstract-only assessment."]
    elif study_type == "RCT":
        risk = "medium"
        rationale = ["Randomized design, but bias details not available in abstract."]
    elif study_type == "observational":
        risk = "high"
        rationale = ["Observational design increases confounding risk."]
    else:
        risk = "unknown"
        rationale = ["Insufficient study design details to assess bias."]

    return Critique(
        paper_id=extraction.paper_id,
        risk_of_bias=risk,
        rationale=rationale,
        red_flags=[],
    )


def _build_prompt(extraction: StudyExtraction, paper: Paper | None) -> str:
    payload = {
        "paper": paper.model_dump() if paper else {},
        "extraction": extraction.model_dump(),
        "notes": [
            "Use risk_of_bias in {low, medium, high, unknown}.",
            "rationale: 2-4 bullets.",
            "red_flags: 0-3 bullets.",
        ],
    }
    return (
        "Assess quality and bias for this study.\n\n"
        f"INPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return JSON ONLY."
    )


async def critique_all(
    papers: List[Paper],
    extractions: List[StudyExtraction],
    llm: ChatLLM,
) -> List[Critique]:
    """Critique all extractions sequentially."""
    paper_lookup = {p.paper_id: p for p in papers}
    results: List[Critique] = []

    for extraction in extractions:
        if not llm.available:
            results.append(_fallback_critique(extraction))
            continue

        try:
            paper = paper_lookup.get(extraction.paper_id)
            raw = await llm.chat(CRITIC_SYSTEM, _build_prompt(extraction, paper), max_tokens=400, temperature=0.2)
            data = safe_json_loads(raw)
            critique = Critique.model_validate(data)
            results.append(critique)
        except (LLMUnavailableError, LLMRequestError, ValueError, AttributeError):
            results.append(_fallback_critique(extraction))

    return results

