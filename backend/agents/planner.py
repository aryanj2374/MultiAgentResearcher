from __future__ import annotations

import re

from ..llm import ChatLLM, LLMRequestError, LLMUnavailableError
from ..schemas import ResearchPlan
from ..utils import safe_json_loads


PLANNER_SYSTEM = """You are a research planning assistant. Your job is to analyze research questions and determine if they should be broken down into sub-questions.

A question is COMPLEX and should be decomposed if:
- It asks about multiple distinct concepts or variables
- It contains words like "and", "or", "versus", "compared to" joining different topics
- It would require searching multiple different research areas
- A single search query would miss important aspects

A question is SIMPLE and should be researched directly if:
- It focuses on one specific topic or relationship
- It can be answered with a single focused literature search
- Breaking it down would create redundant searches

For COMPLEX questions, generate 2-4 focused sub-questions that:
- Each target a specific aspect of the original question
- Are independently searchable
- Together cover all aspects of the original question
- Avoid redundancy between sub-questions

Return ONLY valid JSON matching this schema:
{
  "is_complex": boolean,
  "original_question": "the input question",
  "sub_questions": ["sub-q 1", "sub-q 2", ...] or [] if simple,
  "strategy": "direct" or "decompose",
  "reasoning": "brief explanation of your decision"
}"""


def _build_prompt(question: str) -> str:
    return f"""Analyze this research question and determine the best research strategy:

QUESTION: {question}

Should this be researched directly or decomposed into sub-questions?
Return JSON only."""


def _heuristic_plan(question: str) -> ResearchPlan:
    """Fallback heuristic when LLM is unavailable."""
    # Only split when "and" clearly introduces another complete question.
    # Splitting any occurrence corrupts ordinary phrases such as "diet and
    # exercise" and produces unusable Semantic Scholar queries.
    parts = re.split(
        r"\s+and\s+(?=(?:does|do|is|are|can|could|should|what|how)\b)",
        question.rstrip("?"),
        flags=re.IGNORECASE,
    )[:4]
    sub_questions = [f"{part.strip()}?" for part in parts if len(part.split()) >= 3]
    is_complex = len(sub_questions) >= 2
    if not is_complex:
        sub_questions = []
    
    return ResearchPlan(
        is_complex=is_complex,
        original_question=question,
        sub_questions=sub_questions,
        strategy="decompose" if is_complex else "direct",
        reasoning="Heuristic analysis (LLM unavailable)"
    )


async def plan_research(question: str, llm: ChatLLM) -> ResearchPlan:
    """
    Analyze a question and determine the research strategy.
    
    Returns a ResearchPlan indicating whether to research directly
    or decompose into sub-questions.
    """
    if not llm.available:
        return _heuristic_plan(question)
    
    try:
        raw = await llm.chat(
            PLANNER_SYSTEM,
            _build_prompt(question),
            max_tokens=500,
            temperature=0.1,  # Low temperature for consistent planning
        )
        data = safe_json_loads(raw)
        plan = ResearchPlan.model_validate(data)
        plan.original_question = question
        
        # Normalize LLM output so contradictory plan fields cannot leak into
        # orchestration and UI state.
        normalized_sub_questions = list(
            dict.fromkeys(item.strip() for item in plan.sub_questions if item.strip())
        )[:4]
        plan.sub_questions = normalized_sub_questions
        
        # Ensure at least 2 sub-questions for decompose strategy
        if plan.strategy == "decompose" and len(plan.sub_questions) < 2:
            plan.strategy = "direct"
            plan.sub_questions = []
            plan.reasoning += " (Insufficient sub-questions, falling back to direct)"

        plan.is_complex = plan.strategy == "decompose"
        if plan.strategy == "direct":
            plan.sub_questions = []
        
        return plan
        
    except (LLMUnavailableError, LLMRequestError, ValueError):
        return _heuristic_plan(question)
