from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class Paper(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_id: str
    title: str
    authors: list[str]
    year: int | None = None
    venue: str | None = None
    url: str | None = None
    abstract: str | None = None


StudyType = Literal[
    "meta_analysis",
    "systematic_review",
    "RCT",
    "observational",
    "other",
    "unknown",
]


EffectDirection = Literal["positive", "negative", "mixed", "null", "unclear"]


class StudyExtraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_id: str
    claim_summary: str
    study_type: StudyType
    population: str | None = None
    sample_size: int | None = None
    intervention_exposure: str | None = None
    comparison: str | None = None
    outcomes: str | None = None
    effect_direction: EffectDirection | None = None
    effect_size_text: str | None = None
    key_snippet: str
    limitations: list[str]
    apa_citation: str
    url: str | None = None


RiskOfBias = Literal["low", "medium", "high", "unknown"]


class Critique(BaseModel):
    model_config = ConfigDict(extra="ignore")

    paper_id: str
    risk_of_bias: RiskOfBias
    rationale: list[str]
    red_flags: list[str]


class Synthesis(BaseModel):
    model_config = ConfigDict(extra="ignore")

    final_answer: list[str]
    evidence_consensus: str
    top_limitations_overall: list[str]
    confidence_score: int = Field(ge=0, le=100)
    confidence_rationale: list[str]
    citations_used: list[str]


class Verification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    passed: bool
    issues: list[str]
    revised_synthesis: Synthesis | None = None


class SubQuestionResult(BaseModel):
    """Result from processing a single sub-question in deep research mode."""
    model_config = ConfigDict(extra="ignore")
    
    sub_question: str
    papers: list[Paper]
    extractions: list[StudyExtraction]
    critiques: list[Critique]


class ResearchPlan(BaseModel):
    """Output from the Planner agent."""
    model_config = ConfigDict(extra="ignore")
    
    is_complex: bool
    original_question: str
    sub_questions: list[str]
    strategy: str  # "direct" | "decompose"
    reasoning: str


class RunResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    question: str
    papers: list[Paper]
    extractions: list[StudyExtraction]
    critiques: list[Critique]
    synthesis: Synthesis
    verification: Verification
    logs: dict[str, Any] | None = None
    # Deep research fields
    plan: ResearchPlan | None = None
    sub_results: list[SubQuestionResult] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)

