from __future__ import annotations

import json
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Critique, Paper, StudyExtraction, Synthesis
from utils import citation_label, safe_json_loads

SYNTH_SYSTEM = """You are an expert research synthesizer. Your role is to analyze multiple scientific studies and provide a clear, evidence-based answer to the user's research question.

CRITICAL INSTRUCTION - ANSWER THE QUESTION FIRST:
Your FIRST bullet point in final_answer MUST directly answer the user's question. Start with a clear statement like:
- "Yes, [X] does improve [Y], based on evidence from [N] studies..."
- "The evidence suggests [X] has a moderate positive effect on [Y]..."
- "Current research is mixed on whether [X] affects [Y]..."
- "No, [X] does not appear to significantly impact [Y]..."

Do NOT start with paper titles, methodology descriptions, or abstract sentences. Start with THE ANSWER.

SYNTHESIS GUIDELINES:
1. **Direct Answer First**: The first bullet answers the question directly with a clear verdict.
2. **Supporting Evidence**: Following bullets provide specific findings that support or nuance the answer.
3. **Be Quantitative**: Include effect sizes, percentages, sample sizes when available (e.g., "improved memory by 15% in a study of 200 adults").
4. **Note Disagreements**: If studies conflict, explain both sides.
5. **Weigh by Quality**: Prioritize meta-analyses and RCTs over observational studies.

CITATION REQUIREMENTS:
- Every bullet point MUST include at least one inline citation like [AuthorYear].
- When multiple studies agree, cite all: [Smith2020, Jones2021].

OUTPUT FORMAT:
- final_answer: 5-8 bullet points. FIRST bullet = direct answer to question. Remaining bullets = supporting evidence, nuances, and specific findings.
- evidence_consensus: 1-2 sentences on how well studies agree.
- top_limitations_overall: Key methodological weaknesses.
- confidence_score: 0-100 based on evidence quality.
- confidence_rationale: Why you assigned that score.

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
    citations_used: List[str] = [e.paper_id for e in extractions]
    
    # Analyze effect directions across studies
    positive_studies = []
    negative_studies = []
    mixed_null_studies = []
    
    for extraction in extractions:
        label = citation_map.get(extraction.paper_id, "Unknown")
        entry = {"label": label, "extraction": extraction}
        if extraction.effect_direction == "positive":
            positive_studies.append(entry)
        elif extraction.effect_direction == "negative":
            negative_studies.append(entry)
        else:
            mixed_null_studies.append(entry)
    
    # Build synthesized findings
    bullets: List[str] = []
    
    # Summary bullet based on overall direction
    total = len(extractions)
    pos_count = len(positive_studies)
    neg_count = len(negative_studies)
    
    if pos_count > neg_count and pos_count > len(mixed_null_studies):
        direction_summary = f"The majority of evidence ({pos_count}/{total} studies) suggests positive effects"
    elif neg_count > pos_count and neg_count > len(mixed_null_studies):
        direction_summary = f"The majority of evidence ({neg_count}/{total} studies) suggests negative effects"
    else:
        direction_summary = f"Evidence is mixed across the {total} reviewed studies"
    
    all_labels = [citation_map.get(e.paper_id, "Unknown") for e in extractions]
    bullets.append(f"{direction_summary}. [{', '.join(all_labels[:3])}{'...' if len(all_labels) > 3 else ''}]")
    
    # Add specific findings grouped by outcome
    if positive_studies:
        labels = [s["label"] for s in positive_studies[:3]]
        sample_finding = positive_studies[0]["extraction"].claim_summary
        if len(sample_finding) > 150:
            sample_finding = sample_finding[:150] + "..."
        bullets.append(f"Studies showing positive outcomes report: {sample_finding} [{', '.join(labels)}]")
    
    if negative_studies:
        labels = [s["label"] for s in negative_studies[:3]]
        sample_finding = negative_studies[0]["extraction"].claim_summary
        if len(sample_finding) > 150:
            sample_finding = sample_finding[:150] + "..."
        bullets.append(f"Studies finding negative or null effects: {sample_finding} [{', '.join(labels)}]")
    
    # Add study type breakdown if we have variety
    study_types = {}
    for extraction in extractions:
        st = extraction.study_type.replace("_", " ").title()
        if st not in study_types:
            study_types[st] = []
        study_types[st].append(citation_map.get(extraction.paper_id, "Unknown"))
    
    if len(study_types) > 1:
        type_breakdown = "; ".join([f"{k} ({len(v)})" for k, v in study_types.items()])
        bullets.append(f"Study types reviewed: {type_breakdown}.")
    
    # Add sample size info if available
    sample_sizes = [e.sample_size for e in extractions if e.sample_size]
    if sample_sizes:
        total_n = sum(sample_sizes)
        bullets.append(f"Combined sample size across studies with reported N: {total_n:,} participants.")
    
    # Ensure we have enough bullets
    for extraction in extractions[:3]:
        if len(bullets) >= 6:
            break
        label = citation_map.get(extraction.paper_id, "Unknown")
        summary = extraction.claim_summary
        if len(summary) > 120:
            summary = summary[:120] + "..."
        bullets.append(f"{summary} [{label}]")

    # Build consensus statement
    if pos_count > 0 and neg_count > 0:
        evidence_consensus = f"Evidence is conflicting: {pos_count} studies show positive effects while {neg_count} show negative or null effects."
    elif pos_count > neg_count:
        evidence_consensus = f"Evidence generally supports positive effects, though methodological limitations apply."
    elif neg_count > pos_count:
        evidence_consensus = f"Evidence suggests limited or negative effects."
    else:
        evidence_consensus = "Evidence is inconclusive with mixed or unclear findings across studies."
    
    # Build limitations
    limitations = []
    all_limits = []
    for extraction in extractions:
        all_limits.extend(extraction.limitations)
    # Deduplicate and limit
    seen = set()
    for lim in all_limits:
        if lim not in seen and len(limitations) < 5:
            limitations.append(lim)
            seen.add(lim)
    if not limitations:
        limitations = ["Abstract-only review; full methods not assessed."]

    # Calculate confidence score
    high_bias = sum(1 for c in critiques if c.risk_of_bias == "high")
    medium_bias = sum(1 for c in critiques if c.risk_of_bias == "medium")
    
    # Start with base score based on number of studies
    base_score = min(70, 40 + len(papers) * 5)
    # Deduct for bias
    score = base_score - high_bias * 15 - medium_bias * 8
    # Deduct for conflicting evidence
    if pos_count > 0 and neg_count > 0:
        score -= 10
    score = max(20, min(85, score))

    rationale = []
    rationale.append(f"Based on {len(papers)} studies with {'primarily automated' if not pos_count and not neg_count else 'categorized'} extraction.")
    if high_bias:
        rationale.append(f"{high_bias} studies show high risk of bias.")
    if medium_bias:
        rationale.append(f"{medium_bias} studies show moderate risk of bias.")
    if pos_count > 0 and neg_count > 0:
        rationale.append("Conflicting findings reduce confidence.")

    return Synthesis(
        final_answer=bullets[:8],
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
    
    # Build enriched paper summaries with titles for context
    paper_summaries = []
    for paper in papers:
        extraction = next((e for e in extractions if e.paper_id == paper.paper_id), None)
        critique = next((c for c in critiques if c.paper_id == paper.paper_id), None)
        paper_summaries.append({
            "citation": citation_map.get(paper.paper_id, "Unknown"),
            "paper_id": paper.paper_id,
            "title": paper.title,
            "study_type": extraction.study_type if extraction else "unknown",
            "claim_summary": extraction.claim_summary if extraction else "No summary available",
            "effect_direction": extraction.effect_direction if extraction else None,
            "effect_size": extraction.effect_size_text if extraction else None,
            "sample_size": extraction.sample_size if extraction else None,
            "limitations": extraction.limitations if extraction else [],
            "risk_of_bias": critique.risk_of_bias if critique else "unknown",
            "bias_rationale": critique.rationale if critique else [],
        })
    
    prompt_parts = [
        f"RESEARCH QUESTION: {question}",
        "",
        "AVAILABLE EVIDENCE:",
        "Below are summaries of the retrieved studies. Use these to synthesize a comprehensive answer.",
        "",
        json.dumps(paper_summaries, indent=2, ensure_ascii=False),
        "",
        "CITATION KEY (use these labels in your response):",
        json.dumps(citation_map, indent=2, ensure_ascii=False),
        "",
        "YOUR TASK:",
        "1. Synthesize the evidence to directly answer the research question.",
        "2. Group related findings together - don't just list each paper separately.",
        "3. Highlight specific effect sizes, statistics, or quantitative findings where available.",
        "4. Identify areas of agreement and disagreement across studies.",
        "5. Weight your conclusions toward higher-quality evidence (meta-analyses > RCTs > observational).",
    ]
    
    if issues:
        prompt_parts.extend([
            "",
            "ISSUES TO FIX FROM PREVIOUS ATTEMPT:",
            *[f"- {issue}" for issue in issues],
        ])
    
    prompt_parts.extend([
        "",
        "Return a JSON object with: final_answer (list of 5-8 insightful bullets with citations), "
        "evidence_consensus (1-2 sentences on agreement level), top_limitations_overall (list of key weaknesses), "
        "confidence_score (0-100), confidence_rationale (list explaining score), citations_used (list of paper_ids).",
        "",
        "IMPORTANT: Return ONLY valid JSON, no other text."
    ])
    
    return "\n".join(prompt_parts)


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
            max_tokens=1500,
            temperature=0.3,
        )
        data = safe_json_loads(raw)
        return Synthesis.model_validate(data)
    except (LLMUnavailableError, LLMRequestError, ValueError):
        return _fallback_synthesis(question, papers, extractions, critiques)

