from __future__ import annotations

import json
import logging
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Critique, Paper, StudyExtraction, Synthesis
from utils import citation_label, safe_json_loads

logger = logging.getLogger(__name__)

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


def _detect_comparison_question(question: str) -> tuple[str | None, str | None]:
    """Detect if the question compares two treatments/interventions and extract them."""
    q_lower = question.lower()
    comparison_patterns = [
        (r"is\s+(.+?)\s+(?:more|less|better|worse)\s+(?:effective|beneficial)?\s*than\s+(.+?)[\s?]", True),
        (r"(.+?)\s+(?:vs\.?|versus|compared to|or)\s+(.+?)[\s?]", True),
        (r"does\s+(.+?)\s+outperform\s+(.+)", True),
    ]
    import re
    for pattern, is_comparison in comparison_patterns:
        match = re.search(pattern, q_lower)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    return None, None


def _weight_by_study_quality(extractions: List[StudyExtraction]) -> dict:
    """Weight studies by quality - meta-analyses and RCTs get higher weight."""
    weights = {
        "meta_analysis": 3.0,
        "systematic_review": 2.5,
        "RCT": 2.0,
        "observational": 1.0,
        "other": 0.8,
        "unknown": 0.5,
    }
    weighted_positive = 0.0
    weighted_negative = 0.0
    weighted_mixed = 0.0
    total_weight = 0.0
    
    for e in extractions:
        w = weights.get(e.study_type, 0.5)
        total_weight += w
        if e.effect_direction == "positive":
            weighted_positive += w
        elif e.effect_direction == "negative":
            weighted_negative += w
        else:
            weighted_mixed += w
    
    return {
        "positive": weighted_positive,
        "negative": weighted_negative,
        "mixed": weighted_mixed,
        "total": total_weight,
    }


def _generate_direct_answer(
    question: str,
    extractions: List[StudyExtraction],
    citation_map: dict[str, str],
) -> str:
    """Generate a direct answer to the research question based on weighted evidence."""
    weighted = _weight_by_study_quality(extractions)
    total = len(extractions)
    
    # Detect comparison questions
    item1, item2 = _detect_comparison_question(question)
    
    # Count high-quality studies
    high_quality = [e for e in extractions if e.study_type in ("meta_analysis", "systematic_review", "RCT")]
    hq_positive = [e for e in high_quality if e.effect_direction == "positive"]
    hq_negative = [e for e in high_quality if e.effect_direction == "negative"]
    
    # Generate answer based on weighted evidence
    pos_pct = (weighted["positive"] / weighted["total"] * 100) if weighted["total"] > 0 else 0
    neg_pct = (weighted["negative"] / weighted["total"] * 100) if weighted["total"] > 0 else 0
    
    # Build citation list for the answer
    all_labels = [citation_map.get(e.paper_id, "Unknown") for e in extractions[:4]]
    citation_str = f"[{', '.join(all_labels)}]"
    
    if item1 and item2:
        # Comparison question - provide comparative answer
        if pos_pct > 65:
            return f"Based on the evidence, {item1} appears to be more effective than {item2}. {len(hq_positive)} high-quality studies support this, with {int(pos_pct)}% of weighted evidence showing positive outcomes. {citation_str}"
        elif neg_pct > 65:
            return f"The evidence suggests {item1} is NOT more effective than {item2}. {len(hq_negative)} high-quality studies show no benefit or negative outcomes, representing {int(neg_pct)}% of weighted evidence. {citation_str}"
        elif pos_pct > neg_pct + 15:
            return f"There is moderate evidence suggesting {item1} may be slightly more effective than {item2}, though results are mixed. {int(pos_pct)}% of weighted evidence (favoring higher-quality studies) leans positive. {citation_str}"
        elif neg_pct > pos_pct + 15:
            return f"Current evidence leans toward {item2} being comparable or superior to {item1}, though more research is needed. {int(neg_pct)}% of weighted evidence shows no clear advantage. {citation_str}"
        else:
            return f"The evidence is inconclusive on whether {item1} is more effective than {item2}. Studies are roughly split, with high-quality evidence showing mixed results across {len(high_quality)} studies. {citation_str}"
    else:
        # Non-comparison question - provide direct answer
        if pos_pct > 70:
            return f"Yes, the evidence strongly supports a positive effect. {int(pos_pct)}% of weighted evidence (prioritizing meta-analyses and RCTs) shows beneficial outcomes across {total} studies. {citation_str}"
        elif pos_pct > 55:
            return f"The evidence suggests a moderate positive effect. {int(pos_pct)}% of weighted evidence supports benefits, though some studies show mixed or null results. {citation_str}"
        elif neg_pct > 55:
            return f"The evidence does not support significant benefits. {int(neg_pct)}% of weighted evidence shows negative or null effects across {total} studies. {citation_str}"
        else:
            if high_quality:
                hq_desc = "meta-analyses and RCTs" if len(hq_positive) > len(hq_negative) else "high-quality studies"
                return f"The evidence is mixed. While some {hq_desc} suggest benefits, others show minimal effects. More research is needed for definitive conclusions. {citation_str}"
            else:
                return f"Current evidence is inconclusive ({total} studies reviewed). No clear pattern emerges, though methodological limitations may partially explain conflicting results. {citation_str}"


def _extract_key_themes(extractions: List[StudyExtraction], citation_map: dict[str, str]) -> List[str]:
    """Extract and group key themes from studies rather than listing individual papers."""
    themes: dict[str, List[str]] = {}
    
    # Group by effect direction and extract common findings
    for e in extractions:
        # Try to extract the main outcome from claim summary
        claim = e.claim_summary.lower()
        
        # Categorize by common theme keywords
        if any(w in claim for w in ["weight", "fat", "body mass", "bmi", "obesity"]):
            theme = "weight and body composition"
        elif any(w in claim for w in ["metabolic", "insulin", "glucose", "blood sugar", "metabolism"]):
            theme = "metabolic health"
        elif any(w in claim for w in ["cardiovascular", "heart", "blood pressure", "cholesterol"]):
            theme = "cardiovascular outcomes"
        elif any(w in claim for w in ["cognitive", "memory", "brain", "mental"]):
            theme = "cognitive function"
        elif any(w in claim for w in ["adherence", "compliance", "sustainable", "long-term"]):
            theme = "adherence and sustainability"
        else:
            theme = "general outcomes"
        
        label = citation_map.get(e.paper_id, "Unknown")
        if theme not in themes:
            themes[theme] = []
        themes[theme].append((label, e.effect_direction, e.claim_summary))
    
    bullets = []
    for theme, studies in themes.items():
        if len(studies) >= 2:
            # Synthesize multiple studies on same theme
            pos = [s for s in studies if s[1] == "positive"]
            neg = [s for s in studies if s[1] in ("negative", "null")]
            labels = [s[0] for s in studies[:3]]
            
            if len(pos) > len(neg):
                bullets.append(f"**{theme.title()}**: Multiple studies show positive effects ({len(pos)}/{len(studies)} studies). [{', '.join(labels)}]")
            elif len(neg) > len(pos):
                bullets.append(f"**{theme.title()}**: Studies generally show limited or no significant effects ({len(neg)}/{len(studies)} studies). [{', '.join(labels)}]")
            else:
                bullets.append(f"**{theme.title()}**: Results are mixed across {len(studies)} studies examining this outcome. [{', '.join(labels)}]")
        elif studies:
            # Single study on theme
            label, direction, claim = studies[0]
            if len(claim) > 100:
                claim = claim[:100] + "..."
            bullets.append(f"**{theme.title()}**: {claim} [{label}]")
    
    return bullets[:5]


def _fallback_synthesis(
    question: str,
    papers: List[Paper],
    extractions: List[StudyExtraction],
    critiques: List[Critique],
) -> Synthesis:
    """Generate a structured synthesis when LLM is unavailable, focusing on answering the question directly."""
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
    
    total = len(extractions)
    pos_count = len(positive_studies)
    neg_count = len(negative_studies)
    
    # Build synthesized findings - START WITH DIRECT ANSWER
    bullets: List[str] = []
    
    # 1. Direct answer to the question (most important)
    direct_answer = _generate_direct_answer(question, extractions, citation_map)
    bullets.append(direct_answer)
    
    # 2. Add thematic synthesis of findings
    theme_bullets = _extract_key_themes(extractions, citation_map)
    bullets.extend(theme_bullets)
    
    # 3. Add study quality context
    high_quality = [e for e in extractions if e.study_type in ("meta_analysis", "systematic_review", "RCT")]
    if high_quality:
        hq_labels = [citation_map.get(e.paper_id, "Unknown") for e in high_quality[:3]]
        hq_types = list(set(e.study_type.replace("_", "-") for e in high_quality))
        bullets.append(f"**High-quality evidence**: {len(high_quality)} {'/'.join(hq_types)} provide stronger evidence. [{', '.join(hq_labels)}]")
    
    # 4. Add sample size context if significant
    sample_sizes = [e.sample_size for e in extractions if e.sample_size]
    if sample_sizes and sum(sample_sizes) > 500:
        total_n = sum(sample_sizes)
        bullets.append(f"**Sample coverage**: Combined N = {total_n:,} participants across {len(sample_sizes)} studies with reported sample sizes.")
    
    # 5. Note key limitations affecting conclusions
    if pos_count > 0 and neg_count > 0:
        bullets.append(f"**Important caveat**: Evidence is conflicting ({pos_count} positive vs {neg_count} negative/null studies), suggesting individual variation or methodological differences.")

    # Build consensus statement
    weighted = _weight_by_study_quality(extractions)
    pos_pct = (weighted["positive"] / weighted["total"] * 100) if weighted["total"] > 0 else 0
    
    if pos_pct > 65:
        evidence_consensus = f"Strong consensus: {int(pos_pct)}% of weighted evidence supports positive effects, with agreement across study types."
    elif pos_pct > 45:
        evidence_consensus = f"Moderate consensus with caveats: {int(pos_pct)}% of weighted evidence leans positive, but notable exceptions exist."
    elif pos_count > 0 and neg_count > 0:
        evidence_consensus = f"Limited consensus: Studies are divided ({pos_count} positive, {neg_count} negative/null), indicating the effect may be context-dependent."
    else:
        evidence_consensus = "Weak consensus: Most studies show unclear, mixed, or null effects. More rigorous research is needed."
    
    # Build limitations - focus on synthesis-level issues
    limitations = []
    if not high_quality:
        limitations.append("No meta-analyses or RCTs found; evidence quality is limited.")
    if len(extractions) < 5:
        limitations.append(f"Small evidence base ({len(extractions)} studies) limits generalizability.")
    if pos_count > 0 and neg_count > 0:
        limitations.append("Conflicting results suggest uncontrolled moderators or heterogeneity.")
    
    # Add unique per-study limitations
    all_limits = set()
    for extraction in extractions:
        for lim in extraction.limitations:
            if lim not in all_limits and len(limitations) < 5:
                limitations.append(lim)
                all_limits.add(lim)
    
    if not limitations:
        limitations = ["Abstract-only synthesis; detailed methodology not assessed."]

    # Calculate confidence score with quality weighting
    high_bias = sum(1 for c in critiques if c.risk_of_bias == "high")
    medium_bias = sum(1 for c in critiques if c.risk_of_bias == "medium")
    
    # Base score considers quantity, quality, and consistency
    base_score = min(70, 35 + len(papers) * 4 + len(high_quality) * 6)
    score = base_score - high_bias * 12 - medium_bias * 6
    
    # Adjust for consensus
    if pos_pct > 70 or pos_pct < 30:  # Clear direction = higher confidence
        score += 5
    elif 40 < pos_pct < 60:  # Split evidence = lower confidence
        score -= 8
    
    score = max(15, min(85, score))

    rationale = []
    rationale.append(f"Synthesis of {len(papers)} studies ({len(high_quality)} high-quality).")
    if pos_pct > 60:
        rationale.append(f"Evidence predominantly supports positive effects ({int(pos_pct)}% weighted).")
    elif pos_pct < 40:
        rationale.append(f"Limited support for benefits ({int(pos_pct)}% weighted positive).")
    else:
        rationale.append("Evidence is split, reducing confidence in definitive conclusions.")
    if high_bias:
        rationale.append(f"{high_bias} studies flagged for high bias risk.")

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
        logger.info(f"Synthesizer LLM raw response (first 500 chars): {raw[:500] if raw else 'EMPTY'}")
        data = safe_json_loads(raw)
        logger.info(f"Synthesizer parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        result = Synthesis.model_validate(data)
        logger.info("Synthesizer LLM call succeeded")
        return result
    except LLMUnavailableError as e:
        logger.warning(f"Synthesizer LLM unavailable: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)
    except LLMRequestError as e:
        logger.warning(f"Synthesizer LLM request error: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)
    except ValueError as e:
        logger.warning(f"Synthesizer JSON parse error: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)
    except Exception as e:
        logger.error(f"Synthesizer unexpected error: {type(e).__name__}: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)

