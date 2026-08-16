from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from typing import List

from pydantic import ValidationError

from ..llm import ChatLLM, LLMRequestError, LLMUnavailableError
from ..schemas import Critique, Paper, StudyExtraction, Synthesis
from ..utils import build_citation_map, safe_json_loads

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
- evidence_consensus MUST include at least one inline citation.
- citations_used MUST contain exactly the paper_ids referenced by inline citations.

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
    return build_citation_map(papers)


def _citation_text(
    extractions: List[StudyExtraction], citation_map: dict[str, str], limit: int = 4
) -> str:
    labels = list(
        dict.fromkeys(
            citation_map[extraction.paper_id]
            for extraction in extractions
            if extraction.paper_id in citation_map
        )
    )[:limit]
    return f"[{', '.join(labels)}]" if labels else ""


def _cited_paper_ids(
    texts: List[str], papers: List[Paper], citation_map: dict[str, str]
) -> List[str]:
    used_labels: set[str] = set()
    for text in texts:
        for group in re.findall(r"\[([^\]]+)\]", text):
            used_labels.update(part.strip() for part in group.split(","))
    return [
        paper.paper_id
        for paper in papers
        if citation_map.get(paper.paper_id) in used_labels
    ]


# Keys the model uses for the prose part of a bullet when it returns objects
# instead of plain strings.
_TEXT_KEYS = ("text", "point", "bullet", "content", "statement", "claim", "finding", "answer")
# Keys holding the citation label(s) alongside that prose.
_CITE_KEYS = ("citation", "citations", "cite", "cites", "refs", "references", "source", "sources")


def _flatten_citations(value: object) -> List[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        labels: List[str] = []
        for item in value:
            labels.extend(_flatten_citations(item))
        return labels
    if isinstance(value, dict):
        for key in ("label", "citation", "id", "paper_id", "name"):
            if key in value:
                return _flatten_citations(value[key])
    return []


def _stringify_bullet(item: object) -> str:
    """Render one final_answer entry as text with inline [Citations].

    Models frequently return {"text": ..., "citation": ...} objects rather than
    the plain strings the schema requires. Dropping those responses loses a good
    synthesis, so fold the citation back into the sentence instead.
    """
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        return str(item).strip()

    text = ""
    for key in _TEXT_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip()
            break
    if not text:
        # Unrecognised shape: use the longest string value present.
        strings = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
        if not strings:
            return ""
        text = max(strings, key=len)

    labels: List[str] = []
    for key in _CITE_KEYS:
        if key in item:
            labels.extend(_flatten_citations(item[key]))

    # Keep labels the prose already cites inline from being appended twice.
    existing = set()
    for group in re.findall(r"\[([^\]]+)\]", text):
        existing.update(part.strip() for part in group.split(","))
    new_labels = [label for label in dict.fromkeys(labels) if label and label not in existing]
    if new_labels:
        text = f"{text} [{', '.join(new_labels)}]"
    return text


def _normalize_synthesis_payload(
    data: object, papers: List[Paper], citation_map: dict[str, str]
) -> dict:
    """Coerce a raw LLM payload into the shape Synthesis expects.

    The schema is strict (list[str], int, paper ids), but model output varies:
    bullets arrive as objects, consensus as a list, scores as "85%" strings, and
    citations_used as display labels rather than paper ids. Normalising here
    keeps a usable synthesis instead of silently falling back to heuristics.
    """
    if not isinstance(data, dict):
        raise ValueError("Synthesis payload is not a JSON object")

    payload = dict(data)

    bullets = payload.get("final_answer")
    if isinstance(bullets, (str, dict)):
        bullets = [bullets]
    if isinstance(bullets, list):
        payload["final_answer"] = [
            text for text in (_stringify_bullet(item) for item in bullets) if text
        ]

    consensus = payload.get("evidence_consensus")
    if isinstance(consensus, (list, tuple)):
        parts = [_stringify_bullet(item) for item in consensus]
        payload["evidence_consensus"] = " ".join(part for part in parts if part)
    elif isinstance(consensus, dict):
        payload["evidence_consensus"] = _stringify_bullet(consensus)

    for key in ("top_limitations_overall", "confidence_rationale"):
        value = payload.get(key)
        if isinstance(value, (str, dict)):
            value = [value]
        if isinstance(value, list):
            payload[key] = [
                text for text in (_stringify_bullet(item) for item in value) if text
            ]

    score = payload.get("confidence_score")
    if isinstance(score, str):
        match = re.search(r"-?\d+(?:\.\d+)?", score)
        score = float(match.group()) if match else None
    if isinstance(score, float):
        score = round(score)
    if isinstance(score, bool) or not isinstance(score, int):
        score = None
    if score is not None:
        # Some models answer 0-1 instead of 0-100.
        payload["confidence_score"] = max(0, min(100, score))

    # citations_used must be paper ids; models often return display labels.
    label_to_id = {label: paper_id for paper_id, label in citation_map.items()}
    valid_ids = {paper.paper_id for paper in papers}
    raw_citations = payload.get("citations_used")
    resolved: List[str] = []
    for entry in _flatten_citations(raw_citations):
        if entry in valid_ids:
            resolved.append(entry)
        elif entry in label_to_id:
            resolved.append(label_to_id[entry])
    payload["citations_used"] = list(dict.fromkeys(resolved))

    return payload


def _has_citation(text: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]", text or ""))


class HallucinatedCitationError(ValueError):
    """Raised when a synthesis cites papers that are not in the evidence set."""


# Matches citation labels such as [Smith2024], [Smith2024a] or [Smithn.d.].
_LABEL_RE = re.compile(r"^\S+?(?:\d{4}|n\.d\.)(?:[a-z]|-\d+)?$")


def _assert_citations_exist(synthesis: Synthesis, citation_map: dict[str, str]) -> None:
    """Reject a synthesis that invents sources.

    Small instruction-tuned models will confidently cite papers that were never
    retrieved. Presenting that as evidence is worse than presenting nothing, so
    an unknown label discards the whole LLM synthesis in favour of the
    deterministic fallback.
    """
    allowed = set(citation_map.values())
    seen: set[str] = set()
    for text in list(synthesis.final_answer) + [synthesis.evidence_consensus]:
        for group in re.findall(r"\[([^\]]+)\]", text or ""):
            for part in group.split(","):
                label = part.strip()
                if label and _LABEL_RE.match(label):
                    seen.add(label)

    invented = sorted(seen - allowed)
    if invented:
        raise HallucinatedCitationError(
            f"Synthesis cited papers absent from the evidence set: {', '.join(invented)}"
        )


def _reconcile_citations(
    synthesis: Synthesis, papers: List[Paper], citation_map: dict[str, str]
) -> Synthesis:
    """Repair citation bookkeeping the referee would otherwise reject.

    A missing consensus citation or a citations_used list that disagrees with
    the inline labels are both mechanically derivable, so fixing them here
    avoids a wasted second LLM round-trip that rarely improves the prose.
    """
    updates: dict[str, object] = {}

    consensus = synthesis.evidence_consensus
    if consensus and not _has_citation(consensus):
        labels = list(
            dict.fromkeys(
                part.strip()
                for bullet in synthesis.final_answer
                for group in re.findall(r"\[([^\]]+)\]", bullet)
                for part in group.split(",")
                if part.strip()
            )
        )[:4]
        if labels:
            consensus = f"{consensus.rstrip()} [{', '.join(labels)}]"
            updates["evidence_consensus"] = consensus

    cited = _cited_paper_ids(list(synthesis.final_answer) + [consensus], papers, citation_map)
    if set(cited) != set(synthesis.citations_used):
        updates["citations_used"] = cited

    return synthesis.model_copy(update=updates) if updates else synthesis


def _no_evidence_synthesis(question: str, rate_limited: bool = False) -> Synthesis:
    """Generate a helpful synthesis when no papers were found.

    A rate-limited search is a very different situation from a genuinely empty
    one, so say which happened instead of implying the literature is thin.
    """
    if rate_limited:
        return Synthesis(
            final_answer=[
                "No answer could be produced because the Semantic Scholar search was rate limited.",
                "This is a temporary API limit, not a sign that research on this topic is unavailable.",
                "Wait a minute and ask again; the search should succeed once the limit resets.",
                "Setting SEMANTIC_SCHOLAR_API_KEY in backend/.env raises the request allowance.",
            ],
            evidence_consensus=(
                "No evidence was retrieved, so no consensus can be reported. "
                "The search itself failed rather than returning zero results."
            ),
            top_limitations_overall=[
                "Semantic Scholar returned HTTP 429 (rate limited) for every search attempt.",
                "No papers were analysed, so nothing here is evidence-based.",
            ],
            confidence_score=0,
            confidence_rationale=[
                "Confidence is 0 because the literature search never completed.",
            ],
            citations_used=[],
        )

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
        r"^is\s+(.+?)\s+(?:more|less|better|worse)\s+(?:effective|beneficial)?\s*than\s+(.+?)\??$",
        r"^(.+?)\s+(?:vs\.?|versus|compared to)\s+(.+?)\??$",
        r"^does\s+(.+?)\s+outperform\s+(.+?)\??$",
    ]
    for pattern in comparison_patterns:
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
    weighted_null = 0.0
    weighted_mixed = 0.0
    total_weight = 0.0
    
    for e in extractions:
        w = weights.get(e.study_type, 0.5)
        total_weight += w
        if e.effect_direction == "positive":
            weighted_positive += w
        elif e.effect_direction == "negative":
            weighted_negative += w
        elif e.effect_direction == "null":
            weighted_null += w
        else:
            weighted_mixed += w
    
    return {
        "positive": weighted_positive,
        "negative": weighted_negative,
        "null": weighted_null,
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
    hq_negative = [e for e in high_quality if e.effect_direction in ("negative", "null")]
    
    # Generate answer based on weighted evidence
    pos_pct = (weighted["positive"] / weighted["total"] * 100) if weighted["total"] > 0 else 0
    neg_pct = (
        (weighted["negative"] + weighted["null"]) / weighted["total"] * 100
        if weighted["total"] > 0
        else 0
    )
    
    # Build citation list for the answer
    citation_str = _citation_text(extractions, citation_map)
    
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


_THEME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("weight and body composition", ("weight", "fat mass", "lean mass", "body mass", "bmi", "obesity", "body composition")),
    ("metabolic health", ("metabolic", "insulin", "glucose", "blood sugar", "metabolism", "lipid", "cholesterol")),
    ("cardiovascular outcomes", ("cardiovascular", "heart", "blood pressure", "vascular", "cardiac")),
    ("cognitive function", ("cognitive", "cognition", "memory", "brain", "mental", "attention", "executive function", "reaction time")),
    ("physical performance", ("strength", "performance", "power", "endurance", "muscle", "exercise", "sprint", "training", "recovery", "soreness")),
    ("safety and tolerability", ("adverse", "side effect", "safety", "tolerab", "harm", "risk of")),
    ("adherence and sustainability", ("adherence", "compliance", "sustainable", "long-term", "dropout", "retention")),
)

# A claim only earns a direct quote if it reads like a result rather than a
# placeholder produced when no abstract was available.
_NON_FINDING_PREFIXES = ("study examining:", "no abstract available")


def _is_reportable_finding(claim: str) -> bool:
    text = (claim or "").strip()
    if len(text) < 30:
        return False
    return not text.lower().startswith(_NON_FINDING_PREFIXES)


def _theme_for(claim: str) -> str:
    lowered = claim.lower()
    for theme, keywords in _THEME_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return theme
    return "general outcomes"


def _shorten(claim: str, limit: int = 220) -> str:
    claim = " ".join(claim.split())
    if len(claim) <= limit:
        return claim
    return claim[:limit].rsplit(" ", 1)[0] + "..."


def _describe_direction(pos: int, neg: int, total: int) -> str:
    """Describe a theme's evidence without overstating a minority signal."""
    if pos and pos > total / 2:
        return f"most studies report benefits ({pos}/{total})"
    if neg and neg > total / 2:
        return f"most studies report no benefit ({neg}/{total})"
    if pos and neg:
        return f"findings conflict ({pos} positive vs {neg} negative/null of {total})"
    if pos:
        return f"{pos} of {total} studies report benefits; the rest are unclear"
    if neg:
        return f"{neg} of {total} studies report no benefit; the rest are unclear"
    return f"effects are unclear across {total} studies"


def _extract_key_themes(extractions: List[StudyExtraction], citation_map: dict[str, str]) -> List[str]:
    """Group studies by outcome theme and quote what they actually found.

    Each bullet leads with a real finding from an abstract rather than a bare
    study count, so the reader sees evidence instead of bookkeeping.
    """
    themes: dict[str, List[tuple[str, str | None, str]]] = {}
    for extraction in extractions:
        claim = extraction.claim_summary or ""
        label = citation_map.get(extraction.paper_id, "Unknown")
        themes.setdefault(_theme_for(claim), []).append(
            (label, extraction.effect_direction, claim)
        )

    # Report the best-evidenced themes first.
    ordered = sorted(themes.items(), key=lambda item: len(item[1]), reverse=True)

    bullets: List[str] = []
    for theme, studies in ordered:
        reportable = [s for s in studies if _is_reportable_finding(s[2])]
        pos = sum(1 for s in studies if s[1] == "positive")
        neg = sum(1 for s in studies if s[1] in ("negative", "null"))

        if not reportable:
            # Nothing quotable (e.g. abstracts were unavailable) - say so plainly
            # instead of implying a finding exists.
            labels = ", ".join(label for label, _, _ in studies[:3])
            bullets.append(
                f"**{theme.title()}**: {len(studies)} studies matched this outcome, but no "
                f"usable findings could be extracted from their abstracts. [{labels}]"
            )
            continue

        # Lead with a finding that has a clear direction where one exists.
        directional = [s for s in reportable if s[1] in ("positive", "negative", "null")]
        label, _, claim = (directional or reportable)[0]

        if len(studies) >= 2:
            others = [s[0] for s in reportable[1:3]]
            support = f" Other studies on this outcome: {', '.join(others)}." if others else ""
            bullets.append(
                f"**{theme.title()}**: {_shorten(claim)} [{label}] "
                f"Across {len(studies)} studies, {_describe_direction(pos, neg, len(studies))}.{support}"
            )
        else:
            bullets.append(f"**{theme.title()}**: {_shorten(claim)} [{label}]")

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
    if not extractions:
        labels = list(citation_map.values())[:4]
        citation = f"[{', '.join(labels)}]"
        return Synthesis(
            final_answer=[
                f"Papers were retrieved, but no study evidence could be extracted reliably. {citation}"
            ],
            evidence_consensus=f"No evidence consensus can be calculated from the extracted data. {citation}",
            top_limitations_overall=["Evidence extraction failed for all retrieved papers."],
            confidence_score=0,
            confidence_rationale=["No structured findings were available for synthesis."],
            citations_used=list(citation_map.keys())[:4],
        )
    
    # Analyze effect directions across studies
    positive_studies = []
    negative_studies = []
    mixed_null_studies = []
    
    for extraction in extractions:
        label = citation_map.get(extraction.paper_id, "Unknown")
        entry = {"label": label, "extraction": extraction}
        if extraction.effect_direction == "positive":
            positive_studies.append(entry)
        elif extraction.effect_direction in ("negative", "null"):
            negative_studies.append(entry)
        else:
            mixed_null_studies.append(entry)
    
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
        type_names = {
            "meta_analysis": "meta-analyses",
            "systematic_review": "systematic reviews",
            "RCT": "randomized controlled trials",
        }
        counts = Counter(e.study_type for e in high_quality)
        breakdown = ", ".join(
            f"{count} {type_names.get(study_type, study_type)}"
            for study_type, count in counts.most_common()
        )
        bullets.append(
            f"**Strongest evidence**: {breakdown} carry the most weight here. "
            f"[{', '.join(hq_labels)}]"
        )
    
    # 4. Add sample size context if significant. Reviews pool participants from
    # the primary studies, so summing both would double-count people.
    primary = [
        e
        for e in extractions
        if e.sample_size and e.study_type not in ("meta_analysis", "systematic_review")
    ]
    pooled = [
        e
        for e in extractions
        if e.sample_size and e.study_type in ("meta_analysis", "systematic_review")
    ]
    if primary and sum(e.sample_size for e in primary) > 500:
        total_n = sum(e.sample_size for e in primary)
        note = (
            f" A further {len(pooled)} review(s) pool participants from primary studies "
            f"and are excluded to avoid double-counting."
            if pooled
            else ""
        )
        bullets.append(
            f"**Sample coverage**: Combined N = {total_n:,} participants across "
            f"{len(primary)} primary studies with reported sample sizes.{note} "
            f"{_citation_text(primary, citation_map)}"
        )
    
    # 5. Note key limitations affecting conclusions
    if pos_count > 0 and neg_count > 0:
        bullets.append(
            f"**Important caveat**: Evidence is conflicting ({pos_count} positive vs "
            f"{neg_count} negative/null studies), suggesting individual variation or "
            f"methodological differences. {_citation_text(extractions, citation_map)}"
        )

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

    evidence_consensus = (
        f"{evidence_consensus} {_citation_text(extractions, citation_map)}".strip()
    )
    
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

    # Calculate confidence from evidence volume, study quality, bias, and
    # agreement. Bias penalties are proportional: counting them absolutely meant
    # every extra paper lowered the score, so a large body of medium-bias
    # evidence scored worse than a single unassessed study.
    high_bias = sum(1 for c in critiques if c.risk_of_bias == "high")
    medium_bias = sum(1 for c in critiques if c.risk_of_bias == "medium")
    assessed = len(critiques) or 1
    high_bias_share = high_bias / assessed
    medium_bias_share = medium_bias / assessed
    hq_share = len(high_quality) / len(extractions) if extractions else 0.0

    # Volume saturates: the 12th study adds far less than the 3rd.
    volume_points = min(20.0, 7.0 * math.log2(1 + len(papers)))
    quality_points = 25.0 * hq_share
    score = 35.0 + volume_points + quality_points

    # A body of evidence that is mostly high-bias loses more than one that is
    # merely unblinded or abstract-assessed.
    score -= 25.0 * high_bias_share
    score -= 8.0 * medium_bias_share

    # Agreement between studies matters more than the direction of the effect.
    if pos_pct > 70 or pos_pct < 30:
        score += 6.0
    elif 40 < pos_pct < 60:
        score -= 8.0

    # Very thin evidence should never look authoritative.
    if len(papers) < 3:
        score = min(score, 40.0)

    score = int(round(max(10, min(90, score))))

    rationale = []
    rationale.append(
        f"Synthesis of {len(papers)} studies, {len(high_quality)} of them "
        f"randomized trials or reviews ({hq_share:.0%} of the evidence)."
    )
    if pos_pct > 60:
        rationale.append(f"Evidence predominantly supports positive effects ({int(pos_pct)}% weighted).")
    elif pos_pct < 40:
        rationale.append(f"Limited support for benefits ({int(pos_pct)}% weighted positive).")
    else:
        rationale.append("Evidence is split, reducing confidence in definitive conclusions.")
    if high_bias:
        rationale.append(
            f"{high_bias} of {assessed} studies ({high_bias_share:.0%}) carry high risk of bias."
        )
    elif medium_bias:
        rationale.append(
            f"No study is high-bias, but {medium_bias} of {assessed} carry medium risk."
        )

    return Synthesis(
        final_answer=bullets[:8],
        evidence_consensus=evidence_consensus,
        top_limitations_overall=limitations,
        confidence_score=score,
        confidence_rationale=rationale,
        citations_used=_cited_paper_ids(
            bullets[:8] + [evidence_consensus], papers, citation_map
        ),
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
        "confidence_score (0-100), confidence_rationale (list explaining score), citations_used "
        "(exact list of paper_ids referenced by inline citations).",
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
    rate_limited: bool = False,
) -> Synthesis:
    # Always use no-evidence synthesis when no papers found
    if not papers:
        return _no_evidence_synthesis(question, rate_limited=rate_limited)
    
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
        citation_map = _build_citation_map(papers)
        result = Synthesis.model_validate(
            _normalize_synthesis_payload(data, papers, citation_map)
        )
        if not result.final_answer:
            raise ValueError("Synthesis contained no usable bullets")
        _assert_citations_exist(result, citation_map)
        result = _reconcile_citations(result, papers, citation_map)
        logger.info("Synthesizer LLM call succeeded")
        return result
    except HallucinatedCitationError as e:
        logger.warning(f"Discarding synthesis with fabricated citations: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)
    except ValidationError as e:
        logger.warning(f"Synthesizer schema mismatch after normalization: {e}")
        return _fallback_synthesis(question, papers, extractions, critiques)
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
