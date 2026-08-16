from __future__ import annotations

import asyncio
import json
import re
from typing import List

from ..llm import ChatLLM, LLMRequestError, LLMUnavailableError
from ..schemas import Paper, StudyExtraction
from ..utils import first_sentence, safe_json_loads

EXTRACTOR_SYSTEM = """You extract structured study evidence from paper metadata and abstract.
Return ONLY valid JSON matching the schema. No commentary.
If unknown, use null or "unknown" for enums."""

def _limit_words(text: str, max_words: int = 25) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

def _extract_sample_size(text: str) -> int | None:
    if not text:
        return None
    count = r"(\d{1,3}(?:,\d{3})*|\d{1,7})"
    patterns = [
        rf"\b[Nn]\s*=\s*{count}\b",
        rf"\bsample of {count}\b",
        rf"\b{count}\s+participants\b",
        rf"\b{count}\s+subjects\b",
        rf"\b{count}\s+patients\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1).replace(",", ""))
            except ValueError:
                continue
    return None


def _extract_population(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"\b(adults?|older adults?|older adults?|elderly|children|adolescents|patients|athletes)\b",
        r"\bhealthy (men|women|adults|participants)\b",
        r"\bclinical (patients|sample)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def _extract_intervention(text: str) -> str | None:
    if not text:
        return None
    if "supplement" in text.lower():
        match = re.search(r"\b([a-zA-Z-]+)\s+supplementation\b", text, re.IGNORECASE)
        if match:
            return match.group(0)
    keywords = ["creatine", "caffeine", "exercise", "training", "diet", "sleep", "medication"]
    for keyword in keywords:
        if keyword in text.lower():
            return keyword.capitalize()
    return None


def _extract_comparison(text: str) -> str | None:
    if not text:
        return None
    if "placebo" in text.lower():
        return "Placebo"
    if "control" in text.lower():
        return "Control group"
    match = re.search(r"\bcompared to ([^.]+)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_outcomes(text: str) -> str | None:
    if not text:
        return None
    outcome_terms = [
        "cognitive", "memory", "attention", "reaction time", "executive function",
        "learning", "processing speed", "mental fatigue", "working memory",
    ]
    found = [term for term in outcome_terms if term in text.lower()]
    if found:
        return ", ".join(sorted(set(found)))
    match = re.search(r"\b(outcomes?|measured|assessed)\s+([^.;]+)", text, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return None


def _extract_effect_size_text(text: str) -> str | None:
    if not text:
        return None
    match = re.search(r"\b(Cohen'?s d|Hedges' g|OR|RR|HR)\s*=?\s*([0-9.]+)", text)
    if match:
        return match.group(0)
    match = re.search(r"\bp\s*[=<>]\s*0\.[0-9]+\b", text)
    if match:
        return match.group(0)
    return None


def _detect_effect_direction(text: str) -> str:
    if not text:
        return "unclear"
    lower = text.lower()

    null_patterns = [
        r"\bno\s+(?:statistically\s+)?significant\s+(?:effect|difference|change|improvement|increase|decrease)s?\b",
        r"\b(?:not|wasn['’]t|were not)\s+(?:statistically\s+)?significant\b",
        r"\bno\s+(?:effect|difference|benefit|improvement)s?\b",
        r"\b(?:did|does|do)\s+not\s+(?:improve|increase|enhance|benefit|reduce|decrease)\b",
        r"\bno\s+(?:evidence|association|relationship)\b",
        r"\b(?:was|were|is|are)\s+not\s+associated\b",
        r"\bnull\s+(?:effect|finding|result)s?\b",
    ]
    null = any(re.search(pattern, lower) for pattern in null_patterns)

    # Remove negated phrases before looking for directional keywords. Without
    # this, "no significant improvement" is incorrectly labelled positive.
    directional_text = lower
    for pattern in null_patterns:
        directional_text = re.sub(pattern, " ", directional_text)

    positive = any(
        word in directional_text
        for word in ["improve", "increase", "enhance", "benefit", "better"]
    )
    negative = any(
        word in directional_text
        for word in ["worse", "decrease", "impair", "decline"]
    )
    if positive and negative:
        return "mixed"
    if null and (positive or negative):
        return "mixed"
    if null:
        return "null"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unclear"


_SECTION_WORDS = (
    r"(?:background|introduction|context|rationale|objectives?|purpose"
    r"|aims?|goals?|methods?|methodology|materials and methods|design|setting"
    r"|participants|intervention|results?|findings|outcomes?|conclusions?"
    r"|interpretation|discussion|significance|summary)"
    r"(?:\s*/\s*(?:background|objectives?|aims?|methods?|results?|conclusions?|findings))*"
)

# Structured-abstract headers written with a colon, e.g. "Results:" or
# "Background/Objectives:".
_SECTION_HEADER_RE = re.compile(rf"(?:^|[\s.;])({_SECTION_WORDS})\s*:", re.IGNORECASE)

# Some journals omit the colon ("... decline. Methods Thirty-six older adults
# will be ..."). Require the capitalised form followed by a capitalised word so
# ordinary prose ("standard methods were used") is not treated as a header.
_SECTION_HEADER_NO_COLON_RE = re.compile(
    rf"(?:^|[\s.;])({_SECTION_WORDS})\b\s+(?=[A-Z(\[]|\d)"
)

# Sections that report what the study actually found, best first.
_FINDING_SECTIONS = ("conclusion", "result", "finding", "interpretation", "outcome")
# Sections that describe motivation or procedure rather than a finding.
_NON_FINDING_SECTIONS = (
    "background", "introduction", "context", "rationale", "objective",
    "purpose", "aim", "goal", "method", "design", "setting", "participant",
)

# Phrasing that signals a reported result.
_RESULT_MARKERS = (
    "found that", "we found", "showed that", "shown that", "demonstrated",
    "revealed", "indicated that", "results showed", "results suggest",
    "findings suggest", "concluded", "we conclude", "evidence suggests",
    "significantly", "significant improvement", "significant increase",
    "significant reduction", "no significant", "compared with", "compared to",
    "relative to placebo", "versus placebo", "associated with", "correlated",
    "resulted in", "led to", "improved", "reduced", "increased", "decreased",
    "enhanced", "greater than", "lower than", "effect size", "did not differ",
)

# Phrasing that signals motivation, framing, or an unanswered gap - never a finding.
_AIM_MARKERS = (
    "aimed to", "aim of", "aims to", "objective of", "purpose of",
    "this study evaluates", "this study examines", "this study investigates",
    "this study aimed", "this review", "we investigated whether", "we sought",
    "we aimed", "this investigation aimed", "the goal of", "we examine",
    "have not been", "has not been", "remains unclear", "remain unclear",
    "little is known", "is well-established", "are well-established",
    "is needed", "warrants", "future research", "we hypothesized",
    "we hypothesize", "we hypothesise", "hypothesized that", "hypothesize that",
    "the present study", "here we describe", "this paper",
)

# Quantitative evidence - a strong sign the sentence carries a real result.
_QUANT_RE = re.compile(
    r"(p\s*[=<>]\s*0?\.\d+"
    r"|\d+(?:\.\d+)?\s*%"
    r"|95\s*%\s*ci"
    r"|\bci\b"
    r"|\b(?:cohen'?s\s*d|hedges'?\s*g|smd|md|or|rr|hr|β|beta)\s*[=:]\s*-?\d"
    r"|\bn\s*=\s*\d+"
    r"|\bd\s*=\s*-?\d)",
    re.IGNORECASE,
)

# Study protocols describe planned work, not results.
_FUTURE_TENSE_RE = re.compile(
    r"\bwill\s+(?:be|include|consist|perform|receive|assess|measure|assign|"
    r"recruit|undergo|evaluate|compare|determine|provide|result|lead|improve|"
    r"increase|reduce|show|demonstrate|enhance|have|allow|help)\b"
    r"|\b(?:is|are)\s+expected\s+to\b"
    r"|\bthis\s+protocol\b"
    r"|\btrial\s+registration\b"
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[]|\d)")

_MAX_SUMMARY_CHARS = 240


def _split_sentences(text: str) -> List[str]:
    """Split prose into sentences without breaking on decimals or 'e.g.'."""
    if not text:
        return []
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", text)
    for abbr in ("e.g.", "i.e.", "vs.", "approx.", "Dr.", "et al.", "cf."):
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = _SENTENCE_SPLIT_RE.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _find_section_headers(abstract: str) -> list[re.Match[str]]:
    """Locate section headers, preferring the unambiguous colon form."""
    matches = list(_SECTION_HEADER_RE.finditer(abstract))
    if matches:
        return matches
    # Only fall back to colon-less headers when at least two are present, which
    # signals a genuinely structured abstract rather than a coincidental word.
    loose = [
        match
        for match in _SECTION_HEADER_NO_COLON_RE.finditer(abstract)
        if match.group(1)[:1].isupper()
    ]
    return loose if len(loose) >= 2 else []


def _split_sections(abstract: str) -> List[tuple[str, str]]:
    """Split a structured abstract into (section_name, body) pairs.

    Returns a single ("", abstract) pair when the abstract is unstructured.
    """
    matches = _find_section_headers(abstract)
    if not matches:
        return [("", abstract)]

    sections: List[tuple[str, str]] = []
    preamble = abstract[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for idx, match in enumerate(matches):
        name = match.group(1).lower()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(abstract)
        body = abstract[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


_LEADING_HEADER_RE = re.compile(rf"^{_SECTION_WORDS}\s*[:\-–]?\s+", re.IGNORECASE)


def _truncate(sentence: str) -> str:
    # A colon-less header can survive sentence splitting ("Results Consistent
    # increases in ..."); drop it so the summary reads as prose.
    sentence = _LEADING_HEADER_RE.sub("", sentence).strip()
    if len(sentence) <= _MAX_SUMMARY_CHARS:
        return sentence
    return sentence[:_MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."


def _score_finding_sentence(sentence: str, section: str) -> float:
    """Score how much a sentence reads like a reported finding."""
    lower = sentence.lower()
    score = 0.0

    if any(key in section for key in _FINDING_SECTIONS):
        score += 6.0
    elif any(key in section for key in _NON_FINDING_SECTIONS):
        score -= 5.0

    score += 2.0 * sum(1 for marker in _RESULT_MARKERS if marker in lower)
    score -= 3.0 * sum(1 for marker in _AIM_MARKERS if marker in lower)

    quant_hits = len(_QUANT_RE.findall(sentence))
    score += 2.5 * min(quant_hits, 3)

    # Registered protocols describe work not yet done, so future tense can never
    # be a finding.
    if _FUTURE_TENSE_RE.search(lower):
        score -= 8.0

    # Very short fragments rarely state a complete result.
    words = len(sentence.split())
    if words < 6:
        score -= 4.0
    elif words > 60:
        score -= 1.0

    # Citation-only or reference-heavy sentences are usually background.
    if re.search(r"\[\d+(?:\s*,\s*\d+)*\]", sentence):
        score -= 2.0

    return score


def _extract_key_findings_from_abstract(abstract: str) -> str | None:
    """Return the sentence that best states what the study actually found.

    Prefers Results/Conclusions sections and sentences carrying quantitative
    evidence; explicitly penalises Background/Objective framing so summaries
    stop reading like "This study aimed to evaluate ...".
    """
    if not abstract or len(abstract) < 50:
        return None

    candidates: List[tuple[float, int, str]] = []
    position = 0
    for section, body in _split_sections(abstract):
        for sentence in _split_sentences(body):
            position += 1
            if len(sentence) < 25:
                continue
            candidates.append((_score_finding_sentence(sentence, section), position, sentence))

    if not candidates:
        return None

    best_score, _, best_sentence = max(candidates, key=lambda item: (item[0], -item[1]))

    # Everything scored as framing/motivation: prefer a late sentence, which in
    # an unstructured abstract is usually the takeaway.
    if best_score <= 0:
        tail = [c for c in candidates if c[1] >= max(1, position - 2)]
        pool = tail or candidates
        _, _, best_sentence = max(pool, key=lambda item: item[0])

    return _truncate(best_sentence)


def _fallback_extract(paper: Paper) -> StudyExtraction:
    abstract = paper.abstract or ""
    study_description_lower = f"{paper.title}. {abstract}".lower()
    
    # Try to extract a meaningful finding from the abstract
    summary = _extract_key_findings_from_abstract(abstract)
    
    # If still no good summary, use first sentence but mark it
    if not summary:
        first = first_sentence(abstract)
        if first and len(first) > 20:
            summary = first
        else:
            # Last resort: use title but prefix it clearly
            summary = f"Study examining: {paper.title}" if paper.title else "No abstract available."
    
    if "meta-analysis" in study_description_lower or "meta analysis" in study_description_lower:
        study_type = "meta_analysis"
    elif "systematic review" in study_description_lower:
        study_type = "systematic_review"
    elif "randomized" in study_description_lower or "randomised" in study_description_lower or "controlled trial" in study_description_lower:
        study_type = "RCT"
    elif "observational" in study_description_lower or "cohort" in study_description_lower or "case-control" in study_description_lower:
        study_type = "observational"
    else:
        study_type = "unknown"

    apa = f"{paper.authors[0].split()[-1]} et al. ({paper.year or 'n.d.'}). {paper.title}. {paper.venue or 'Unknown venue'}." if paper.authors else f"Unknown ({paper.year or 'n.d.'}). {paper.title}."

    return StudyExtraction(
        paper_id=paper.paper_id,
        claim_summary=summary,
        study_type=study_type,
        population=_extract_population(abstract),
        sample_size=_extract_sample_size(abstract),
        intervention_exposure=_extract_intervention(abstract),
        comparison=_extract_comparison(abstract),
        outcomes=_extract_outcomes(abstract),
        effect_direction=_detect_effect_direction(abstract),
        effect_size_text=_extract_effect_size_text(abstract),
        key_snippet=_limit_words(summary, 25),
        limitations=[
            "Abstract-only extraction; methods and bias details may be missing.",
            "Sample size or comparator details may be incomplete.",
        ],
        apa_citation=apa,
        url=paper.url,
    )


def _build_prompt(paper: Paper) -> str:
    payload = {
        "paper": paper.model_dump(),
        "schema": {
            "effect_direction": ["positive", "negative", "mixed", "null", "unclear"],
            "study_type": [
                "meta_analysis",
                "systematic_review",
                "RCT",
                "observational",
                "other",
                "unknown",
            ],
        },
        "notes": [
            "Fill apa_citation as 'AuthorLastName et al. (Year). Title. Venue.'",
            "key_snippet <= 25 words (paraphrase allowed; if direct quote keep it short)",
            "limitations: 2-5 bullets if possible",
        ],
    }

    return (
        "Extract a StudyExtraction from this input.\n\n"
        f"INPUT JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return JSON ONLY."
    )


async def extract_all(papers: List[Paper], llm: ChatLLM) -> List[StudyExtraction]:
    """Extract study information concurrently, preserving paper order."""
    async def extract_one(paper: Paper) -> StudyExtraction:
        if not llm.available:
            return _fallback_extract(paper)

        try:
            raw = await llm.chat(EXTRACTOR_SYSTEM, _build_prompt(paper), max_tokens=700, temperature=0.1)
            data = safe_json_loads(raw)
            extraction = StudyExtraction.model_validate(data)
            extraction = extraction.model_copy(
                update={
                    # The paper id is pipeline identity, not an LLM-generated field.
                    "paper_id": paper.paper_id,
                    "url": extraction.url or paper.url,
                }
            )
            return extraction
        except (LLMUnavailableError, LLMRequestError, ValueError):
            return _fallback_extract(paper)

    return list(await asyncio.gather(*(extract_one(paper) for paper in papers)))
