from __future__ import annotations

import asyncio
import json
import re
from typing import List

from llm import ChatLLM, LLMRequestError, LLMUnavailableError
from schemas import Paper, StudyExtraction
from utils import first_sentence, safe_json_loads

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
    patterns = [
        r"\b[Nn]\s*=\s*(\d{2,5})\b",
        r"\bsample of (\d{2,5})\b",
        r"\b(\d{2,5})\s+participants\b",
        r"\b(\d{2,5})\s+subjects\b",
        r"\b(\d{2,5})\s+patients\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
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
    positive = any(word in lower for word in ["improve", "increase", "enhance", "benefit", "better"])
    negative = any(word in lower for word in ["worse", "decrease", "impair", "decline"])
    null = any(word in lower for word in ["no effect", "no significant", "not significant", "null"])
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    if null:
        return "null"
    return "unclear"


def _extract_key_findings_from_abstract(abstract: str) -> str | None:
    """Extract the most informative sentence from an abstract focusing on findings/conclusions."""
    if not abstract or len(abstract) < 50:
        return None
    
    # Split into sentences
    sentences = []
    for sep in [". ", ".\n"]:
        if sep in abstract:
            parts = abstract.split(sep)
            sentences = [s.strip() + "." for s in parts if s.strip()]
            break
    
    if not sentences:
        sentences = [abstract.strip()]
    
    # Prioritize sentences with findings/results keywords
    findings_keywords = [
        "found that", "showed that", "demonstrated", "revealed", "indicates",
        "results suggest", "findings", "concluded", "evidence suggests",
        "significantly", "improved", "reduced", "increased", "enhanced",
        "effect", "associated with", "correlated", "relationship",
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(kw in sentence_lower for kw in findings_keywords):
            # Found a findings sentence
            if len(sentence) > 200:
                # Truncate long sentences
                return sentence[:200].rsplit(" ", 1)[0] + "..."
            return sentence
    
    # If no findings sentence, try to find a conclusion-like sentence (often near end)
    for sentence in reversed(sentences[-3:]):
        if len(sentence) > 30 and not sentence.lower().startswith(("background", "introduction", "purpose", "objective", "aim")):
            if len(sentence) > 200:
                return sentence[:200].rsplit(" ", 1)[0] + "..."
            return sentence
    
    # Return the longest informative sentence
    best = max(sentences, key=len) if sentences else None
    if best and len(best) > 200:
        return best[:200].rsplit(" ", 1)[0] + "..."
    return best


def _fallback_extract(paper: Paper) -> StudyExtraction:
    abstract = paper.abstract or ""
    abstract_lower = abstract.lower()
    
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
    
    if "meta-analysis" in abstract_lower or "meta analysis" in abstract_lower:
        study_type = "meta_analysis"
    elif "systematic review" in abstract_lower:
        study_type = "systematic_review"
    elif "randomized" in abstract_lower or "randomised" in abstract_lower or "controlled trial" in abstract_lower:
        study_type = "RCT"
    elif "observational" in abstract_lower or "cohort" in abstract_lower or "case-control" in abstract_lower:
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
    """Extract study information from papers sequentially."""
    results: List[StudyExtraction] = []

    for paper in papers:
        if not llm.available:
            results.append(_fallback_extract(paper))
            continue

        try:
            raw = await llm.chat(EXTRACTOR_SYSTEM, _build_prompt(paper), max_tokens=700, temperature=0.1)
            data = safe_json_loads(raw)
            extraction = StudyExtraction.model_validate(data)
            if not extraction.url:
                extraction.url = paper.url
            results.append(extraction)
        except (LLMUnavailableError, LLMRequestError, ValueError):
            results.append(_fallback_extract(paper))

    return results
