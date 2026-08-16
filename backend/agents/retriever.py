from __future__ import annotations

import asyncio
import re
from typing import List, Tuple

from ..semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarError,
    SemanticScholarRateLimited,
)
from ..schemas import Paper


_CLIENT = SemanticScholarClient()

# Base pause after a 429 when the response carries no Retry-After header.
_RATE_LIMIT_BACKOFF_S = 4.0


def _simplify_query(question: str) -> List[str]:
    """Generate alternative search queries from the original question."""
    queries = [question]
    
    # Remove question words and punctuation
    simplified = re.sub(r'^(does|do|is|are|what|how|can|could|should|would|will)\s+', '', question.lower())
    simplified = re.sub(r'\?+$', '', simplified).strip()
    if simplified and simplified != question.lower():
        queries.append(simplified)
    
    # Extract key noun phrases (simple heuristic)
    words = question.lower().split()
    # Filter out common stop words
    stop_words = {'does', 'do', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'of', 'to', 'for', 'and', 'or', 'with', 'by', 'from', 'that', 'this', 'it', 'be', 'have', 'has', 'had', 'been', 'being', 'was', 'were', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'whose'}
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    if len(keywords) >= 2:
        keyword_query = ' '.join(keywords[:5])  # Limit to 5 keywords
        if keyword_query not in queries:
            queries.append(keyword_query)
    
    return list(dict.fromkeys(query.strip() for query in queries if query.strip()))


# Words that carry no topical signal when deciding whether a paper is on-topic.
_TOPIC_STOP_WORDS = {
    "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "in", "on",
    "of", "to", "for", "and", "or", "with", "by", "from", "that", "this", "it",
    "be", "have", "has", "had", "been", "being", "will", "would", "could",
    "should", "can", "may", "might", "must", "what", "how", "why", "when",
    "where", "which", "who", "whom", "whose", "any", "there", "their", "its",
    "effect", "effects", "affect", "affects", "impact", "impacts", "influence",
    "benefit", "benefits", "outcome", "outcomes", "result", "results", "study",
    "studies", "research", "evidence", "adverse", "health", "improve",
    "improves", "improved", "increase", "decrease", "between", "among", "than",
    "more", "less", "better", "worse", "versus", "vs", "compared", "long",
    "term", "people", "adults", "human", "humans",
}


def _topic_terms(question: str) -> List[str]:
    """Content words that a relevant paper should plausibly mention."""
    words = re.findall(r"[a-z][a-z\-]{2,}", question.lower())
    return list(dict.fromkeys(w for w in words if w not in _TOPIC_STOP_WORDS))


# "effects of creatine on cognition" -> subject is "creatine". The subject is
# the intervention under study; the rest of the question names outcomes.
_SUBJECT_RE = re.compile(
    r"\b(?:effects?|impacts?|influences?|benefits?|risks?|safety)\s+of\s+(.+?)"
    r"(?:\s+on\b|\s+for\b|\s+in\b|[?.,]|$)",
    re.IGNORECASE,
)


def _subject_terms(question: str) -> List[str]:
    """Terms naming what the question is *about*, not what it measures.

    A paper that never mentions the intervention is off-topic even if it
    discusses the same outcome, so these terms are treated as mandatory.
    """
    match = _SUBJECT_RE.search(question)
    if match:
        terms = _topic_terms(match.group(1))
        if terms:
            return terms
    # No "effects of X on Y" shape: fall back to the first content term, which
    # in practice is the subject ("Creatine improves memory?").
    terms = _topic_terms(question)
    return terms[:1]


def _term_matches(term: str, text: str) -> bool:
    """Match a term allowing for simple morphological variation."""
    if term in text:
        return True
    # "physicals" -> "physical", "cognition" -> "cognitive"
    stem = term.rstrip("s")
    if len(stem) >= 5 and stem in text:
        return True
    if len(term) >= 6 and term[:6] in text:
        return True
    return False


def score_relevance(paper: Paper, terms: List[str]) -> float:
    """Fraction of the question's content terms the paper appears to address.

    Title matches count double: Semantic Scholar's relevance search happily
    returns papers that merely share one generic word with the query, and those
    off-topic papers otherwise flow through the whole pipeline as evidence.
    """
    if not terms:
        return 1.0
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    haystack = f"{title} {abstract}"

    score = 0.0
    for term in terms:
        if _term_matches(term, title):
            score += 1.0
        elif _term_matches(term, haystack):
            score += 0.5
    return score / len(terms)


def _mentions_subject(paper: Paper, subject_terms: List[str]) -> bool:
    haystack = f"{(paper.title or '').lower()} {(paper.abstract or '').lower()}"
    return any(_term_matches(term, haystack) for term in subject_terms)


def filter_relevant(
    papers: List[Paper], question: str, min_score: float = 0.25
) -> Tuple[List[Paper], List[Paper]]:
    """Split papers into (kept, dropped) by topical overlap with the question.

    A paper must mention the question's subject and clear a modest overall
    overlap bar. Semantic Scholar readily returns papers sharing only a generic
    outcome word, and without this gate they reach the synthesizer as evidence.
    """
    terms = _topic_terms(question)
    subject = _subject_terms(question)
    if not terms:
        return list(papers), []

    kept: List[Paper] = []
    dropped: List[Paper] = []
    scored: List[Tuple[float, Paper]] = []
    for paper in papers:
        score = score_relevance(paper, terms)
        scored.append((score, paper))
        if score >= min_score and (not subject or _mentions_subject(paper, subject)):
            kept.append(paper)
        else:
            dropped.append(paper)

    # Never return nothing: if the gate rejects everything, keep the best few so
    # the pipeline degrades to "weak evidence" rather than "no evidence".
    if not kept and scored:
        scored.sort(key=lambda item: item[0], reverse=True)
        kept = [paper for _, paper in scored[:3]]
        dropped = [paper for _, paper in scored[3:]]
    return kept, dropped


async def retrieve_papers(
    question: str, 
    limit: int = 8,
    max_retries: int = 3
) -> Tuple[List[Paper], dict]:
    """
    Retrieve papers with query reformulation and retry logic.
    
    Returns:
        Tuple of (papers list, metadata dict with search info)
    """
    metadata = {
        "original_query": question,
        "queries_tried": [],
        "total_attempts": 0,
        "successful_query": None,
        "errors": [],
    }
    
    queries = _simplify_query(question)
    
    for query in queries:
        for attempt in range(max(1, max_retries)):
            metadata["total_attempts"] += 1
            metadata["queries_tried"].append(query)

            try:
                papers = await _CLIENT.search_papers(query, limit=limit)
            except SemanticScholarRateLimited as exc:
                metadata["errors"].append(str(exc))
                metadata["rate_limited"] = True
                if attempt < max_retries - 1:
                    # A 429 needs a real pause; the generic backoff below is far
                    # too short and simply earns another 429.
                    await asyncio.sleep(exc.retry_after or _RATE_LIMIT_BACKOFF_S * (attempt + 1))
                continue
            except SemanticScholarError as exc:
                metadata["errors"].append(str(exc))
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                continue
            
            if papers:
                relevant, dropped = filter_relevant(papers, question)
                metadata["successful_query"] = query
                metadata["retrieved_count"] = len(papers)
                metadata["dropped_as_irrelevant"] = [
                    {"paper_id": p.paper_id, "title": p.title} for p in dropped
                ]
                return relevant, metadata

            # A successful empty response is not transient; reformulate the
            # query instead of repeating the same search and sleeping.
            break
    
    # No papers found with any query
    return [], metadata
