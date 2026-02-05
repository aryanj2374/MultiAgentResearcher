from __future__ import annotations

import asyncio
import re
from typing import List, Tuple

from semantic_scholar import SemanticScholarClient
from schemas import Paper


_CLIENT = SemanticScholarClient()


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
    
    return queries


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
    }
    
    queries = _simplify_query(question)
    
    for query in queries:
        for attempt in range(max_retries):
            metadata["total_attempts"] += 1
            metadata["queries_tried"].append(query)
            
            papers = await _CLIENT.search_papers(query, limit=limit)
            
            if papers:
                metadata["successful_query"] = query
                return papers, metadata
            
            # Wait before retry with exponential backoff
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (2 ** attempt))
    
    # No papers found with any query
    return [], metadata
