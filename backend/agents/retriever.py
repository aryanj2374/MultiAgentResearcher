from __future__ import annotations

from typing import List

from semantic_scholar import SemanticScholarClient
from schemas import Paper


_CLIENT = SemanticScholarClient()


async def retrieve_papers(question: str, limit: int = 8) -> List[Paper]:
    return await _CLIENT.search_papers(question, limit=limit)
