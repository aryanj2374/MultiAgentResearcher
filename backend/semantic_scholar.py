from __future__ import annotations

from typing import Dict, List

import httpx

from config import get_settings
from schemas import Paper


class SemanticScholarClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.semantic_scholar_base_url.rstrip("/")
        self._timeout = settings.semantic_scholar_timeout_s
        self._api_key = settings.semantic_scholar_api_key
        self._user_agent = settings.semantic_scholar_user_agent
        self._cache: Dict[str, List[Paper]] = {}

    async def search_papers(self, query: str, limit: int = 8) -> List[Paper]:
        if query in self._cache:
            return self._cache[query]

        url = f"{self._base_url}/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,authors,year,venue,url,abstract",
        }
        headers = {"User-Agent": self._user_agent}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError:
            return []

        papers = []
        for item in data.get("data", []):
            authors = [a.get("name", "") for a in item.get("authors", []) if a.get("name")]
            paper = Paper(
                paper_id=item.get("paperId", ""),
                title=item.get("title", ""),
                authors=authors,
                year=item.get("year"),
                venue=item.get("venue"),
                url=item.get("url"),
                abstract=item.get("abstract"),
            )
            if paper.paper_id and paper.title:
                papers.append(paper)

        self._cache[query] = papers
        return papers
