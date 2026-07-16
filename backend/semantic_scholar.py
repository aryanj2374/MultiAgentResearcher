from __future__ import annotations

from typing import Dict, List, Tuple

import httpx

from .config import get_settings
from .schemas import Paper


class SemanticScholarError(RuntimeError):
    """Raised when Semantic Scholar could not complete a search request."""


class SemanticScholarClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._base_url = settings.semantic_scholar_base_url.rstrip("/")
        self._timeout = settings.semantic_scholar_timeout_s
        self._api_key = settings.semantic_scholar_api_key
        self._user_agent = settings.semantic_scholar_user_agent
        self._cache: Dict[Tuple[str, int], List[Paper]] = {}

    async def search_papers(self, query: str, limit: int = 8) -> List[Paper]:
        cache_key = (query, limit)
        if cache_key in self._cache:
            return list(self._cache[cache_key])

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
        except (httpx.HTTPError, ValueError) as exc:
            raise SemanticScholarError(f"Semantic Scholar search failed: {exc}") from exc

        if not isinstance(data, dict) or not isinstance(data.get("data", []), list):
            raise SemanticScholarError("Semantic Scholar returned an unexpected response format")

        papers = []
        seen_paper_ids: set[str] = set()
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            raw_authors = item.get("authors", [])
            if not isinstance(raw_authors, list):
                raw_authors = []
            authors = [
                author.get("name", "")
                for author in raw_authors
                if isinstance(author, dict) and author.get("name")
            ]
            try:
                paper = Paper(
                    paper_id=item.get("paperId", ""),
                    title=item.get("title", ""),
                    authors=authors,
                    year=item.get("year"),
                    venue=item.get("venue"),
                    url=item.get("url"),
                    abstract=item.get("abstract"),
                )
            except ValueError:
                continue
            if paper.paper_id and paper.title and paper.paper_id not in seen_paper_ids:
                papers.append(paper)
                seen_paper_ids.add(paper.paper_id)

        self._cache[cache_key] = papers
        return list(papers)
