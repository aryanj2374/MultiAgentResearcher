from __future__ import annotations

import re
from typing import List

from ..schemas import Critique, Paper, Synthesis, Verification
from ..utils import build_citation_map


def _citation_map(papers: List[Paper]) -> dict[str, str]:
    return {label: paper_id for paper_id, label in build_citation_map(papers).items()}


def _extract_citations(text: str) -> List[str]:
    found = []
    for group in re.findall(r"\[(.*?)\]", text or ""):
        parts = [part.strip() for part in group.split(",") if part.strip()]
        found.extend(
            part
            for part in parts
            if re.match(r"^\S+(?:\d{4}|n\.d\.)(?:[a-z]|-\d+)?$", part)
        )
    return found


def verify_synthesis(
    synthesis: Synthesis,
    papers: List[Paper],
    critiques: List[Critique],
) -> Verification:
    # If no papers were found, skip full verification
    if not papers:
        return Verification(
            passed=True,
            issues=[],
            revised_synthesis=None,
        )
    
    issues: List[str] = []
    citation_map = _citation_map(papers)
    allowed = set(citation_map.keys())

    def check_text(label: str, text: str) -> None:
        citations = _extract_citations(text)
        if not citations:
            issues.append(f"Missing citation in {label}.")
            return
        for cite in citations:
            if cite not in allowed:
                issues.append(f"Unknown citation label '{cite}' in {label}.")

    for idx, bullet in enumerate(synthesis.final_answer, start=1):
        check_text(f"final_answer bullet {idx}", bullet)

    check_text("evidence_consensus", synthesis.evidence_consensus)

    used_labels = set()
    for section in (
        synthesis.final_answer
        + [synthesis.evidence_consensus]
    ):
        used_labels.update(_extract_citations(section))

    mapped_paper_ids = {citation_map[label] for label in used_labels if label in citation_map}
    unknown_paper_ids = set(synthesis.citations_used) - {paper.paper_id for paper in papers}
    if unknown_paper_ids:
        issues.append("citations_used contains unknown paper ids.")
    if set(synthesis.citations_used) != mapped_paper_ids:
        issues.append("citations_used does not match the citations referenced in text.")

    high_bias = sum(1 for c in critiques if c.risk_of_bias == "high")
    if high_bias and synthesis.confidence_score > 75:
        issues.append("Confidence score is high despite high risk-of-bias studies.")

    return Verification(passed=len(issues) == 0, issues=issues, revised_synthesis=None)
