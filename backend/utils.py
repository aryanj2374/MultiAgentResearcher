from __future__ import annotations

import json
import re
from typing import Any

from .schemas import Paper


def extract_json_fragment(text: str) -> str:
    if not text:
        raise ValueError("Empty response")

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text.strip()

    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        start = first_brace
    else:
        start = first_bracket

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in ("}", "]"):
            if not stack or char != stack[-1]:
                break
            stack.pop()
            if not stack:
                return text[start : idx + 1]

    return text.strip()


def safe_json_loads(text: str) -> Any:
    fragment = extract_json_fragment(text)
    return json.loads(fragment)


def first_sentence(text: str | None) -> str:
    if not text:
        return ""
    for sep in [". ", "? ", "! "]:
        if sep in text:
            return text.split(sep)[0].strip() + "."
    return text.strip()


def citation_label(authors: list[str], year: int | None) -> str:
    if authors:
        last = authors[0].split()[-1]
        last = re.sub(r"[\s,\[\]]+", "", last) or "Unknown"
    else:
        last = "Unknown"
    if year:
        return f"{last}{year}"
    return f"{last}n.d."


def build_citation_map(papers: list[Paper]) -> dict[str, str]:
    """Return paper-id -> unique, stable human-readable citation labels."""
    base_labels = [citation_label(paper.authors, paper.year) for paper in papers]
    totals = {label: base_labels.count(label) for label in set(base_labels)}
    seen: dict[str, int] = {}
    mapping: dict[str, str] = {}

    for paper, base in zip(papers, base_labels):
        seen[base] = seen.get(base, 0) + 1
        if totals[base] == 1:
            label = base
        else:
            index = seen[base]
            suffix = chr(ord("a") + index - 1) if index <= 26 else f"-{index}"
            label = f"{base}{suffix}"
        mapping[paper.paper_id] = label

    return mapping
