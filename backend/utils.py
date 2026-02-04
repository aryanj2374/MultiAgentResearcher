from __future__ import annotations

import json
from typing import Any


def extract_json_fragment(text: str) -> str:
    if not text:
        raise ValueError("Empty response")

    first_brace = text.find("{")
    first_bracket = text.find("[")

    if first_brace == -1 and first_bracket == -1:
        return text.strip()

    if first_bracket == -1 or (first_brace != -1 and first_brace < first_bracket):
        open_char, close_char = "{", "}"
        start = first_brace
    else:
        open_char, close_char = "[", "]"
        start = first_bracket

    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
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
    else:
        last = "Unknown"
    if year:
        return f"{last}{year}"
    return f"{last}n.d."
