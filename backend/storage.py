from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas import RunResponse


def _runs_dir() -> Path:
    base = Path(__file__).resolve().parent / "data" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base


def save_run(run: RunResponse) -> Path:
    path = _runs_dir() / f"{run.run_id}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(run.model_dump(), handle, ensure_ascii=False, indent=2)
    return path


def load_run(run_id: str) -> dict[str, Any] | None:
    path = _runs_dir() / f"{run_id}.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
