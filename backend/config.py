from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_model: str = os.getenv("HF_MODEL", "")
    hf_timeout_s: float = _env_float("HF_TIMEOUT_S", 60)
    hf_max_concurrency: int = _env_int("HF_MAX_CONCURRENCY", 5)

    semantic_scholar_base_url: str = os.getenv(
        "SEMANTIC_SCHOLAR_BASE_URL", "https://api.semanticscholar.org/graph/v1"
    )
    semantic_scholar_timeout_s: float = _env_float("SEMANTIC_SCHOLAR_TIMEOUT_S", 20)
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    # Minimum gap between Semantic Scholar requests. The unauthenticated pool is
    # shared and returns 429 well below 1 req/s, so default conservatively.
    semantic_scholar_min_interval_s: float = _env_float(
        "SEMANTIC_SCHOLAR_MIN_INTERVAL_S", 1.2 if os.getenv("SEMANTIC_SCHOLAR_API_KEY") else 3.0
    )
    semantic_scholar_user_agent: str = os.getenv(
        "SEMANTIC_SCHOLAR_USER_AGENT", "multi-agent-research-assistant"
    )
    cors_origins_raw: str = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
