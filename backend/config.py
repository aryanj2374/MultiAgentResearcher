from __future__ import annotations

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_token: str = os.getenv("HF_TOKEN", "")
    hf_model: str = os.getenv("HF_MODEL", "")
    hf_timeout_s: float = float(os.getenv("HF_TIMEOUT_S", "60"))

    semantic_scholar_base_url: str = os.getenv(
        "SEMANTIC_SCHOLAR_BASE_URL", "https://api.semanticscholar.org/graph/v1"
    )
    semantic_scholar_timeout_s: float = float(os.getenv("SEMANTIC_SCHOLAR_TIMEOUT_S", "20"))
    semantic_scholar_api_key: str = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    semantic_scholar_user_agent: str = os.getenv(
        "SEMANTIC_SCHOLAR_USER_AGENT", "multi-agent-research-assistant"
    )


_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings()
    return _SETTINGS
