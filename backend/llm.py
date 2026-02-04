from __future__ import annotations

import asyncio
from typing import Any

from huggingface_hub import InferenceClient

from config import get_settings


class LLMUnavailableError(RuntimeError):
    pass


class LLMRequestError(RuntimeError):
    pass


class ChatLLM:
    def __init__(self) -> None:
        settings = get_settings()
        self._token = settings.hf_token
        self._model = settings.hf_model
        self._timeout = settings.hf_timeout_s
        self._client: InferenceClient | None = None

        if self._token and self._model:
            self._client = InferenceClient(model=self._model, token=self._token, timeout=self._timeout)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def chat(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> str:
        if not self.available:
            raise LLMUnavailableError("LLM is not configured. Set HF_TOKEN and HF_MODEL.")

        def _call() -> Any:
            assert self._client is not None
            return self._client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )

        try:
            resp = await asyncio.to_thread(_call)
        except Exception as exc:  # pragma: no cover - network error
            raise LLMRequestError(f"LLM request failed: {exc}") from exc

        try:
            choice = resp.choices[0]
            message = choice.message
            if isinstance(message, dict):
                content = message.get("content")
            else:
                content = getattr(message, "content", None)
            if not content:
                raise KeyError("Missing content")
            return content
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("Unexpected LLM response format") from exc
