"""Professeur auto-heberge : tout endpoint compatible OpenAI /v1.

C'est le chemin du mode 100 % local du cahier des charges : un gros modele
open-weight servi par vLLM (ou Ollama) sur la machine, aucune donnee qui part
dans le cloud, aucune CGU de fournisseur a verifier avant d'entrainer sur les
sorties.

Pas de SDK ici : l'interface /v1/chat/completions est stable et universelle,
une requete HTTP directe evite d'imposer une dependance de plus.
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import TeacherConfig
from .base import BaseTeacher, LLMResponse, Prompt

DEFAULT_TIMEOUT = httpx.Timeout(600.0, connect=10.0)


class OpenAICompatibleTeacher(BaseTeacher):
    """Backend HTTP pour vLLM, Ollama, ou tout serveur exposant /v1."""

    provider_id = "openai_compatible"

    def __init__(self, cfg: TeacherConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        if not cfg.base_url:
            raise ValueError("teacher.base_url est obligatoire pour provider: openai_compatible")
        import os

        api_key = os.environ.get(cfg.api_key_env, "")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )

    def _call(self, prompt: Prompt, max_tokens: int) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
        }
        if self.cfg.temperature is not None:
            payload["temperature"] = self.cfg.temperature

        response = self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        body = response.json()

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"reponse sans 'choices' : {str(body)[:200]}")
        message = choices[0].get("message") or {}
        usage = body.get("usage") or {}
        return LLMResponse(
            text=str(message.get("content") or ""),
            input_tokens=int(usage.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage.get("completion_tokens", 0) or 0),
            stop_reason=choices[0].get("finish_reason"),
        )

    def _is_retryable(self, exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status >= 500 or status in (408, 409, 429)
        return False

    def close(self) -> None:
        self._client.close()
