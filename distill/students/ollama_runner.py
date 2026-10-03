"""Inference via Ollama, c'est-a-dire le modele tel qu'il tournera en production.

Les chiffres mesures par ce runner sont les plus honnetes a publier : ils
portent sur l'artefact quantifie reellement livre, pas sur les poids en
bfloat16 restes sur le GPU de developpement.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from ..config import TaskConfig
from ..prompts import chat_messages
from .base import Prediction, Predictor

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class OllamaPredictor(Predictor):
    """Eleve servi par Ollama, en local."""

    def __init__(
        self,
        cfg: TaskConfig,
        model: str | None = None,
        *,
        host: str = DEFAULT_HOST,
        kind: str = "student",
        name: str | None = None,
        num_predict: int = 256,
        timeout: float = 120.0,
    ) -> None:
        self.cfg = cfg
        self.model = model or cfg.model_slug
        self.kind = kind
        self.billing = "gpu"
        self.name = name or f"{self.model} (Ollama)"
        self.num_predict = num_predict
        self._client = httpx.Client(base_url=host.rstrip("/"), timeout=timeout)

    def predict(self, text: str) -> Prediction:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages(self.cfg, text),
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.num_predict},
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return Prediction(raw="", latency_s=time.perf_counter() - started, error=str(exc))
        latency = time.perf_counter() - started
        body = response.json()
        return Prediction(
            raw=str((body.get("message") or {}).get("content") or "").strip(),
            latency_s=latency,
            input_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=int(body.get("eval_count", 0) or 0),
        )

    def is_available(self) -> bool:
        """Vrai si le serveur Ollama repond et connait le modele."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
        except httpx.HTTPError:
            return False
        names = {m.get("name", "").split(":")[0] for m in response.json().get("models", [])}
        return self.model.split(":")[0] in names

    def close(self) -> None:
        self._client.close()
