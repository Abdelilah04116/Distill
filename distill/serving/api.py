"""F7 - API FastAPI devant le modele local.

Un seul endpoint utile, `/predict`, qui prend du texte et rend du JSON valide.
La validation Pydantic est refaite ici : le modele quantifie peut deraper, et
une API qui renvoie du JSON casse a son appelant n'a aucun interet.

Configuration par variables d'environnement, pour que l'image Docker soit
parametrable sans la reconstruire :
    DISTILL_TASK   chemin du task.yaml (obligatoire)
    DISTILL_MODEL  nom du modele Ollama (defaut : celui du task.yaml)
    OLLAMA_HOST    URL du serveur Ollama (defaut : http://localhost:11434)
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import TaskConfig
from ..prompts import system_prompt
from ..schema import build_output_model, extract_json, validate_output
from ..students.ollama_runner import OllamaPredictor

MAX_BATCH = 64


class PredictRequest(BaseModel):
    input: str = Field(..., min_length=1, description="Le texte a traiter.")


class BatchRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    output: dict[str, Any] | None
    valid: bool
    latency_ms: float
    raw: str | None = None
    error: str | None = None


class _State:
    """Ressources chargees une seule fois au demarrage."""

    cfg: TaskConfig
    predictor: OllamaPredictor
    output_model: type[BaseModel]


state = _State()


def _load_state() -> None:
    task_path = os.environ.get("DISTILL_TASK")
    if not task_path:
        raise RuntimeError("DISTILL_TASK doit pointer vers un task.yaml")
    state.cfg = TaskConfig.load(Path(task_path))
    state.predictor = OllamaPredictor(state.cfg, os.environ.get("DISTILL_MODEL"))
    state.output_model = build_output_model(state.cfg.output_schema)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _load_state()
    yield
    state.predictor.close()


app = FastAPI(
    title="Distill",
    description="Modele specialise local, servi par Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)


def _predict_one(text: str) -> PredictResponse:
    started = time.perf_counter()
    prediction = state.predictor.predict(text)
    if prediction.error:
        raise HTTPException(status_code=503, detail=f"modele indisponible : {prediction.error}")
    try:
        payload = extract_json(prediction.raw)
    except ValueError as exc:
        return PredictResponse(
            output=None,
            valid=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=prediction.raw,
            error=f"sortie non parsable : {exc}",
        )
    ok, normalised, error = validate_output(state.output_model, payload)
    return PredictResponse(
        output=normalised,
        valid=ok,
        latency_ms=(time.perf_counter() - started) * 1000,
        raw=None if ok else prediction.raw,
        error=error,
    )


@app.post("/predict", response_model=PredictResponse, summary="Traite une entree")
def predict(request: PredictRequest) -> PredictResponse:
    return _predict_one(request.input)


@app.post("/predict/batch", response_model=list[PredictResponse], summary="Traite un lot")
def predict_batch(request: BatchRequest) -> list[PredictResponse]:
    return [_predict_one(text) for text in request.inputs]


@app.get("/health", summary="Etat du service")
def health() -> dict[str, Any]:
    available = state.predictor.is_available()
    return {
        "status": "ok" if available else "degraded",
        "task": state.cfg.name,
        "model": state.predictor.model,
        "ollama_reachable": available,
    }


@app.get("/schema", summary="Schema de sortie attendu")
def output_schema() -> dict[str, Any]:
    return {
        "task": state.cfg.name,
        "description": state.cfg.description,
        "schema": state.cfg.output_schema,
        "system_prompt": system_prompt(state.cfg),
    }
