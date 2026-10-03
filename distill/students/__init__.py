"""Modeles interrogeables pendant l'evaluation (eleve, baseline, professeur)."""

from __future__ import annotations

from .base import Prediction, Predictor

__all__ = ["HFPredictor", "OllamaPredictor", "Prediction", "Predictor", "TeacherPredictor"]


def __getattr__(name: str) -> object:
    """Import tardif : charger torch pour lire un rapport serait absurde."""
    if name == "HFPredictor":
        from .hf_runner import HFPredictor

        return HFPredictor
    if name == "OllamaPredictor":
        from .ollama_runner import OllamaPredictor

        return OllamaPredictor
    if name == "TeacherPredictor":
        from .teacher_predictor import TeacherPredictor

        return TeacherPredictor
    raise AttributeError(name)
