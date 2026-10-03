"""Exceptions du pipeline Distill."""

from __future__ import annotations


class DistillError(Exception):
    """Erreur de base du pipeline."""


class ConfigError(DistillError):
    """task.yaml invalide ou incoherent."""


class BudgetExceeded(DistillError):
    """Le plafond de cout ou de duree defini dans task.yaml est atteint."""


class TeacherError(DistillError):
    """Le professeur a echoue (API, refus, reponse inexploitable)."""


class SchemaError(DistillError):
    """Le schema de sortie ne peut pas etre converti en modele Pydantic."""


class ExportError(DistillError):
    """La fusion LoRA, la quantification GGUF ou l'export Ollama a echoue."""


class TrainingError(DistillError):
    """L'entrainement a echoue ou les dependances sont absentes."""
