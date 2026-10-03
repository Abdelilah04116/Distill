"""Entrainement LoRA / QLoRA et fusion de l'adaptateur."""

from __future__ import annotations

__all__ = ["TrainingResult", "merge_lora", "train_lora"]


def __getattr__(name: str) -> object:
    """Import tardif : torch ne doit pas etre charge par `distill --help`."""
    if name in {"train_lora", "TrainingResult"}:
        from . import lora_trainer

        return getattr(lora_trainer, name)
    if name == "merge_lora":
        from .merge import merge_lora

        return merge_lora
    raise AttributeError(name)
