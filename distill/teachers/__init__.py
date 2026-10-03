"""Backends professeur : API hebergee ou modele local."""

from __future__ import annotations

from ..cache import DiskCache
from ..config import TeacherConfig
from ..costs import CostTracker
from .base import BaseTeacher, LLMResponse, Prompt


def build_teacher(
    cfg: TeacherConfig,
    *,
    cache: DiskCache | None = None,
    tracker: CostTracker | None = None,
) -> BaseTeacher:
    """Instancie le professeur decrit par task.yaml.

    Import tardif : on ne charge que le backend reellement utilise, pour qu'un
    run 100 % local n'ait pas besoin du SDK Anthropic et inversement.
    """
    if cfg.provider == "anthropic":
        from .anthropic_teacher import AnthropicTeacher

        return AnthropicTeacher(cfg, cache=cache, tracker=tracker)
    if cfg.provider == "openai_compatible":
        from .openai_compatible import OpenAICompatibleTeacher

        return OpenAICompatibleTeacher(cfg, cache=cache, tracker=tracker)
    raise ValueError(f"provider de professeur inconnu : {cfg.provider}")


__all__ = ["BaseTeacher", "LLMResponse", "Prompt", "build_teacher"]
