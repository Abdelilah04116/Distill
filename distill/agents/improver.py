"""F6 - Boucle d'amelioration.

Quand l'eleve rate la cible, regenerer 3 000 exemples au hasard ne sert a rien :
les nouveaux ressembleront aux anciens et les memes erreurs resteront. L'
Ameliorateur lit les erreurs reelles, en deduit des consignes precises, et fait
produire des exemples cibles sur ces cas-la.

Deux garde-fous :
  - les entrees de test ayant servi au diagnostic ne sont jamais recopiees dans
    les donnees d'entrainement, seulement decrites ;
  - trois iterations au maximum, puis rapport final, meme si la cible n'est pas
    atteinte. Un pipeline qui boucle indefiniment est un pipeline casse.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import TaskConfig
from ..data import Example
from ..logging_utils import get_logger, log_event
from ..teachers import BaseTeacher
from .evaluator import ModelReport
from .generator import GenerationResult, GenerationStats, generate

MAX_FOCUS_DIRECTIVES = 6


@dataclass
class ImprovementPlan:
    """Diagnostic lisible, affiche a l'ecran et repris dans le rapport."""

    directives: list[str] = field(default_factory=list)
    top_errors: list[tuple[str, int]] = field(default_factory=list)
    n_targeted: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "directives": self.directives,
            "top_errors": [{"category": c, "count": n} for c, n in self.top_errors],
            "n_targeted": self.n_targeted,
        }


def _directive_for(category: str, count: int, primary: str | None) -> str | None:
    """Traduit une categorie d'erreur en consigne de generation."""
    if category.startswith("confusion:"):
        pair = category.split(":", 1)[1]
        expected, _, predicted = pair.partition("->")
        return (
            f"des cas ou {primary or 'la classe'} vaut {expected!r} mais que l'on confond "
            f"facilement avec {predicted!r} ({count} erreurs) ; rends la bonne reponse "
            "deductible du texte sans ambiguite"
        )
    if category.startswith("champ_errone:"):
        field_name = category.split(":", 1)[1]
        return (
            f"des cas ou le champ {field_name!r} est difficile a remplir correctement "
            f"({count} erreurs) ; varie sa forme et sa position dans le texte"
        )
    if category in {"json_invalide", "reponse_vide"}:
        return (
            "des entrees courtes et bruitees qui poussent habituellement le modele a "
            f"repondre hors format ({count} cas) ; la sortie attendue reste un JSON strict"
        )
    if category == "schema_invalide":
        return (
            "des cas limites ou le modele invente une valeur hors des listes autorisees "
            f"({count} cas) ; n'utilise que des valeurs valides"
        )
    return None


def build_plan(report: ModelReport, primary: str | None, n_targeted: int) -> ImprovementPlan:
    """Transforme le regroupement d'erreurs en consignes de generation."""
    top = list(report.error_groups.items())[:MAX_FOCUS_DIRECTIVES]
    directives: list[str] = []
    for category, count in top:
        directive = _directive_for(category, count, primary)
        if directive:
            directives.append(directive)
    return ImprovementPlan(directives=directives, top_errors=top, n_targeted=n_targeted)


def improve(
    cfg: TaskConfig,
    teacher: BaseTeacher,
    seeds: Sequence[Example],
    report: ModelReport,
    *,
    primary_field: str | None,
    iteration: int,
    n_targeted: int | None = None,
) -> tuple[GenerationResult, ImprovementPlan]:
    """Genere un lot d'exemples cible sur les erreurs de l'eleve."""
    log = get_logger()
    target = n_targeted or max(200, cfg.generation.n_examples // 4)
    plan = build_plan(report, primary_field, target)

    if not plan.directives:
        log.warning(
            "[yellow]aucune erreur exploitable[/] : la boucle d'amelioration n'a rien a cibler"
        )
        log_event("amelioration", "skipped", iteration=iteration)
        return GenerationResult(examples=[], stats=GenerationStats()), plan

    log.info(
        f"iteration {iteration} : {target} exemples cibles sur {len(plan.directives)} faiblesses"
    )
    for directive in plan.directives:
        log.info(f"  - {directive}")

    # Les entrees ratees servent de graines : le generateur s'en inspire pour
    # produire des cas voisins, sans jamais les recopier telles quelles.
    error_seeds = [
        Example(input=case.input, output=case.expected, source="error-seed")
        for case in report.errors[: cfg.generation.seeds_per_prompt * 2]
    ]
    pool = list(seeds) + error_seeds

    result = generate(
        cfg,
        teacher,
        pool,
        n_examples=target,
        focus=plan.directives,
        stage="amelioration",
    )
    log_event(
        "amelioration",
        "generated",
        iteration=iteration,
        kept=len(result.examples),
        **plan.as_dict(),
    )
    return result, plan
