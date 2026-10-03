"""F2 - Agent Generateur.

Produit quelques milliers d'exemples synthetiques a partir d'une vingtaine de
seeds. Le risque connu de ce genre de generation est l'uniformite : mille
variantes du meme email poli de reclamation. On le combat en forcant, a chaque
appel, un tirage explicite de persona, de ton, de longueur, de niveau de fautes
et de cas limite - le persona sampling du cahier des charges.

Le meme agent sert a la boucle d'amelioration (F6) : il suffit de lui passer
des consignes ciblees sur les erreurs observees.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import TaskConfig
from ..data import Example
from ..logging_utils import get_logger, log_event
from ..schema import build_output_model, extract_json, schema_hint, validate_output
from ..teachers import BaseTeacher, Prompt

DEFAULT_PERSONAS = [
    "un client particulier presse",
    "un client professionnel tres formel",
    "un client agace par un probleme recurrent",
    "une assistante de direction qui ecrit au nom de quelqu'un d'autre",
    "un client age peu a l'aise avec l'ecrit",
    "un developpeur qui decrit un bug precis",
    "un acheteur qui compare avant de signer",
    "un client fidele qui demande un geste commercial",
]

NOISE_DIRECTIVES = [
    "glisse deux ou trois fautes de frappe realistes",
    "oublie les accents comme sur un clavier mal configure",
    "ecris sans majuscules ni ponctuation finale",
    "laisse une phrase inachevee au milieu du message",
]

EDGE_CASE_DIRECTIVES = [
    "melange deux sujets dans le meme message, un seul doit l'emporter",
    "reste ambigu : le message pourrait tomber dans deux categories",
    "fais un message tres court, moins de dix mots",
    "ajoute une signature, une citation du message precedent et une piece jointe mentionnee",
    "ecris un message hors sujet qui doit tomber dans la categorie fourre-tout",
]

SYSTEM_TEMPLATE = """Tu es un generateur de donnees d'entrainement pour un petit modele specialise.

Tache a couvrir : {description}

Tu produis des exemples REALISTES et VARIES, chacun compose d'une entree et de la sortie attendue.
Regles strictes :
- la sortie respecte exactement le schema demande, sans champ en plus ni en moins ;
- l'entree est redigee en {language}, comme un vrai message, pas comme un exercice ;
- chaque exemple est different des autres : varie le vocabulaire, la structure, le contexte, les noms, les montants, les dates ;
- n'invente jamais de valeur hors des listes autorisees du schema ;
- ne numerote pas, ne commente pas, n'explique pas.

Tu reponds UNIQUEMENT par un tableau JSON de la forme :
[{{"input": "...", "output": {{...}}}}, ...]"""

USER_TEMPLATE = """Schema de sortie attendu :
{schema}

Exemples de reference (respecte ce style de correspondance entree -> sortie) :
{seeds}

Consignes de variation pour CE lot :
{directives}

Produis exactement {count} nouveaux exemples, tous differents des exemples de reference."""


@dataclass
class GenerationStats:
    """Ce qui s'est passe pendant la generation, repris dans le rapport."""

    requested: int = 0
    calls: int = 0
    parsed: int = 0
    invalid_json: int = 0
    schema_violations: int = 0
    kept: int = 0
    personas: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "calls": self.calls,
            "parsed": self.parsed,
            "invalid_json": self.invalid_json,
            "schema_violations": self.schema_violations,
            "kept": self.kept,
            "personas": self.personas,
        }


@dataclass
class GenerationResult:
    examples: list[Example]
    stats: GenerationStats


def _format_seeds(seeds: Sequence[Example]) -> str:
    import json

    return "\n".join(
        json.dumps({"input": s.input, "output": s.output}, ensure_ascii=False) for s in seeds
    )


def _directives(cfg: TaskConfig, rng: random.Random, focus: Sequence[str]) -> tuple[str, str]:
    """Tire les consignes de variation d'un lot. Retourne (texte, persona)."""
    gen = cfg.generation
    personas = gen.personas or DEFAULT_PERSONAS
    persona = rng.choice(personas)
    lines = [
        f"- auteur : {persona}",
        f"- ton : {rng.choice(gen.tones)}",
        f"- longueur : {rng.choice(gen.lengths)}",
    ]
    if rng.random() < gen.noise_ratio:
        lines.append(f"- qualite d'ecriture : {rng.choice(NOISE_DIRECTIVES)}")
    if rng.random() < gen.edge_case_ratio:
        lines.append(f"- difficulte : {rng.choice(EDGE_CASE_DIRECTIVES)}")
    for item in focus:
        lines.append(f"- CIBLE PRIORITAIRE : {item}")
    return "\n".join(lines), persona


def build_prompts(
    cfg: TaskConfig,
    seeds: Sequence[Example],
    *,
    n_examples: int,
    focus: Sequence[str] = (),
    rng: random.Random | None = None,
) -> list[tuple[Prompt, str]]:
    """Construit les prompts d'un run de generation, avec leur persona."""
    rng = rng or random.Random(cfg.seed)
    per_call = cfg.generation.examples_per_call
    n_calls = max(1, -(-n_examples // per_call))
    system = SYSTEM_TEMPLATE.format(description=cfg.description, language=cfg.language)
    hint = schema_hint(cfg.output_schema)

    prompts: list[tuple[Prompt, str]] = []
    for _ in range(n_calls):
        k = min(cfg.generation.seeds_per_prompt, len(seeds))
        chosen = rng.sample(list(seeds), k) if k else []
        directives, persona = _directives(cfg, rng, focus)
        user = USER_TEMPLATE.format(
            schema=hint,
            seeds=_format_seeds(chosen),
            directives=directives,
            count=per_call,
        )
        prompts.append((Prompt(system=system, user=user), persona))
    return prompts


def generate(
    cfg: TaskConfig,
    teacher: BaseTeacher,
    seeds: Sequence[Example],
    *,
    n_examples: int | None = None,
    focus: Sequence[str] = (),
    stage: str = "generation",
    rng: random.Random | None = None,
) -> GenerationResult:
    """Genere des exemples et ne garde que ceux qui respectent le schema."""
    log = get_logger()
    target = n_examples if n_examples is not None else cfg.generation.n_examples
    model = build_output_model(cfg.output_schema)
    prompts = build_prompts(cfg, seeds, n_examples=target, focus=focus, rng=rng)
    stats = GenerationStats(requested=target, calls=len(prompts))

    log.info(
        f"generation de ~{target} exemples en {len(prompts)} appels (concurrence {cfg.teacher.concurrency})"
    )
    responses = teacher.complete_many([p for p, _ in prompts], stage=stage)

    kept: list[Example] = []
    for (_, persona), response in zip(prompts, responses, strict=True):
        if response is None:
            continue
        try:
            payload = extract_json(response.text)
        except ValueError:
            stats.invalid_json += 1
            continue
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            stats.invalid_json += 1
            continue

        for row in payload:
            stats.parsed += 1
            if not isinstance(row, dict) or "input" not in row or "output" not in row:
                stats.schema_violations += 1
                continue
            ok, normalised, _ = validate_output(model, row["output"])
            if not ok or normalised is None:
                stats.schema_violations += 1
                continue
            text = str(row["input"]).strip()
            if not text:
                stats.schema_violations += 1
                continue
            kept.append(
                Example(
                    input=text,
                    output=normalised,
                    source="targeted" if focus else "generated",
                    persona=persona,
                )
            )
            stats.personas[persona] = stats.personas.get(persona, 0) + 1

    stats.kept = len(kept)
    log_event(stage, "generated", **stats.as_dict())
    if stats.parsed and stats.schema_violations / stats.parsed > 0.3:
        log.warning(
            f"[yellow]{stats.schema_violations}/{stats.parsed} exemples rejetes par le schema[/] : "
            "la description de la tache ou le schema est probablement ambigu"
        )
    return GenerationResult(examples=kept, stats=stats)
