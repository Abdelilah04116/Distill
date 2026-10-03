"""F3 - Agent Critique.

Le Critique est ce qui separe un dataset synthetique exploitable d'un tas de
texte. Il enchaine quatre controles, du moins cher au plus cher :

  1. doublons exacts (gratuit) ;
  2. doublons semantiques et fuite du test set (embeddings) ;
  3. LLM-as-judge : chaque exemple note de 1 a 5, rejet sous le seuil ;
  4. equilibre des classes, signale sans corriger silencieusement.

L'ordre compte : juger des doublons au LLM serait payer deux fois pour le meme
exemple. Et l'isolation du test set se fait ici, avant tout entrainement.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import TaskConfig
from ..data import (
    Example,
    class_distribution,
    dedup_exact,
    remove_overlap,
    stratified_split,
    underrepresented,
)
from ..dedup import deduplicate
from ..logging_utils import get_logger, log_event
from ..schema import enum_values, extract_json, primary_label_field, schema_hint
from ..teachers import BaseTeacher, Prompt

JUDGE_SYSTEM = """Tu es un relecteur exigeant de donnees d'entrainement.

Tache concernee : {description}

Tu notes chaque exemple de 1 a 5 selon ces criteres, dans cet ordre :
5 - entree realiste et sortie exacte, l'exemple est utile a l'apprentissage ;
4 - correct, une imprecision mineure sans consequence ;
3 - sortie discutable ou entree artificielle ;
2 - sortie probablement fausse, ou entree qui ne ressemble a rien de reel ;
1 - sortie clairement fausse, incoherente, ou hors schema.

Sois severe sur un point precis : une sortie qui ne decoule pas de l'entree
merite au maximum 2, meme si l'entree est bien ecrite.

Tu reponds UNIQUEMENT par un tableau JSON :
[{{"index": 0, "score": 5, "reason": "..."}}, ...]
Une entree par exemple recu, dans le meme ordre, avec une raison de dix mots maximum."""

JUDGE_USER = """Schema attendu :
{schema}

Exemples a noter :
{items}"""


@dataclass
class CritiqueStats:
    received: int = 0
    exact_duplicates: int = 0
    semantic_duplicates: int = 0
    test_leaks: int = 0
    judged: int = 0
    judge_rejected: int = 0
    judge_mean_score: float = 0.0
    kept: int = 0
    embedder: str = "none"
    class_balance: dict[str, int] = field(default_factory=dict)
    underrepresented: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "exact_duplicates": self.exact_duplicates,
            "semantic_duplicates": self.semantic_duplicates,
            "test_leaks": self.test_leaks,
            "judged": self.judged,
            "judge_rejected": self.judge_rejected,
            "judge_mean_score": round(self.judge_mean_score, 2),
            "kept": self.kept,
            "embedder": self.embedder,
            "class_balance": self.class_balance,
            "underrepresented": self.underrepresented,
        }


@dataclass
class CritiqueResult:
    train: list[Example]
    test: list[Example]
    rejected: list[Example]
    stats: CritiqueStats
    warnings: list[str] = field(default_factory=list)


def judge(
    cfg: TaskConfig,
    teacher: BaseTeacher,
    examples: Sequence[Example],
    *,
    stage: str = "critique",
) -> tuple[list[Example], list[Example], float]:
    """Note les exemples et separe acceptes / rejetes.

    Un exemple que le juge n'a pas pu noter (reponse illisible, appel echoue)
    est garde : on ne jette pas une donnee pour une panne d'infrastructure.
    """
    critic = cfg.critic
    if not examples or critic.judge_sample_ratio <= 0:
        return list(examples), [], 0.0

    rng = random.Random(cfg.seed)
    pool = list(examples)
    if critic.judge_sample_ratio < 1.0:
        n = max(1, int(len(pool) * critic.judge_sample_ratio))
        judged_set = set(rng.sample(range(len(pool)), n))
    else:
        judged_set = set(range(len(pool)))

    batches: list[list[int]] = []
    current: list[int] = []
    for idx in sorted(judged_set):
        current.append(idx)
        if len(current) == critic.judge_batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    system = JUDGE_SYSTEM.format(description=cfg.description)
    hint = schema_hint(cfg.output_schema)
    prompts = [
        Prompt(
            system=system,
            user=JUDGE_USER.format(
                schema=hint,
                items="\n".join(
                    json.dumps(
                        {"index": position, "input": pool[idx].input, "output": pool[idx].output},
                        ensure_ascii=False,
                    )
                    for position, idx in enumerate(batch)
                ),
            ),
        )
        for batch in batches
    ]

    judge_teacher = teacher
    responses = judge_teacher.complete_many(prompts, stage=stage, max_tokens=2048)

    scores: dict[int, tuple[int, str]] = {}
    for batch, response in zip(batches, responses, strict=True):
        if response is None:
            continue
        try:
            payload = extract_json(response.text)
        except ValueError:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            position = row.get("index")
            if not isinstance(position, int) or not 0 <= position < len(batch):
                continue
            try:
                score = int(row.get("score", 0))
            except (TypeError, ValueError):
                continue
            scores[batch[position]] = (score, str(row.get("reason", ""))[:200])

    accepted: list[Example] = []
    rejected: list[Example] = []
    for idx, example in enumerate(pool):
        if idx not in scores:
            accepted.append(example)
            continue
        score, reason = scores[idx]
        example.judge_score = score
        example.judge_reason = reason
        (accepted if score >= critic.judge_threshold else rejected).append(example)

    mean = sum(s for s, _ in scores.values()) / len(scores) if scores else 0.0
    return accepted, rejected, mean


def critique(
    cfg: TaskConfig,
    teacher: BaseTeacher,
    candidates: Sequence[Example],
    *,
    held_out: Sequence[Example] = (),
    stage: str = "critique",
) -> CritiqueResult:
    """Nettoie un lot d'exemples et produit les ensembles train / test."""
    log = get_logger()
    stats = CritiqueStats(received=len(candidates))
    warnings: list[str] = []
    rejected: list[Example] = []

    # 1. doublons exacts
    kept, exact = dedup_exact(candidates)
    stats.exact_duplicates = exact

    # 2. recouvrement strict avec le test set fourni, puis dedup semantique
    if held_out:
        kept, overlaps = remove_overlap(kept, held_out)
        stats.test_leaks += overlaps

    dedup = deduplicate(
        kept,
        threshold=cfg.critic.dedup_threshold,
        embedding_model=cfg.critic.embedding_model,
        protected=held_out,
        use_qdrant=cfg.critic.use_qdrant,
    )
    stats.semantic_duplicates = len(dedup.removed)
    stats.test_leaks += len(dedup.leaked)
    stats.embedder = dedup.embedder
    rejected.extend(dedup.removed)
    rejected.extend(dedup.leaked)
    kept = dedup.kept
    log.info(
        f"dedup : {stats.exact_duplicates} doublons exacts, {stats.semantic_duplicates} semantiques, "
        f"{stats.test_leaks} fuites de test ecartees ({dedup.embedder})"
    )

    # 3. LLM-as-judge
    accepted, judge_rejected, mean_score = judge(cfg, teacher, kept, stage=stage)
    stats.judged = len(kept)
    stats.judge_rejected = len(judge_rejected)
    stats.judge_mean_score = mean_score
    rejected.extend(judge_rejected)
    kept = accepted
    if stats.judged:
        log.info(
            f"juge : note moyenne {mean_score:.2f}/5, {stats.judge_rejected} exemples rejetes "
            f"(seuil {cfg.critic.judge_threshold})"
        )

    # 4. test set : l'ensemble ecrit a la main prime toujours
    if held_out:
        train, test = kept, list(held_out)
    else:
        label_field = primary_label_field(cfg.output_schema, cfg.eval.primary_field)
        train, test = stratified_split(
            kept, test_ratio=cfg.critic.test_split, field=label_field, seed=cfg.seed
        )
        warnings.append(
            "aucun test set ecrit a la main : il a ete decoupe dans les donnees synthetiques, "
            "ce qui rend le benchmark moins credible. Ecris-en un, meme petit."
        )

    # 5. equilibre des classes
    label_field = primary_label_field(cfg.output_schema, cfg.eval.primary_field)
    stats.class_balance = class_distribution(train, label_field)
    stats.underrepresented = underrepresented(
        train,
        label_field,
        enum_values(cfg.output_schema, label_field or ""),
        cfg.critic.min_class_ratio,
    )
    if stats.underrepresented:
        warnings.append(
            "classes sous-representees dans l'entrainement : " + ", ".join(stats.underrepresented)
        )
        log.warning("[yellow]classes sous-representees[/] " + ", ".join(stats.underrepresented))

    if len(test) < 200 and held_out:
        warnings.append(
            f"test set de {len(test)} exemples : le protocole en recommande au moins 200."
        )

    stats.kept = len(train)
    log_event(stage, "critique", **stats.as_dict())
    return CritiqueResult(train=train, test=test, rejected=rejected, stats=stats, warnings=warnings)
