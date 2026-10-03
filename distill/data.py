"""Representation et manipulation des exemples.

Un exemple est toujours la meme chose dans tout le pipeline : une entree
textuelle, une sortie structuree conforme au schema, et des metadonnees qui
disent d'ou il vient (seed, genere, cible sur une erreur).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

_WS = re.compile(r"\s+")


class Example(BaseModel):
    """Une paire entree / sortie attendue."""

    model_config = ConfigDict(extra="allow")

    input: str
    output: dict[str, Any]
    source: str = "generated"
    persona: str | None = None
    judge_score: int | None = None
    judge_reason: str | None = None

    @property
    def fingerprint(self) -> str:
        """Empreinte de l'entree normalisee, pour la deduplication exacte."""
        normalised = _WS.sub(" ", self.input.strip().lower())
        return hashlib.sha1(normalised.encode("utf-8")).hexdigest()

    def label(self, field: str | None) -> Any:
        return self.output.get(field) if field else None

    def to_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def load_examples(
    path: Path | str,
    *,
    input_key: str = "input",
    output_key: str = "output",
    source: str = "seed",
) -> list[Example]:
    """Lit un .jsonl d'exemples. Les lignes vides sont ignorees, les lignes
    invalides remontent avec leur numero pour etre corrigees a la main."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"fichier d'exemples introuvable : {p}")
    examples: list[Example] = []
    for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{lineno} JSON invalide : {exc}") from exc
        if input_key not in row or output_key not in row:
            raise ValueError(f"{p}:{lineno} il manque '{input_key}' ou '{output_key}'")
        examples.append(
            Example(
                input=str(row[input_key]),
                output=dict(row[output_key]),
                source=str(row.get("source", source)),
                persona=row.get("persona"),
            )
        )
    return examples


def save_examples(path: Path | str, examples: Iterable[Example]) -> int:
    """Ecrit un .jsonl et retourne le nombre de lignes ecrites."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_row(), ensure_ascii=False) + "\n")
            count += 1
    return count


def examples_from_inline(
    rows: Sequence[dict[str, Any]],
    *,
    input_key: str = "input",
    output_key: str = "output",
) -> list[Example]:
    """Exemples declares directement dans task.yaml."""
    out: list[Example] = []
    for i, row in enumerate(rows):
        if input_key not in row or output_key not in row:
            raise ValueError(f"seed_examples[{i}] : il manque '{input_key}' ou '{output_key}'")
        out.append(Example(input=str(row[input_key]), output=dict(row[output_key]), source="seed"))
    return out


def dedup_exact(examples: Sequence[Example]) -> tuple[list[Example], int]:
    """Supprime les doublons stricts (entree identique a la casse et aux espaces pres)."""
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in examples:
        fp = ex.fingerprint
        if fp in seen:
            continue
        seen.add(fp)
        kept.append(ex)
    return kept, len(examples) - len(kept)


def remove_overlap(
    candidates: Sequence[Example],
    protected: Sequence[Example],
) -> tuple[list[Example], int]:
    """Retire des candidats tout ce qui touche au test set.

    C'est la dedup croisee du cahier des charges : la premiere ligne de defense
    contre la fuite du test set dans l'entrainement.
    """
    blocked = {ex.fingerprint for ex in protected}
    kept = [ex for ex in candidates if ex.fingerprint not in blocked]
    return kept, len(candidates) - len(kept)


def class_distribution(examples: Sequence[Example], field: str | None) -> dict[str, int]:
    """Comptage par classe sur un champ categoriel."""
    if not field:
        return {}
    return dict(Counter(str(ex.output.get(field)) for ex in examples))


def underrepresented(
    examples: Sequence[Example],
    field: str | None,
    expected: Sequence[Any],
    min_ratio: float,
) -> list[str]:
    """Classes sous le seuil de representation, signalees par le Critique."""
    if not field or not expected:
        return []
    counts = class_distribution(examples, field)
    total = max(1, len(examples))
    return [str(label) for label in expected if counts.get(str(label), 0) / total < min_ratio]


def stratified_split(
    examples: Sequence[Example],
    *,
    test_ratio: float,
    field: str | None,
    seed: int = 42,
) -> tuple[list[Example], list[Example]]:
    """Separe train / test en gardant la repartition des classes.

    Utilise uniquement quand aucun test set ecrit a la main n'est fourni : le
    protocole d'evaluation prefere toujours un test set humain.
    """
    if test_ratio <= 0 or len(examples) < 2:
        return list(examples), []
    rng = random.Random(seed)
    buckets: dict[str, list[Example]] = {}
    for ex in examples:
        key = str(ex.output.get(field)) if field else "_"
        buckets.setdefault(key, []).append(ex)

    train: list[Example] = []
    test: list[Example] = []
    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_ratio)
        # Une classe rare garde au moins un exemple de test, mais jamais tous.
        if n_test == 0 and len(shuffled) > 1:
            n_test = 1
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def sample(examples: Sequence[Example], k: int, seed: int = 42) -> list[Example]:
    """Echantillon reproductible."""
    if k >= len(examples):
        return list(examples)
    return random.Random(seed).sample(list(examples), k)
