"""Relecture humaine du test set.

Le protocole d'evaluation repose sur un test set valide a la main. C'est la
seule etape du pipeline qu'aucun modele ne peut faire a la place de l'humain :
si le test set est faux, tous les chiffres le sont, y compris ceux du
professeur.

Ce module en fait un geste outille plutot qu'une consigne. Il parcourt les
exemples un par un, affiche l'entree et la sortie attendue, et permet de
valider, corriger ou supprimer. L'avancement est ecrit a chaque reponse : on
peut s'arreter au bout de vingt exemples et reprendre le lendemain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.panel import Panel

from .config import TaskConfig
from .logging_utils import console
from .schema import build_output_model, enum_values, validate_output

REVIEWED_KEY = "reviewed"


@dataclass
class ReviewStats:
    total: int = 0
    reviewed: int = 0
    corrected: int = 0
    deleted: int = 0
    invalid: list[int] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.total - self.reviewed

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "reviewed": self.reviewed,
            "corrected": self.corrected,
            "deleted": self.deleted,
            "invalid": self.invalid,
        }


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Lit le fichier en conservant les champs inconnus, dont l'etat de relecture."""
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} JSON invalide : {exc}") from exc
    return rows


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Ecriture atomique : une interruption ne doit pas tronquer le test set."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    tmp.replace(path)


def stats_for(cfg: TaskConfig, rows: list[dict[str, Any]]) -> ReviewStats:
    """Etat de relecture d'un fichier, sans rien modifier."""
    model = build_output_model(cfg.output_schema)
    stats = ReviewStats(total=len(rows))
    for index, row in enumerate(rows):
        if row.get(REVIEWED_KEY):
            stats.reviewed += 1
        ok, _, _ = validate_output(model, row.get(cfg.output_key, {}))
        if not ok:
            stats.invalid.append(index)
    return stats


def render_example(cfg: TaskConfig, row: dict[str, Any], position: int, total: int) -> Panel:
    """Affiche une entree et sa sortie attendue, pretes a juger."""
    body = [str(row.get(cfg.input_key, "")).strip(), "", "[dim]sortie attendue[/]"]
    for key, value in (row.get(cfg.output_key) or {}).items():
        marker = "[bold]" if enum_values(cfg.output_schema, key) else ""
        close = "[/]" if marker else ""
        body.append(f"  {key} : {marker}{value}{close}")
    done = "[green]deja relu[/]" if row.get(REVIEWED_KEY) else "[yellow]a relire[/]"
    return Panel("\n".join(body), title=f"{position}/{total} — {done}", border_style="blue")


def correct(cfg: TaskConfig, row: dict[str, Any], prompt: Any) -> dict[str, Any]:
    """Demande une nouvelle valeur pour chaque champ et rend la sortie proposee.

    Ne touche pas a l'exemple : l'appelant n'ecrit qu'apres validation, sinon
    une correction refusee laisserait des valeurs hors schema dans le fichier.
    """
    output = dict(row.get(cfg.output_key) or {})
    for key in cfg.schema_fields:
        current = output.get(key)
        options = enum_values(cfg.output_schema, key)
        if options:
            console.print(f"  {key} — choix : {' | '.join(str(o) for o in options)}")
        value = prompt(f"  {key}", default=str(current) if current is not None else "")
        output[key] = value
    return output


def review(
    cfg: TaskConfig,
    path: Path,
    *,
    prompt: Any,
    confirm: Any,
    only_pending: bool = True,
    limit: int | None = None,
) -> ReviewStats:
    """Boucle de relecture. `prompt` et `confirm` sont injectes pour etre testables."""
    rows = load_rows(path)
    model = build_output_model(cfg.output_schema)
    stats = stats_for(cfg, rows)
    console.print(
        f"{path.name} : {stats.total} exemples, {stats.reviewed} deja relus, "
        f"{stats.remaining} restants"
    )
    if stats.invalid:
        console.print(f"[yellow]{len(stats.invalid)} exemples ne respectent pas le schema[/]")

    seen = 0
    index = 0
    while index < len(rows):
        row = rows[index]
        if only_pending and row.get(REVIEWED_KEY):
            index += 1
            continue
        if limit is not None and seen >= limit:
            break

        console.print(render_example(cfg, row, index + 1, len(rows)))
        answer = (
            prompt("[v]alider  [c]orriger  [s]upprimer  [q]uitter", default="v").strip().lower()[:1]
        )

        if answer == "q":
            break
        if answer == "s":
            if confirm("supprimer definitivement cet exemple ?", default=False):
                rows.pop(index)
                stats.deleted += 1
                stats.total -= 1
                save_rows(path, rows)
                seen += 1
                continue
            index += 1
            continue
        if answer == "c":
            proposed = correct(cfg, row, prompt)
            ok, normalised, error = validate_output(model, proposed)
            if not ok:
                console.print(f"[red]correction refusee[/] : {error}")
                console.print("[dim]l'exemple reste inchange.[/]")
                continue
            row[cfg.output_key] = normalised
            stats.corrected += 1

        row[REVIEWED_KEY] = True
        stats.reviewed += 1
        save_rows(path, rows)
        seen += 1
        index += 1

    save_rows(path, rows)
    final = stats_for(cfg, rows)
    final.corrected = stats.corrected
    final.deleted = stats.deleted
    return final
