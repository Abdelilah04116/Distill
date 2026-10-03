"""Tests de la relecture humaine du test set.

La boucle est interactive : `prompt` et `confirm` sont injectes, ce qui permet
de rejouer un scenario de relecture complet sans terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from distill.cli import app
from distill.config import TaskConfig
from distill.review import REVIEWED_KEY, load_rows, review, save_rows, stats_for

runner = CliRunner()


def scripted(answers: list[str]):  # type: ignore[no-untyped-def]
    """Rejoue une suite de reponses, puis valide tout le reste."""
    queue = list(answers)

    def prompt(message: str, default: str = "") -> str:
        return queue.pop(0) if queue else (default or "v")

    return prompt


def always(value: bool):  # type: ignore[no-untyped-def]
    def confirm(message: str, default: bool = False) -> bool:
        return value

    return confirm


def test_status_counts_what_remains(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    rows = load_rows(cfg.test_set)
    stats = stats_for(cfg, rows)
    assert stats.total == 5
    assert stats.reviewed == 0
    assert stats.remaining == 5
    assert stats.invalid == []


def test_status_flags_examples_that_break_the_schema(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    rows = load_rows(cfg.test_set)
    rows[1]["output"]["categorie"] = "inexistante"
    save_rows(cfg.test_set, rows)
    assert stats_for(cfg, load_rows(cfg.test_set)).invalid == [1]


def test_validating_everything_marks_the_file(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    stats = review(cfg, cfg.test_set, prompt=scripted([]), confirm=always(False))
    assert stats.reviewed == 5
    assert stats.remaining == 0
    assert all(row[REVIEWED_KEY] for row in load_rows(cfg.test_set))


def test_quitting_keeps_the_progress_already_made(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    stats = review(cfg, cfg.test_set, prompt=scripted(["v", "v", "q"]), confirm=always(False))
    assert stats.reviewed == 2
    assert stats.remaining == 3

    # Reprise : on ne repasse pas sur ce qui est deja relu.
    again = review(cfg, cfg.test_set, prompt=scripted([]), confirm=always(False))
    assert again.reviewed == 5


def test_a_correction_is_written_back(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    rows = load_rows(cfg.test_set)
    original = rows[0]["output"]["categorie"]

    # corriger le premier exemple : categorie, urgence, resume, puis tout valider
    prompt = scripted(["c", "facturation", "haute", "resume corrige", "q"])
    stats = review(cfg, cfg.test_set, prompt=prompt, confirm=always(False))

    updated = load_rows(cfg.test_set)[0]
    assert stats.corrected == 1
    assert updated["output"]["categorie"] == "facturation" != original
    assert updated["output"]["resume"] == "resume corrige"
    assert updated[REVIEWED_KEY] is True


def test_a_correction_outside_the_schema_is_refused(cfg: TaskConfig) -> None:
    """On ne peut pas introduire une valeur hors schema en relisant."""
    assert cfg.test_set is not None
    before = load_rows(cfg.test_set)[0]["output"]["categorie"]

    prompt = scripted(["c", "categorie-inventee", "haute", "x", "q"])
    review(cfg, cfg.test_set, prompt=prompt, confirm=always(False))

    after = load_rows(cfg.test_set)[0]
    assert after["output"]["categorie"] == before
    assert not after.get(REVIEWED_KEY), "un exemple refuse ne doit pas etre marque relu"


def test_deleting_requires_confirmation(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    review(cfg, cfg.test_set, prompt=scripted(["s", "q"]), confirm=always(False))
    assert len(load_rows(cfg.test_set)) == 5, "sans confirmation, rien n'est supprime"

    stats = review(cfg, cfg.test_set, prompt=scripted(["s", "q"]), confirm=always(True))
    assert stats.deleted == 1
    assert len(load_rows(cfg.test_set)) == 4


def test_limit_stops_after_n_examples(cfg: TaskConfig) -> None:
    assert cfg.test_set is not None
    stats = review(cfg, cfg.test_set, prompt=scripted([]), confirm=always(False), limit=2)
    assert stats.reviewed == 2


def test_the_file_stays_readable_by_the_pipeline(cfg: TaskConfig) -> None:
    """Le champ de relecture ne doit pas casser le chargement des exemples."""
    from distill.data import load_examples

    assert cfg.test_set is not None
    review(cfg, cfg.test_set, prompt=scripted([]), confirm=always(False))
    examples = load_examples(cfg.test_set)
    assert len(examples) == 5
    assert examples[0].output["categorie"]


def test_cli_review_status(cfg: TaskConfig) -> None:
    result = runner.invoke(app, ["review", "--task", str(cfg.source_path), "--status"])
    assert result.exit_code == 0
    assert "relus" in result.stdout
    assert "restants" in result.stdout


def test_cli_review_validates_interactively(cfg: TaskConfig) -> None:
    result = runner.invoke(
        app,
        ["review", "--task", str(cfg.source_path), "--limit", "2"],
        input="v\nv\n",
    )
    assert result.exit_code == 0
    assert "2/5 relus" in result.stdout
    assert "3 exemples restants" in result.stdout


def test_cli_review_without_a_test_set(task_dir: Path, tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    del raw["test_set"]
    (task_dir / "sans-test.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    result = runner.invoke(app, ["review", "--task", str(task_dir / "sans-test.yaml")])
    assert result.exit_code == 1


def test_save_rows_is_atomic(cfg: TaskConfig) -> None:
    """Pas de fichier temporaire laisse derriere, pas de contenu tronque."""
    assert cfg.test_set is not None
    rows = load_rows(cfg.test_set)
    save_rows(cfg.test_set, rows)
    assert not list(cfg.test_set.parent.glob("*.tmp"))
    assert len(cfg.test_set.read_text(encoding="utf-8").strip().splitlines()) == len(rows)
    json.loads(cfg.test_set.read_text(encoding="utf-8").splitlines()[0])


@pytest.mark.parametrize("answer", ["", "v", "V", "valider"])
def test_default_answer_validates(cfg: TaskConfig, answer: str) -> None:
    assert cfg.test_set is not None
    stats = review(cfg, cfg.test_set, prompt=scripted([answer]), confirm=always(False), limit=1)
    assert stats.reviewed == 1
