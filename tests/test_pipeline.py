"""Tests d'integration : le graphe de bout en bout, le rapport et la CLI.

L'entrainement et l'export sont neutralises (ils demandent un GPU et
llama.cpp) ; tout le reste - generation, critique, isolation du test set,
evaluation, boucle d'amelioration, rapport - tourne pour de vrai.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from distill import graph as graph_module
from distill.agents.evaluator import EvaluationResult, evaluate_model
from distill.cli import app
from distill.config import TaskConfig
from distill.data import load_examples
from distill.graph import PipelineOptions, build_context, run_sequential
from distill.report.html_report import ReportData, render_report

from .conftest import FakeTeacher, StubPredictor

runner = CliRunner()


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch, cfg: TaskConfig, tmp_path: Path):
    """Remplace le professeur et l'eleve par des doubles, garde tout le reste."""
    teachers: list[FakeTeacher] = []

    def fake_build_teacher(teacher_cfg, *, cache=None, tracker=None):  # type: ignore[no-untyped-def]
        teacher = FakeTeacher(teacher_cfg, cache=cache, tracker=tracker)
        teachers.append(teacher)
        return teacher

    monkeypatch.setattr(graph_module, "build_teacher", fake_build_teacher)

    mode = {"value": "perfect"}

    def fake_predictors(ctx, state):  # type: ignore[no-untyped-def]
        return [StubPredictor("Distill 1.5B", kind="student", mode=mode["value"])]

    monkeypatch.setattr(graph_module, "_build_predictors", fake_predictors)

    cfg.eval.baseline = False  # la baseline chargerait de vrais poids
    cfg.generation.n_examples = 40
    options = PipelineOptions(
        runs_dir=tmp_path / "runs",
        skip_training=True,
        skip_export=True,
    )
    return cfg, options, mode, teachers


def test_full_run_produces_every_artifact(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, _mode, _ = pipeline
    ctx = build_context(cfg, options)
    state = run_sequential(ctx)

    assert ctx.paths.config_snapshot.is_file()
    assert ctx.paths.raw_examples.is_file()
    assert ctx.paths.train_set.is_file()
    assert ctx.paths.test_set.is_file()
    assert ctx.paths.metrics.is_file()
    assert ctx.paths.report.is_file()
    assert ctx.paths.state_file.is_file()
    assert ctx.paths.events.is_file()

    assert state["target_met"] is True
    assert state["train_size"] > 0
    assert state["test_size"] == 5


def test_the_test_set_never_leaks_into_training(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, _mode, _ = pipeline
    ctx = build_context(cfg, options)
    run_sequential(ctx)

    train = {e.fingerprint for e in load_examples(ctx.paths.train_set)}
    test = {e.fingerprint for e in load_examples(ctx.paths.test_set)}
    assert train.isdisjoint(test), "aucune entree de test ne doit se retrouver a l'entrainement"


def test_the_run_is_fully_accounted_for(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, _mode, _ = pipeline
    ctx = build_context(cfg, options)
    run_sequential(ctx)

    summary = ctx.tracker.summary()
    assert summary["total_cost_usd"] > 0
    assert set(summary["stages"]) >= {"generation", "critique", "evaluation"}
    events = [
        json.loads(line) for line in ctx.paths.events.read_text(encoding="utf-8").splitlines()
    ]
    stages = {e["stage"] for e in events}
    assert {"preparation", "generation", "critique", "evaluation"} <= stages


def test_a_failing_student_triggers_the_improvement_loop(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, mode, _ = pipeline
    mode["value"] = "confused"
    cfg.eval.max_iterations = 2

    ctx = build_context(cfg, options)
    state = run_sequential(ctx)

    assert state["target_met"] is False
    assert state["iteration"] == 1, "une seule iteration d'amelioration avec max_iterations=2"
    assert state["iterations"], "le plan d'amelioration doit etre trace"
    assert state["iterations"][0]["directives"]
    # Le run se termine quand meme : rapport produit, pas de boucle infinie.
    assert ctx.paths.report.is_file()


def test_resume_reuses_the_examples_already_generated(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, _mode, teachers = pipeline
    ctx = build_context(cfg, options)
    run_sequential(ctx)
    first_calls = len(teachers[0].calls)
    produced = len(load_examples(ctx.paths.raw_examples))

    resumed = PipelineOptions(
        runs_dir=options.runs_dir,
        run_id=ctx.paths.run_id,
        resume=True,
        skip_generation=True,
        skip_training=True,
        skip_export=True,
    )
    ctx2 = build_context(cfg, resumed)
    assert ctx2.paths.run_id == ctx.paths.run_id
    run_sequential(ctx2)

    assert len(load_examples(ctx2.paths.raw_examples)) == produced
    assert len(teachers[1].calls) < first_calls, "la generation ne doit pas etre refaite"


def test_budget_ceiling_stops_the_run_cleanly(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Depasser le plafond arrete le run, sans perdre ce qui a deja ete produit."""
    from distill.errors import BudgetExceeded

    cfg, options, _mode, _ = pipeline
    cfg.budget.max_cost_usd = 0.0001
    ctx = build_context(cfg, options)
    with pytest.raises(BudgetExceeded, match="budget depasse"):
        run_sequential(ctx)
    assert ctx.paths.state_file.is_file()
    assert ctx.paths.raw_examples.is_file(), "les exemples deja generes sont conserves"


def test_budget_exceeded_is_caught_by_the_runner(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Vu de la CLI, un depassement n'est pas un plantage mais un arret trace."""
    from distill.graph import run_pipeline

    cfg, options, _mode, _ = pipeline
    cfg.budget.max_cost_usd = 0.0001
    state, _ctx = run_pipeline(cfg, options)
    assert "budget" in str(state.get("stop_reason", "")).lower()


# --------------------------------------------------------------------- rapport


def test_report_contains_the_numbers_and_the_failures(
    cfg: TaskConfig, test_examples, tmp_path: Path
) -> None:
    student = evaluate_model(
        cfg, StubPredictor("Distill 1.5B", kind="student", mode="confused"), test_examples
    )
    teacher = evaluate_model(
        cfg, StubPredictor("Professeur", kind="teacher", latency=0.4), test_examples
    )
    baseline = evaluate_model(
        cfg, StubPredictor("Qwen brut", kind="baseline", mode="broken"), test_examples
    )
    evaluation = EvaluationResult(
        reports=[teacher, baseline, student], primary_field="categorie", target_ratio=0.9
    )

    data = ReportData(
        cfg=cfg,
        evaluation=evaluation,
        run_id="20261003-120000",
        generation={"requested": 40, "kept": 35, "schema_violations": 5},
        critique={
            "exact_duplicates": 2,
            "semantic_duplicates": 3,
            "test_leaks": 1,
            "kept": 30,
            "embedder": "hashing",
        },
        loss_history=[{"step": s, "loss": 2.0 - s * 0.02} for s in range(0, 100, 10)],
        costs={
            "total_cost_usd": 0.42,
            "elapsed_min": 12.5,
            "stages": {
                "generation": {
                    "calls": 4,
                    "cache_hits": 0,
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cost_usd": 0.42,
                }
            },
        },
        warnings=["un avertissement de test"],
        artifacts={"GGUF Q4_K_M": "runs/x/export/model.gguf"},
    )
    output = render_report(data, tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")

    assert "<!doctype html>" in html
    assert "Distill 1.5B" in html and "Professeur" in html and "Qwen brut" in html
    assert "Cible non atteinte" in html
    assert "La ou l'eleve echoue" in html
    assert "confusion:" in html
    assert "un avertissement de test" in html
    assert "polyline" in html, "la courbe de loss doit etre tracee"
    assert "prefers-color-scheme: dark" in html
    assert "Abdelilah OURTI" in html


def test_report_without_a_loss_curve_still_renders(
    cfg: TaskConfig, test_examples, tmp_path: Path
) -> None:
    evaluation = EvaluationResult(
        reports=[evaluate_model(cfg, StubPredictor("eleve"), test_examples)]
    )
    output = render_report(
        ReportData(cfg=cfg, evaluation=evaluation, run_id="x"), tmp_path / "r.html"
    )
    html = output.read_text(encoding="utf-8")
    assert "Courbe d'entrainement" not in html
    assert "Tableau comparatif" in html


# --------------------------------------------------------------------- CLI


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "Abdelilah OURTI" in result.stdout


def test_cli_validate_accepts_the_example(task_dir: Path) -> None:
    result = runner.invoke(app, ["validate", "--task", str(task_dir / "task.yaml")])
    assert result.exit_code == 0
    assert "tout est conforme" in result.stdout


def test_cli_validate_reports_non_conforming_seeds(task_dir: Path) -> None:
    (task_dir / "seeds.jsonl").write_text(
        json.dumps(
            {"input": "x", "output": {"categorie": "inconnue", "urgence": "basse", "resume": "y"}}
        )
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["validate", "--task", str(task_dir / "task.yaml")])
    assert result.exit_code == 1
    assert "non conformes" in result.stdout


def test_cli_estimate_shows_a_price(task_dir: Path) -> None:
    result = runner.invoke(app, ["estimate", "--task", str(task_dir / "task.yaml")])
    assert result.exit_code == 0
    assert "estimation" in result.stdout


def test_cli_dry_run_launches_nothing(task_dir: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "--task",
            str(task_dir / "task.yaml"),
            "--dry-run",
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0
    assert "rien n'a ete lance" in result.stdout
    assert not (tmp_path / "runs").exists()


def test_cli_reports_a_broken_task_file(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("name: [pas un slug]\n", encoding="utf-8")
    result = runner.invoke(app, ["validate", "--task", str(tmp_path / "bad.yaml")])
    assert result.exit_code == 1


# --------------------------------------------------------------------- LangGraph

langgraph = pytest.importorskip("langgraph", reason="LangGraph est optionnel")


def test_the_langgraph_pipeline_runs_end_to_end(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Le graphe produit le meme resultat que l'execution sequentielle."""
    from distill.graph import run_pipeline

    cfg, options, _mode, _ = pipeline
    state, ctx = run_pipeline(cfg, options)

    assert state["target_met"] is True
    assert ctx.paths.report.is_file()
    assert ctx.paths.metrics.is_file()
    assert ctx.paths.checkpoint_db.is_file(), "l'etat du graphe doit etre persiste"


def test_the_graph_loops_back_through_the_critic(pipeline) -> None:  # type: ignore[no-untyped-def]
    """Les exemples cibles repassent par la dedup et le juge, comme les autres."""
    from distill.graph import run_pipeline

    cfg, options, mode, _ = pipeline
    mode["value"] = "confused"
    cfg.eval.max_iterations = 2
    state, ctx = run_pipeline(cfg, options)

    assert state["target_met"] is False
    assert state["iterations"]
    assert ctx.paths.report.is_file()


# --------------------------------------------------------------------- suivi


def test_watch_renders_the_state_of_a_finished_run(pipeline) -> None:  # type: ignore[no-untyped-def]
    from rich.console import Console

    from distill.dashboard import render

    cfg, options, _mode, _ = pipeline
    ctx = build_context(cfg, options)
    run_sequential(ctx)

    buffer = Console(record=True, width=120)
    buffer.print(render(ctx.paths))
    output = buffer.export_text()
    assert "Etat du run" in output
    assert "generation" in output
    assert "evaluation" in output


def test_sparkline_summarises_a_descending_curve() -> None:
    from distill.dashboard import sparkline

    line = sparkline([5.0, 4.0, 3.0, 2.0, 1.0])
    assert len(line) == 5
    assert line[0] == "█" and line[-1] == "▁"
    assert sparkline([1.0]) == ""


def test_cli_watch_once_on_a_finished_run(pipeline) -> None:  # type: ignore[no-untyped-def]
    cfg, options, _mode, _ = pipeline
    ctx = build_context(cfg, options)
    run_sequential(ctx)

    result = runner.invoke(
        app,
        [
            "watch",
            "--task",
            str(cfg.source_path),
            "--runs-dir",
            str(options.runs_dir),
            "--run-id",
            ctx.paths.run_id,
            "--once",
        ],
    )
    assert result.exit_code == 0
    assert "Etat du run" in result.stdout


# --------------------------------------------------------------------- publication


def test_model_card_carries_the_measured_numbers_and_the_limits(
    cfg: TaskConfig, test_examples, tmp_path: Path
) -> None:
    from distill.export.hub import build_model_card

    student = evaluate_model(
        cfg, StubPredictor("Distill 1.5B", kind="student", mode="confused"), test_examples
    )
    teacher = evaluate_model(
        cfg, StubPredictor("Professeur", kind="teacher", latency=0.9), test_examples
    )
    metrics = EvaluationResult(reports=[teacher, student], primary_field="categorie").as_dict()

    card = build_model_card(cfg, metrics)

    assert card.startswith("---"), "la carte doit porter son front matter YAML"
    assert "license: apache-2.0" in card
    assert f"base_model: {cfg.student.model}" in card
    assert "## Resultats mesures" in card
    assert "## Limites" in card
    assert "## Ou le modele echoue" in card
    assert "confusion:" in card, "les cas d'echec reels doivent figurer dans la carte"
    # Le prompt systeme publie doit etre exactement celui de l'entrainement.
    from distill.prompts import system_prompt

    assert system_prompt(cfg) in card
    # Les champs libres sont signales comme non mesures.
    assert "resume" in card


def test_publication_plan_ships_the_right_files(
    cfg: TaskConfig, test_examples, tmp_path: Path
) -> None:
    from distill.export.hub import plan_publication
    from distill.paths import RunPaths, write_json

    paths = RunPaths.create(tmp_path / "runs", cfg.name, "20261003-000000")
    report = evaluate_model(cfg, StubPredictor("Distill", kind="student"), test_examples)
    write_json(paths.metrics, EvaluationResult(reports=[report]).as_dict())
    for name in (f"{cfg.model_slug}-f16.gguf", f"{cfg.model_slug}-Q4_K_M.gguf"):
        (paths.export_dir / name).write_bytes(b"gguf")
    paths.modelfile.write_text("FROM ./x.gguf", encoding="utf-8")

    plan = plan_publication(cfg, paths, "compte/distill-test")
    names = {p.name for p in plan.files}

    assert f"{cfg.model_slug}-Q4_K_M.gguf" in names
    assert f"{cfg.model_slug}-f16.gguf" not in names, (
        "le f16 ne part pas une fois le quantifie disponible"
    )
    assert "Modelfile" in names
    assert plan.card.is_file()
    assert "compte/distill-test" in plan.describe()


def test_publication_refuses_a_run_without_metrics(cfg: TaskConfig, tmp_path: Path) -> None:
    from distill.errors import ExportError
    from distill.export.hub import plan_publication
    from distill.paths import RunPaths

    paths = RunPaths.create(tmp_path / "runs", cfg.name, "20261003-000001")
    with pytest.raises(ExportError, match="metrique"):
        plan_publication(cfg, paths, "compte/x")


def test_cli_publish_dry_run_writes_the_card_without_sending(
    cfg: TaskConfig, test_examples, tmp_path: Path
) -> None:
    from distill.paths import RunPaths, write_json

    paths = RunPaths.create(tmp_path / "runs", cfg.name, "20261003-000002")
    report = evaluate_model(cfg, StubPredictor("Distill", kind="student"), test_examples)
    write_json(paths.metrics, EvaluationResult(reports=[report]).as_dict())
    (paths.export_dir / f"{cfg.model_slug}-Q4_K_M.gguf").write_bytes(b"gguf")

    result = runner.invoke(
        app,
        [
            "publish",
            "--task",
            str(cfg.source_path),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--run-id",
            paths.run_id,
            "--repo",
            "compte/distill-test",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0
    assert "rien n'a ete envoye" in result.stdout
    assert (paths.export_dir / "README.md").is_file()
