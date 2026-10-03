"""CLI Distill : `distill run`, `eval`, `export`, `report`, `serve`.

La promesse du projet tient dans une ligne de commande :

    distill run --task examples/email-triage/task.yaml

Tout le reste de la CLI sert a reprendre, inspecter ou rejouer une partie d'un
run qui a deja tourne.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .config import TaskConfig
from .costs import estimate_generation_cost
from .errors import DistillError
from .logging_utils import console, die, setup_logging
from .paths import RunPaths, cache_dir

app = typer.Typer(
    name="distill",
    help="Ton propre modele, specialise, local et 10x moins cher.",
    add_completion=False,
    no_args_is_help=True,
)

TaskOption = Annotated[Path, typer.Option("--task", "-t", help="Chemin du task.yaml.")]
RunsOption = Annotated[Path, typer.Option("--runs-dir", help="Dossier des runs.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Logs detailles.")]


def _load(task: Path) -> TaskConfig:
    try:
        return TaskConfig.load(task)
    except DistillError as exc:
        die(str(exc))
        raise  # inatteignable, pour le typage


def _summary_table(cfg: TaskConfig, state: dict) -> Table:
    """Le tableau final affiche a l'ecran : c'est le moment wow numero 2."""
    table = Table(title=f"Distill — {cfg.name}", title_style="bold", header_style="dim")
    table.add_column("Modele")
    table.add_column("Exactitude", justify="right")
    table.add_column("F1 macro", justify="right")
    table.add_column("JSON valide", justify="right")
    table.add_column("Latence moy.", justify="right")
    table.add_column("Cout / 1 000 req.", justify="right")

    for model in (state.get("evaluation") or {}).get("models", []):
        style = "bold" if model["kind"] == "student" else None
        table.add_row(
            model["name"],
            f"{model['accuracy']:.1%}",
            f"{model['macro_f1']:.1%}",
            f"{model['json_valid_rate']:.1%}",
            f"{model['latency_mean_s'] * 1000:.0f} ms",
            f"{model['cost_per_1k_usd']:.4f} $",
            style=style,
        )
    return table


@app.command()
def run(
    task: TaskOption,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Identifiant de run impose.")
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reprend le dernier run de cette tache.")
    ] = False,
    skip_generation: Annotated[
        bool, typer.Option("--skip-generation", help="Reutilise les exemples deja generes.")
    ] = False,
    skip_training: Annotated[
        bool, typer.Option("--skip-training", help="N'entraine pas (evalue le modele brut).")
    ] = False,
    skip_export: Annotated[
        bool, typer.Option("--skip-export", help="Pas de fusion LoRA ni de GGUF.")
    ] = False,
    skip_report: Annotated[
        bool, typer.Option("--skip-report", help="Pas de rapport HTML.")
    ] = False,
    eval_backend: Annotated[
        str, typer.Option("--eval-backend", help="hf (poids locaux) ou ollama.")
    ] = "hf",
    max_examples: Annotated[
        int | None, typer.Option("--max-examples", help="Surcharge generation.n_examples.")
    ] = None,
    student: Annotated[
        str | None, typer.Option("--student", help="Surcharge le modele eleve.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Affiche le plan et l'estimation, sans rien lancer.")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Ne demande pas confirmation du budget.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Lance le pipeline complet : generation, critique, entrainement, evaluation, export."""
    from .graph import PipelineOptions, run_pipeline

    cfg = _load(task)
    if max_examples:
        cfg.generation.n_examples = max_examples
    if student:
        cfg.student.model = student

    estimate = estimate_generation_cost(
        cfg.generation.n_examples,
        cfg.generation.examples_per_call,
        cfg.teacher.model,
        price_override=(cfg.teacher.price_input_per_mtok, cfg.teacher.price_output_per_mtok),
    )
    console.print(
        Panel(
            f"[bold]{cfg.name}[/] — {cfg.description}\n"
            f"eleve    : {cfg.student.model}\n"
            f"professeur: {cfg.teacher.model}"
            f"{' [green](100 % local)[/]' if cfg.teacher.is_local else ''}\n"
            f"exemples : {cfg.generation.n_examples} · test set : "
            f"{cfg.test_set or 'decoupe automatique'}\n"
            f"cout estime : ~{estimate:.2f} $ (plafond {cfg.budget.max_cost_usd:.2f} $, "
            f"{cfg.budget.max_duration_min} min)",
            title="plan du run",
            border_style="cyan",
        )
    )
    if dry_run:
        console.print("[dim]--dry-run : rien n'a ete lance.[/]")
        raise typer.Exit()

    needs_confirmation = (
        not yes and not cfg.teacher.is_local and estimate > cfg.budget.max_cost_usd * 0.5
    )
    if needs_confirmation and not typer.confirm(
        f"Lancer la generation (~{estimate:.2f} $) ?", default=True
    ):
        raise typer.Exit(code=1)

    options = PipelineOptions(
        runs_dir=runs_dir,
        run_id=run_id,
        resume=resume,
        skip_generation=skip_generation,
        skip_training=skip_training,
        skip_export=skip_export,
        skip_report=skip_report,
        eval_backend=eval_backend,
        assume_yes=yes,
        verbose=verbose,
    )
    paths = RunPaths.latest(runs_dir, cfg.name) if resume else None
    setup_logging(paths.events if paths else None, verbose=verbose)

    try:
        state, ctx = run_pipeline(cfg, options)
    except DistillError as exc:
        die(str(exc))
        return

    console.print()
    console.print(_summary_table(cfg, dict(state)))
    verdict = state.get("target_met")
    ratio = state.get("quality_ratio", 0.0)
    console.print(
        Panel(
            f"{'[bold green]Cible atteinte[/]' if verdict else '[bold yellow]Cible non atteinte[/]'} — "
            f"l'eleve atteint {ratio:.1%} du score du professeur (cible {cfg.eval.target_ratio:.0%}).\n"
            f"cout du run : {ctx.tracker.total_usd:.2f} $ · duree : {ctx.tracker.elapsed_min:.0f} min · "
            f"cache : {ctx.cache.hits} hits\n"
            f"rapport : {ctx.paths.report}\n"
            f"essai   : [bold]ollama run {cfg.model_slug}[/]",
            border_style="green" if verdict else "yellow",
        )
    )


@app.command("eval")
def evaluate_cmd(
    task: TaskOption,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    backend: Annotated[str, typer.Option("--backend", help="hf ou ollama.")] = "ollama",
    with_teacher: Annotated[
        bool, typer.Option("--with-teacher/--no-teacher", help="Inclure le professeur.")
    ] = False,
    verbose: VerboseOption = False,
) -> None:
    """Reevalue un modele deja entraine sur le test set d'un run."""
    from .agents.evaluator import EvaluationResult, ModelReport, evaluate_model
    from .data import load_examples
    from .schema import primary_label_field
    from .students import Predictor

    cfg = _load(task)
    paths = (
        RunPaths.create(runs_dir, cfg.name, run_id)
        if run_id
        else RunPaths.latest(runs_dir, cfg.name)
    )
    if paths is None or not paths.test_set.is_file():
        die(f"aucun test set trouve pour la tache {cfg.name} dans {runs_dir}")
        return
    setup_logging(paths.events, verbose=verbose)

    test_set = load_examples(paths.test_set)
    reports: list[ModelReport] = []
    predictor: Predictor
    if backend == "ollama":
        from .students.ollama_runner import OllamaPredictor

        predictor = OllamaPredictor(cfg, kind="student")
        if not predictor.is_available():
            die(
                f"le modele {cfg.model_slug} n'est pas disponible dans Ollama (lance d'abord `distill export`)"
            )
        reports.append(evaluate_model(cfg, predictor, test_set))
    else:
        from .students.hf_runner import HFPredictor

        predictor = HFPredictor(cfg, adapter_dir=paths.adapter_dir, kind="student")
        reports.append(evaluate_model(cfg, predictor, test_set))
    predictor.close()

    if with_teacher:
        from .cache import DiskCache
        from .costs import CostTracker
        from .students.teacher_predictor import TeacherPredictor
        from .teachers import build_teacher

        tracker = CostTracker(max_cost_usd=cfg.budget.max_cost_usd)
        teacher = build_teacher(cfg.teacher, cache=DiskCache(cache_dir(runs_dir)), tracker=tracker)
        reports.append(evaluate_model(cfg, TeacherPredictor(cfg, teacher), test_set))
        teacher.close()

    result = EvaluationResult(
        reports=reports,
        primary_field=primary_label_field(cfg.output_schema, cfg.eval.primary_field),
        target_ratio=cfg.eval.target_ratio,
    )
    console.print(_summary_table(cfg, {"evaluation": result.as_dict()}))


@app.command("export")
def export_cmd(
    task: TaskOption,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    verbose: VerboseOption = False,
) -> None:
    """Fusionne le LoRA, produit les GGUF et enregistre le modele dans Ollama."""
    from .graph import PipelineOptions, build_context, node_export

    cfg = _load(task)
    paths = (
        RunPaths.create(runs_dir, cfg.name, run_id)
        if run_id
        else RunPaths.latest(runs_dir, cfg.name)
    )
    if paths is None or not paths.adapter_dir.is_dir():
        die(f"aucun adaptateur LoRA trouve pour {cfg.name} (lance d'abord `distill run`)")
        return
    setup_logging(paths.events, verbose=verbose)

    options = PipelineOptions(runs_dir=runs_dir, run_id=paths.run_id, resume=True)
    ctx = build_context(cfg, options)
    state = node_export(ctx, {})
    for name, path in state.get("artifacts", {}).items():
        console.print(f"  {name}: [cyan]{path}[/]")


@app.command("report")
def report_cmd(
    task: TaskOption,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
) -> None:
    """Regenere le rapport HTML a partir des metriques d'un run."""
    import json

    from .agents.evaluator import EvaluationResult, ModelReport
    from .report.html_report import ReportData, render_report

    cfg = _load(task)
    paths = (
        RunPaths.create(runs_dir, cfg.name, run_id)
        if run_id
        else RunPaths.latest(runs_dir, cfg.name)
    )
    if paths is None or not paths.metrics.is_file():
        die(f"aucune metrique trouvee pour {cfg.name}")
        return
    setup_logging(paths.events)

    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))
    state = (
        json.loads(paths.state_file.read_text(encoding="utf-8"))
        if paths.state_file.is_file()
        else {}
    )
    evaluation = EvaluationResult(
        reports=[
            ModelReport(
                **{
                    k: v
                    for k, v in model.items()
                    if k in ModelReport.__dataclass_fields__ and k != "errors"
                }
            )
            for model in metrics.get("models", [])
        ],
        primary_field=metrics.get("primary_field"),
        target_ratio=metrics.get("target_ratio", cfg.eval.target_ratio),
    )
    loss_history = []
    if paths.loss_curve.is_file():
        loss_history = [
            json.loads(line)
            for line in paths.loss_curve.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    data = ReportData(
        cfg=cfg,
        evaluation=evaluation,
        run_id=paths.run_id,
        generation=state.get("generation", {}),
        critique=state.get("critique", {}),
        training=state.get("training"),
        loss_history=loss_history,
        costs=state.get("costs", {}),
        iterations=state.get("iterations", []),
        warnings=state.get("warnings", []),
        artifacts=state.get("artifacts", {}),
    )
    console.print(f"rapport : [cyan]{render_report(data, paths.report)}[/]")


@app.command("serve")
def serve_cmd(
    task: TaskOption,
    host: Annotated[str, typer.Option("--host")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port")] = 8000,
    model: Annotated[
        str | None, typer.Option("--model", help="Nom du modele Ollama a servir.")
    ] = None,
) -> None:
    """Lance l'API FastAPI /predict devant le modele Ollama."""
    try:
        import uvicorn
    except ImportError:
        die("uvicorn est requis : pip install 'distill-slm[serve]'")
        return

    import os

    cfg = _load(task)
    os.environ["DISTILL_TASK"] = str(Path(task).resolve())
    if model:
        os.environ["DISTILL_MODEL"] = model
    console.print(f"API sur http://{host}:{port}/docs — modele {model or cfg.model_slug}")
    uvicorn.run("distill.serving.api:app", host=host, port=port, log_level="info")


@app.command("review")
def review_cmd(
    task: TaskOption,
    file: Annotated[
        Path | None,
        typer.Option("--file", help="Fichier a relire (defaut : le test set de la tache)."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="S'arreter apres N exemples relus.")
    ] = None,
    all_examples: Annotated[
        bool, typer.Option("--all", help="Repasser aussi sur les exemples deja relus.")
    ] = False,
    status: Annotated[
        bool, typer.Option("--status", help="Afficher l'avancement sans rien relire.")
    ] = False,
) -> None:
    """Relit le test set a la main : valider, corriger ou supprimer, exemple par exemple."""
    from .review import load_rows, review, stats_for

    cfg = _load(task)
    target = file or cfg.test_set
    if target is None or not Path(target).is_file():
        die("aucun test set a relire : renseigne test_set dans task.yaml ou passe --file")
        return
    target = Path(target)

    if status:
        stats = stats_for(cfg, load_rows(target))
        console.print(f"{target.name}")
        console.print(f"  exemples  : {stats.total}")
        console.print(f"  relus     : {stats.reviewed}")
        console.print(f"  restants  : {stats.remaining}")
        if stats.invalid:
            console.print(f"  [yellow]hors schema : {len(stats.invalid)}[/]")
        return

    stats = review(
        cfg,
        target,
        prompt=typer.prompt,
        confirm=typer.confirm,
        only_pending=not all_examples,
        limit=limit,
    )
    console.print("")
    console.print(
        f"{stats.reviewed}/{stats.total} relus · {stats.corrected} corriges · "
        f"{stats.deleted} supprimes"
    )
    if stats.remaining:
        console.print(
            f"[dim]{stats.remaining} exemples restants : relance la commande pour continuer.[/]"
        )
    else:
        console.print(
            "[bold green]test set entierement relu.[/] Les chiffres publies reposent dessus."
        )


@app.command("publish")
def publish_cmd(
    task: TaskOption,
    repo: Annotated[
        str | None,
        typer.Option("--repo", help="Depot Hugging Face, ex: compte/distill-email-triage."),
    ] = None,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    private: Annotated[bool, typer.Option("--private", help="Depot prive.")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Ecrit la carte de modele sans rien envoyer.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Ne demande pas confirmation.")] = False,
) -> None:
    """Publie le GGUF, le Modelfile et la carte de modele sur Hugging Face."""
    from .export.hub import plan_publication, publish

    cfg = _load(task)
    target = repo or cfg.export.hub_repo
    if not target:
        die("indique le depot avec --repo, ou renseigne export.hub_repo dans task.yaml")
        return

    paths = (
        RunPaths.create(runs_dir, cfg.name, run_id)
        if run_id
        else RunPaths.latest(runs_dir, cfg.name)
    )
    if paths is None:
        die(f"aucun run trouve pour {cfg.name} dans {runs_dir}")
        return
    setup_logging(paths.events)

    try:
        plan = plan_publication(cfg, paths, target, private=private)
    except DistillError as exc:
        die(str(exc))
        return

    console.print(plan.describe())
    console.print(f"carte de modele : [cyan]{plan.card}[/]")
    if dry_run:
        console.print("[dim]--dry-run : rien n'a ete envoye. Relis la carte avant de publier.[/]")
        return

    # Une publication est publique et difficile a defaire : on confirme.
    if not yes and not typer.confirm(f"Publier sur {target} ?", default=False):
        raise typer.Exit(code=1)
    console.print(f"publie : [cyan]{publish(plan)}[/]")


@app.command("watch")
def watch_cmd(
    task: TaskOption,
    runs_dir: RunsOption = Path("runs"),
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    once: Annotated[bool, typer.Option("--once", help="Affiche l'etat une fois et sort.")] = False,
) -> None:
    """Suit un run en cours : etapes, cout, courbe de loss."""
    from .dashboard import watch

    cfg = _load(task)
    paths = (
        RunPaths.create(runs_dir, cfg.name, run_id)
        if run_id
        else RunPaths.latest(runs_dir, cfg.name)
    )
    if paths is None:
        die(f"aucun run trouve pour {cfg.name} dans {runs_dir}")
        return
    watch(paths, once=once)


@app.command("validate")
def validate_cmd(task: TaskOption) -> None:
    """Verifie le task.yaml, le schema et la conformite des seeds."""
    from .data import examples_from_inline, load_examples
    from .schema import build_output_model, validate_output

    cfg = _load(task)
    model = build_output_model(cfg.output_schema)
    seeds = (
        load_examples(cfg.seeds, input_key=cfg.input_key, output_key=cfg.output_key)
        if cfg.seeds
        else examples_from_inline(
            cfg.seed_examples, input_key=cfg.input_key, output_key=cfg.output_key
        )
    )

    problems: list[str] = []
    for i, example in enumerate(seeds):
        ok, _, error = validate_output(model, example.output)
        if not ok:
            problems.append(f"  seed #{i + 1} : {error}")
    if cfg.test_set:
        for i, example in enumerate(
            load_examples(cfg.test_set, input_key=cfg.input_key, output_key=cfg.output_key)
        ):
            ok, _, error = validate_output(model, example.output)
            if not ok:
                problems.append(f"  test #{i + 1} : {error}")

    console.print(f"task.yaml valide : [bold]{cfg.name}[/]")
    console.print(f"  champs de sortie : {', '.join(cfg.schema_fields)}")
    console.print(f"  seeds : {len(seeds)}")
    if cfg.test_set:
        console.print(f"  test set : {cfg.test_set}")
    if problems:
        console.print("[bold red]exemples non conformes au schema :[/]")
        for problem in problems:
            console.print(problem)
        raise typer.Exit(code=1)
    console.print("[bold green]tout est conforme.[/]")


@app.command("estimate")
def estimate_cmd(task: TaskOption) -> None:
    """Affiche le cout estime de la generation avant de depenser quoi que ce soit."""
    cfg = _load(task)
    estimate = estimate_generation_cost(
        cfg.generation.n_examples,
        cfg.generation.examples_per_call,
        cfg.teacher.model,
        price_override=(cfg.teacher.price_input_per_mtok, cfg.teacher.price_output_per_mtok),
    )
    console.print(f"professeur : {cfg.teacher.model}")
    console.print(f"exemples   : {cfg.generation.n_examples}")
    console.print(
        f"estimation : [bold]~{estimate:.2f} $[/] (plafond {cfg.budget.max_cost_usd:.2f} $)"
    )
    if cfg.teacher.is_local:
        console.print(
            "[green]professeur local : aucun cout API, aucune donnee envoyee au cloud.[/]"
        )


@app.command("cache")
def cache_cmd(
    runs_dir: RunsOption = Path("runs"),
    clear: Annotated[
        bool, typer.Option("--clear", help="Vide le cache des reponses du professeur.")
    ] = False,
) -> None:
    """Inspecte ou vide le cache des appels au professeur."""
    from .cache import DiskCache

    cache = DiskCache(cache_dir(runs_dir))
    entries = len(list(cache.directory.rglob("*.json")))
    if clear:
        removed = cache.clear()
        console.print(f"cache vide : {removed} entrees supprimees.")
        return
    console.print(f"{entries} reponses en cache dans {cache.directory}")


@app.command("version")
def version_cmd() -> None:
    """Affiche la version."""
    console.print(f"distill {__version__} — Abdelilah OURTI, AI Engineer")


if __name__ == "__main__":  # pragma: no cover
    app()
