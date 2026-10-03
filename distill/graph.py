"""Orchestration : le graphe LangGraph et son etat persiste.

Le pipeline est une suite d'etapes avec une seule boucle :

    task.yaml -> Generateur -> Critique -> Entrainement -> Evaluateur
                     ^                                         |
                     |  cible atteinte ? non -> Ameliorateur --+
                     |                          oui -> Export -> Rapport

Deux principes de conception :

1. **Les donnees vivent sur le disque, pas dans l'etat.** L'etat ne contient que
   des compteurs et des metriques. Un etat leger se serialise vite, se relit a
   l'oeil, et la reprise apres crash consiste simplement a retrouver les
   fichiers deja produits.

2. **LangGraph est optionnel.** Le meme enchainement tourne sans lui, en
   sequentiel. Cela garde les tests rapides et permet d'installer Distill sans
   la dependance quand on veut juste un run d'un bout a l'autre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import yaml

from .agents.critic import critique
from .agents.evaluator import EvaluationResult, ModelReport, evaluate_model
from .agents.generator import generate
from .agents.improver import improve
from .cache import DiskCache
from .config import TaskConfig
from .costs import CostTracker, estimate_generation_cost
from .data import Example, examples_from_inline, load_examples, save_examples
from .errors import BudgetExceeded, DistillError
from .logging_utils import get_logger, log_event, setup_logging, step
from .paths import RunPaths, cache_dir, read_json, write_json
from .schema import build_output_model, primary_label_field, validate_output
from .students import Predictor
from .teachers import BaseTeacher, build_teacher


class PipelineState(TypedDict, total=False):
    """Etat persiste du graphe. Volontairement plat et serialisable."""

    iteration: int
    generated_total: int
    train_size: int
    test_size: int
    target_met: bool
    quality_ratio: float
    stop_reason: str
    generation: dict[str, Any]
    critique: dict[str, Any]
    training: dict[str, Any]
    evaluation: dict[str, Any]
    iterations: list[dict[str, Any]]
    warnings: list[str]
    artifacts: dict[str, str]


@dataclass
class PipelineOptions:
    """Ce que la CLI peut changer sans toucher au task.yaml."""

    runs_dir: Path = Path("runs")
    run_id: str | None = None
    resume: bool = False
    skip_generation: bool = False
    skip_training: bool = False
    skip_export: bool = False
    skip_report: bool = False
    eval_backend: str = "hf"
    dry_run: bool = False
    assume_yes: bool = False
    verbose: bool = False


@dataclass
class PipelineContext:
    """Objets partages par toutes les etapes."""

    cfg: TaskConfig
    paths: RunPaths
    options: PipelineOptions
    tracker: CostTracker
    cache: DiskCache
    teacher: BaseTeacher
    seeds: list[Example] = field(default_factory=list)
    test_set: list[Example] = field(default_factory=list)
    train_set: list[Example] = field(default_factory=list)
    teacher_report: ModelReport | None = None
    baseline_report: ModelReport | None = None
    student_report: ModelReport | None = None
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    evaluation: EvaluationResult | None = None

    def save_state(self, state: PipelineState) -> None:
        write_json(self.paths.state_file, dict(state))


# --------------------------------------------------------------------------
# Etapes
# --------------------------------------------------------------------------


def node_prepare(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """Charge les seeds et le test set, fige la config, estime le cout."""
    cfg = ctx.cfg
    log = get_logger()
    with step("preparation", f"tache {cfg.name}") as info:
        if cfg.seeds is not None:
            ctx.seeds = load_examples(cfg.seeds, input_key=cfg.input_key, output_key=cfg.output_key)
        else:
            ctx.seeds = examples_from_inline(
                cfg.seed_examples, input_key=cfg.input_key, output_key=cfg.output_key
            )
        if len(ctx.seeds) < 10:
            log.warning(
                f"[yellow]{len(ctx.seeds)} seeds seulement[/] : le cahier des charges en recommande 10 a 50"
            )

        # Les seeds doivent eux-memes respecter le schema, sinon tout le reste
        # herite de l'erreur.
        model = build_output_model(cfg.output_schema)
        bad = [i for i, ex in enumerate(ctx.seeds) if not validate_output(model, ex.output)[0]]
        if bad:
            raise DistillError(
                f"{len(bad)} seeds ne respectent pas output_schema (indices {bad[:5]}). "
                "Corrige-les avant de lancer un run."
            )

        if cfg.test_set is not None:
            ctx.test_set = load_examples(
                cfg.test_set, input_key=cfg.input_key, output_key=cfg.output_key
            )
        info["seeds"] = len(ctx.seeds)
        info["test_set"] = len(ctx.test_set)

        ctx.paths.config_snapshot.write_text(
            yaml.safe_dump(cfg.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

        estimate = estimate_generation_cost(
            cfg.generation.n_examples,
            cfg.generation.examples_per_call,
            cfg.teacher.model,
            price_override=(cfg.teacher.price_input_per_mtok, cfg.teacher.price_output_per_mtok),
        )
        log.info(
            f"cout de generation estime : ~{estimate:.2f} USD "
            f"(plafond {cfg.budget.max_cost_usd:.2f} USD, duree max {cfg.budget.max_duration_min} min)"
        )
        info["estimated_usd"] = round(estimate, 3)

    return {**state, "warnings": list(state.get("warnings", []))}


def node_generate(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F2 - genere le corpus synthetique."""
    cfg = ctx.cfg
    existing = _read_jsonl_examples(ctx.paths.raw_examples)
    if ctx.options.skip_generation and existing:
        get_logger().info(f"generation ignoree : {len(existing)} exemples deja presents")
        return {**state, "generated_total": len(existing)}

    with step("generation", f"{cfg.generation.n_examples} exemples cibles") as info:
        result = generate(cfg, ctx.teacher, ctx.seeds)
        all_examples = existing + result.examples
        save_examples(ctx.paths.raw_examples, all_examples)
        info["produits"] = len(result.examples)
        info["total"] = len(all_examples)
        # Plafond atteint pendant la generation : on s'arrete ici plutot que
        # d'entrainer sur un corpus tronque en silence.
        ctx.tracker.check()

    return {
        **state,
        "generated_total": len(all_examples),
        "generation": result.stats.as_dict(),
    }


def node_critique(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F3 - nettoie, juge, isole le test set."""
    cfg = ctx.cfg
    candidates = _read_jsonl_examples(ctx.paths.raw_examples)
    with step("critique", f"{len(candidates)} exemples a trier") as info:
        result = critique(cfg, ctx.teacher, candidates, held_out=ctx.test_set)
        ctx.train_set = result.train
        ctx.test_set = result.test
        save_examples(ctx.paths.train_set, result.train)
        save_examples(ctx.paths.test_set, result.test)
        save_examples(ctx.paths.rejected, result.rejected)
        info["train"] = len(result.train)
        info["test"] = len(result.test)
        info["rejetes"] = len(result.rejected)

    return {
        **state,
        "train_size": len(result.train),
        "test_size": len(result.test),
        "critique": result.stats.as_dict(),
        "warnings": [*state.get("warnings", []), *result.warnings],
    }


def node_train(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F4 - entraine l'adaptateur LoRA."""
    cfg = ctx.cfg
    if ctx.options.skip_training:
        get_logger().info("entrainement ignore (--skip-training)")
        return state

    from .training.lora_trainer import train_lora

    iteration = state.get("iteration", 0)
    output_dir = (
        ctx.paths.adapter_dir if iteration == 0 else ctx.paths.iteration_dir(iteration) / "adapter"
    )
    with step("entrainement", f"{len(ctx.train_set)} exemples, iteration {iteration}") as info:
        result = train_lora(cfg, ctx.train_set, ctx.paths, output_dir=output_dir)
        ctx.loss_history = result.history
        info["loss"] = round(result.final_loss, 4)
        info["steps"] = result.steps
    return {**state, "training": result.as_dict()}


def _build_predictors(ctx: PipelineContext, state: PipelineState) -> list[Predictor]:
    """Construit les modeles a comparer, en reutilisant ce qui ne bouge pas.

    Le professeur et la baseline donnent le meme resultat a chaque iteration :
    les re-evaluer couterait de l'argent pour rien.
    """
    cfg = ctx.cfg
    predictors: list[Predictor] = []
    iteration = state.get("iteration", 0)
    adapter = (
        ctx.paths.adapter_dir if iteration == 0 else ctx.paths.iteration_dir(iteration) / "adapter"
    )

    if ctx.options.eval_backend == "ollama":
        from .students.ollama_runner import OllamaPredictor

        predictors.append(OllamaPredictor(cfg, kind="student"))
    else:
        from .students.hf_runner import HFPredictor

        predictors.append(
            HFPredictor(
                cfg,
                adapter_dir=adapter if adapter.is_dir() else None,
                kind="student",
                name=f"Distill {cfg.student.model.split('/')[-1]}",
            )
        )
    return predictors


def node_evaluate(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F5 - compare eleve, professeur et baseline sur le meme test set."""
    cfg = ctx.cfg
    subset = (
        ctx.test_set[: cfg.eval.max_test_examples] if cfg.eval.max_test_examples else ctx.test_set
    )
    if not subset:
        raise DistillError("test set vide : impossible d'evaluer quoi que ce soit")

    with step("evaluation", f"{len(subset)} exemples de test") as info:
        reports: list[ModelReport] = []

        for predictor in _build_predictors(ctx, state):
            try:
                reports.append(evaluate_model(cfg, predictor, subset))
            finally:
                predictor.close()
        ctx.student_report = next((r for r in reports if r.kind == "student"), None)

        if (
            cfg.eval.baseline
            and ctx.baseline_report is None
            and ctx.options.eval_backend != "ollama"
        ):
            from .students.hf_runner import HFPredictor

            baseline = HFPredictor(
                cfg,
                kind="baseline",
                name=f"{cfg.student.model.split('/')[-1]} brut",
            )
            try:
                ctx.baseline_report = evaluate_model(cfg, baseline, subset)
            finally:
                baseline.close()
        if ctx.baseline_report:
            reports.append(ctx.baseline_report)

        if ctx.teacher_report is None:
            from .students.teacher_predictor import TeacherPredictor

            ctx.teacher_report = evaluate_model(cfg, TeacherPredictor(cfg, ctx.teacher), subset)
        reports.append(ctx.teacher_report)

        evaluation = EvaluationResult(
            reports=sorted(
                reports, key=lambda r: {"teacher": 0, "baseline": 1, "student": 2}[r.kind]
            ),
            primary_field=primary_label_field(cfg.output_schema, cfg.eval.primary_field),
            target_ratio=cfg.eval.target_ratio,
            concurrency=cfg.teacher.concurrency,
        )
        ctx.evaluation = evaluation
        write_json(ctx.paths.metrics, evaluation.as_dict())
        if ctx.student_report:
            save_errors(ctx.paths.errors, ctx.student_report)

        info["ratio"] = f"{evaluation.quality_ratio:.1%}"
        info["cible"] = (
            "atteinte" if evaluation.target_met(cfg.eval.min_json_valid) else "non atteinte"
        )

    return {
        **state,
        "evaluation": evaluation.as_dict(),
        "quality_ratio": evaluation.quality_ratio,
        "target_met": evaluation.target_met(cfg.eval.min_json_valid),
    }


def save_errors(path: Path, report: ModelReport) -> None:
    """Journalise les erreurs de l'eleve, pour le rapport et l'analyse manuelle."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for case in report.errors:
            fh.write(json.dumps(case.as_dict(), ensure_ascii=False) + "\n")


def node_improve(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F6 - genere des exemples cibles sur les erreurs, puis relance."""
    cfg = ctx.cfg
    iteration = state.get("iteration", 0) + 1
    if ctx.student_report is None:
        return {**state, "iteration": iteration, "stop_reason": "aucune evaluation exploitable"}

    with step("amelioration", f"iteration {iteration}/{cfg.eval.max_iterations}") as info:
        result, plan = improve(
            cfg,
            ctx.teacher,
            ctx.seeds,
            ctx.student_report,
            primary_field=primary_label_field(cfg.output_schema, cfg.eval.primary_field),
            iteration=iteration,
        )
        existing = _read_jsonl_examples(ctx.paths.raw_examples)
        save_examples(ctx.paths.raw_examples, existing + result.examples)
        info["ajoutes"] = len(result.examples)

    entry = {
        "iteration": iteration,
        "score": ctx.student_report.score,
        "ratio": state.get("quality_ratio", 0.0),
        "added": len(result.examples),
        "directives": plan.directives,
    }
    return {
        **state,
        "iteration": iteration,
        "iterations": [*state.get("iterations", []), entry],
        "generated_total": len(existing) + len(result.examples),
    }


def node_export(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """F7 - fusion LoRA, GGUF, Modelfile, enregistrement Ollama."""
    cfg = ctx.cfg
    artifacts = dict(state.get("artifacts", {}))
    if ctx.options.skip_export:
        get_logger().info("export ignore (--skip-export)")
        return {**state, "artifacts": artifacts}

    from .export import convert_to_gguf, create_model, write_modelfile
    from .training.merge import merge_lora

    with step("export", f"{cfg.model_slug}") as info:
        merged = merge_lora(cfg, ctx.paths.adapter_dir, ctx.paths.merged_dir)
        artifacts["modele fusionne"] = str(merged)

        gguf = convert_to_gguf(
            merged,
            ctx.paths.export_dir,
            cfg.model_slug,
            quantizations=cfg.export.quantizations,
            llama_cpp_dir=cfg.export.llama_cpp_dir,
        )
        for quant, path in gguf.quantized.items():
            artifacts[f"GGUF {quant}"] = str(path)
        artifacts["GGUF f16"] = str(gguf.f16)

        modelfile = write_modelfile(cfg, gguf.best(), ctx.paths.modelfile)
        artifacts["Modelfile"] = str(modelfile)
        if create_model(cfg.model_slug, modelfile):
            artifacts["modele Ollama"] = cfg.model_slug
        info["quantifications"] = ", ".join(gguf.quantized) or "f16 seulement"

    return {**state, "artifacts": artifacts}


def node_report(ctx: PipelineContext, state: PipelineState) -> PipelineState:
    """Rapport HTML final."""
    if ctx.options.skip_report or ctx.evaluation is None:
        return state

    from .report.html_report import ReportData, render_report

    with step("rapport", "rapport HTML"):
        artifacts = dict(state.get("artifacts", {}))
        artifacts.setdefault("jeu d'entrainement", str(ctx.paths.train_set))
        artifacts.setdefault("jeu de test", str(ctx.paths.test_set))
        artifacts.setdefault("metriques", str(ctx.paths.metrics))
        data = ReportData(
            cfg=ctx.cfg,
            evaluation=ctx.evaluation,
            run_id=ctx.paths.run_id,
            generation=state.get("generation", {}),
            critique=state.get("critique", {}),
            training=state.get("training"),
            loss_history=ctx.loss_history,
            costs=ctx.tracker.summary(),
            iterations=state.get("iterations", []),
            warnings=state.get("warnings", []),
            artifacts=artifacts,
        )
        render_report(data, ctx.paths.report)
    return {**state, "artifacts": {**state.get("artifacts", {}), "rapport": str(ctx.paths.report)}}


def should_continue(ctx: PipelineContext, state: PipelineState) -> str:
    """Arete de decision : exporter, ou relancer une iteration ?"""
    cfg = ctx.cfg
    if state.get("target_met"):
        return "export"
    if state.get("iteration", 0) >= cfg.eval.max_iterations - 1:
        get_logger().warning(
            f"[yellow]cible non atteinte apres {cfg.eval.max_iterations} iterations[/] : "
            "export du meilleur modele obtenu et rapport final"
        )
        return "export"
    if ctx.tracker.remaining_usd <= 0:
        get_logger().warning("[yellow]budget epuise[/] : on s'arrete et on exporte ce qu'on a")
        return "export"
    return "improve"


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _read_jsonl_examples(path: Path) -> list[Example]:
    """Relit les exemples deja produits : c'est la base de la reprise apres crash."""
    if not path.is_file():
        return []
    return load_examples(path, input_key="input", output_key="output", source="generated")


def build_context(cfg: TaskConfig, options: PipelineOptions) -> PipelineContext:
    """Prepare le run : dossiers, cache, compteur de cout, professeur."""
    paths = (
        RunPaths.latest(options.runs_dir, cfg.name)
        if options.resume and options.run_id is None
        else RunPaths.create(options.runs_dir, cfg.name, options.run_id)
    )
    if paths is None:
        paths = RunPaths.create(options.runs_dir, cfg.name, options.run_id)
    paths.ensure()
    # Le journal structure du run doit exister des la premiere etape, meme
    # quand le pipeline est appele depuis du code et non depuis la CLI.
    setup_logging(paths.events, verbose=options.verbose)

    tracker = CostTracker(
        max_cost_usd=cfg.budget.max_cost_usd,
        max_duration_min=cfg.budget.max_duration_min,
    )
    cache = DiskCache(cache_dir(options.runs_dir), enabled=cfg.teacher.use_cache)
    teacher = build_teacher(cfg.teacher, cache=cache, tracker=tracker)
    return PipelineContext(
        cfg=cfg, paths=paths, options=options, tracker=tracker, cache=cache, teacher=teacher
    )


def _initial_state(ctx: PipelineContext) -> PipelineState:
    if ctx.options.resume:
        saved = read_json(ctx.paths.state_file, None)
        if isinstance(saved, dict):
            get_logger().info(
                f"reprise du run {ctx.paths.run_id} a l'iteration {saved.get('iteration', 0)}"
            )
            return dict(saved)  # type: ignore[return-value]
    return {"iteration": 0, "warnings": [], "iterations": [], "artifacts": {}}


def run_sequential(ctx: PipelineContext) -> PipelineState:
    """Execution sans LangGraph : meme enchainement, meme etat sur disque."""
    state = _initial_state(ctx)
    state = node_prepare(ctx, state)
    ctx.save_state(state)

    state = node_generate(ctx, state)
    ctx.save_state(state)

    while True:
        state = node_critique(ctx, state)
        ctx.save_state(state)
        state = node_train(ctx, state)
        ctx.save_state(state)
        state = node_evaluate(ctx, state)
        ctx.save_state(state)

        if should_continue(ctx, state) == "export":
            break
        state = node_improve(ctx, state)
        ctx.save_state(state)

    state = node_export(ctx, state)
    ctx.save_state(state)
    state = node_report(ctx, state)
    ctx.save_state(state)
    return state


def build_graph(ctx: PipelineContext) -> Any:
    """Construit le graphe LangGraph, avec checkpoint SQLite si disponible."""
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PipelineState)
    builder.add_node("preparation", lambda s: node_prepare(ctx, s))
    builder.add_node("generation", lambda s: node_generate(ctx, s))
    builder.add_node("critique", lambda s: node_critique(ctx, s))
    builder.add_node("entrainement", lambda s: node_train(ctx, s))
    builder.add_node("evaluation", lambda s: node_evaluate(ctx, s))
    builder.add_node("amelioration", lambda s: node_improve(ctx, s))
    builder.add_node("export", lambda s: node_export(ctx, s))
    builder.add_node("rapport", lambda s: node_report(ctx, s))

    builder.add_edge(START, "preparation")
    builder.add_edge("preparation", "generation")
    builder.add_edge("generation", "critique")
    builder.add_edge("critique", "entrainement")
    builder.add_edge("entrainement", "evaluation")
    builder.add_conditional_edges(
        "evaluation",
        lambda s: should_continue(ctx, s),
        {"export": "export", "improve": "amelioration"},
    )
    # La boucle repasse par le Critique : les exemples cibles doivent subir la
    # meme dedup et le meme juge que les autres.
    builder.add_edge("amelioration", "critique")
    builder.add_edge("export", "rapport")
    builder.add_edge("rapport", END)

    checkpointer = None
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        # Connexion explicite plutot que le gestionnaire de contexte
        # from_conn_string : le graphe doit survivre a la portee de ce bloc.
        connection = sqlite3.connect(str(ctx.paths.checkpoint_db), check_same_thread=False)
        checkpointer = SqliteSaver(connection)
    except ImportError:
        get_logger().debug(
            "langgraph-checkpoint-sqlite absent : etat persiste dans state.json seulement"
        )
    return builder.compile(checkpointer=checkpointer)


def run_pipeline(
    cfg: TaskConfig, options: PipelineOptions
) -> tuple[PipelineState, PipelineContext]:
    """Lance le pipeline complet et retourne l'etat final."""
    ctx = build_context(cfg, options)
    log = get_logger()
    log.info(f"run [bold]{ctx.paths.run_id}[/] -> {ctx.paths.root}")
    log_event("run", "start", run_id=ctx.paths.run_id, task=cfg.name)

    try:
        from langgraph.graph import StateGraph  # noqa: F401

        has_langgraph = True
    except ImportError:
        has_langgraph = False

    try:
        if has_langgraph:
            graph = build_graph(ctx)
            config = {"configurable": {"thread_id": ctx.paths.run_id}, "recursion_limit": 60}
            state: PipelineState = graph.invoke(_initial_state(ctx), config=config)
            ctx.save_state(state)
        else:
            log.info("langgraph absent : execution sequentielle (pip install 'distill-slm[graph]')")
            state = run_sequential(ctx)
    except BudgetExceeded as exc:
        log.warning(f"[yellow]{exc}[/]")
        state = read_json(ctx.paths.state_file, {})  # type: ignore[assignment]
        state["stop_reason"] = str(exc)
        ctx.save_state(state)
    finally:
        ctx.teacher.close()

    log_event(
        "run",
        "end",
        run_id=ctx.paths.run_id,
        **{k: v for k, v in state.items() if isinstance(v, (int, float, str, bool))},
    )
    return state, ctx
