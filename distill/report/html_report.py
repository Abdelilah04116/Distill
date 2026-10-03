"""F7 - Rapport HTML d'un run complet.

Un seul fichier, sans dependance externe : il doit s'ouvrir depuis une cle USB,
s'envoyer par mail et se lire sur telephone. Tout le calcul de mise en forme
(positions des points, largeurs de barres) se fait ici, en Python ; le gabarit
ne fait que placer des valeurs deja pretes.

Le rapport applique la regle d'honnetete du cahier des charges : les erreurs de
l'eleve y sont montrees, pas resumees.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import __version__
from ..agents.evaluator import EvaluationResult
from ..config import TaskConfig
from ..logging_utils import get_logger

TEMPLATE_DIR = Path(__file__).parent

# Couleurs de series, validees pour les deux modes (voir README, section rapport).
# L'identite suit le modele, jamais son classement : la couleur du professeur
# reste la meme qu'il finisse premier ou dernier.
SERIES_COLORS = {
    "student": ("#2a78d6", "#3987e5"),
    "teacher": ("#eb6834", "#d95926"),
    "baseline": ("#1baf7a", "#199e70"),
}

MAX_REPORT_ERRORS = 12
CHART_W = 720
CHART_H = 200
CHART_PAD = 28


@dataclass
class ReportData:
    """Tout ce que le rapport affiche, rassemble par le graphe."""

    cfg: TaskConfig
    evaluation: EvaluationResult
    run_id: str
    generation: dict[str, Any] = field(default_factory=dict)
    critique: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] | None = None
    loss_history: list[dict[str, Any]] = field(default_factory=list)
    costs: dict[str, Any] = field(default_factory=dict)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)


def _loss_chart(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prepare la courbe de loss : une seule serie, donc pas de legende."""
    points = [(float(h["step"]), float(h["loss"])) for h in history if "loss" in h and "step" in h]
    if len(points) < 2:
        return {"has_data": False}

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span_x = (x_max - x_min) or 1.0
    span_y = (y_max - y_min) or 1.0
    # Un peu d'air en haut et en bas : une courbe collee au cadre se lit mal.
    y_lo = y_min - span_y * 0.1
    y_hi = y_max + span_y * 0.1
    span_y = y_hi - y_lo

    def project(x: float, y: float) -> tuple[float, float]:
        px = CHART_PAD + (x - x_min) / span_x * (CHART_W - 2 * CHART_PAD)
        py = CHART_H - CHART_PAD - (y - y_lo) / span_y * (CHART_H - 2 * CHART_PAD)
        return round(px, 2), round(py, 2)

    projected = [project(x, y) for x, y in points]
    return {
        "has_data": True,
        "polyline": " ".join(f"{x},{y}" for x, y in projected),
        "markers": [
            {"x": px, "y": py, "step": int(step), "loss": round(loss, 4)}
            for (px, py), (step, loss) in zip(projected, points, strict=True)
        ][:: max(1, len(points) // 24)],
        "y_top": round(y_hi, 3),
        "y_bottom": round(y_lo, 3),
        "x_first": int(x_min),
        "x_last": int(x_max),
        "first_loss": round(ys[0], 4),
        "last_loss": round(ys[-1], 4),
        "width": CHART_W,
        "height": CHART_H,
        "pad": CHART_PAD,
    }


def _model_rows(evaluation: EvaluationResult) -> list[dict[str, Any]]:
    """Lignes du tableau et barres du graphique, memes donnees, meme ordre."""
    rows = [r.as_dict() for r in evaluation.reports]
    best = max((r["macro_f1"] or r["accuracy"] for r in rows), default=0.0) or 1.0
    for row, report in zip(rows, evaluation.reports, strict=True):
        light, dark = SERIES_COLORS.get(report.kind, SERIES_COLORS["student"])
        row["color_light"] = light
        row["color_dark"] = dark
        row["score"] = report.score
        # Barre proportionnelle au meilleur score, avec un minimum visible
        # pour qu'un modele a 2 % ne disparaisse pas completement.
        row["bar_pct"] = round(max(1.5, report.score / best * 100), 1)
        row["latency_ms"] = round(report.latency_mean_s * 1000)
    return rows


def _student_errors(evaluation: EvaluationResult) -> list[dict[str, Any]]:
    student = evaluation.student
    if student is None:
        return []
    return [e.as_dict() for e in student.errors[:MAX_REPORT_ERRORS]]


def build_context(data: ReportData) -> dict[str, Any]:
    """Assemble le contexte du gabarit."""
    cfg = data.cfg
    evaluation = data.evaluation
    student = evaluation.student
    teacher = evaluation.teacher
    baseline = evaluation.baseline

    return {
        "task": {
            "name": cfg.name,
            "description": cfg.description,
            "language": cfg.language,
            "student_model": cfg.student.model,
            "teacher_model": cfg.teacher.model,
            "teacher_local": cfg.teacher.is_local,
            "ollama_name": cfg.model_slug,
            "seed": cfg.seed,
            "fields": cfg.schema_fields,
        },
        "run": {
            "id": data.run_id,
            "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "distill_version": __version__,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()}",
        },
        "verdict": {
            "target_met": evaluation.target_met(cfg.eval.min_json_valid),
            "target_ratio": cfg.eval.target_ratio,
            "min_json_valid": cfg.eval.min_json_valid,
            "quality_ratio": evaluation.quality_ratio,
            "speedup": evaluation.speedup,
            "cost_factor": evaluation.cost_factor,
            "has_teacher": teacher is not None,
            "baseline_gain": (student.score - baseline.score) if (student and baseline) else None,
        },
        "models": _model_rows(evaluation),
        "primary_field": evaluation.primary_field,
        "per_class": student.per_class if student else [],
        "field_accuracy": student.field_accuracy if student else {},
        "error_groups": student.error_groups if student else {},
        "errors": _student_errors(evaluation),
        "loss": _loss_chart(data.loss_history),
        "generation": data.generation,
        "critique": data.critique,
        "training": data.training,
        "costs": data.costs,
        "iterations": data.iterations,
        "warnings": data.warnings,
        "artifacts": data.artifacts,
        "concurrency": evaluation.concurrency,
    }


def render_report(data: ReportData, output_path: Path | str) -> Path:
    """Ecrit le rapport HTML et retourne son chemin."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: f"{float(v) * 100:.1f} %" if v is not None else "-"
    env.filters["usd"] = lambda v: f"{float(v):.4f} $" if v is not None else "-"
    env.filters["num"] = lambda v, d=2: f"{float(v):.{d}f}" if v is not None else "-"

    template = env.get_template("template.html.j2")
    html = template.render(**build_context(data))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    get_logger().info(f"rapport HTML : {out}")
    return out
