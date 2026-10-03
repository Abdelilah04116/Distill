"""F5 - Agent Evaluateur.

Mesure l'eleve, le professeur et la baseline sur exactement le meme test set,
avec exactement le meme prompt, et regroupe les erreurs par type pour que la
boucle d'amelioration sache quoi corriger.

Regle d'honnetete appliquee dans le code : une sortie illisible est une erreur,
pas une ligne a ignorer. Un modele qui repond n'importe quoi une fois sur dix
ne doit pas obtenir 100 % sur les neuf autres.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import TaskConfig
from ..costs import local_cost_per_1k
from ..data import Example
from ..logging_utils import get_logger, log_event
from ..metrics import (
    ClassificationReport,
    classification_report,
    cost_factor,
    mean,
    percentile,
    ratio,
    speedup,
)
from ..schema import (
    build_output_model,
    enum_values,
    extract_json,
    label_fields,
    primary_label_field,
    validate_output,
)
from ..students import Predictor

MAX_STORED_ERRORS = 60


@dataclass
class ErrorCase:
    """Une erreur conservee pour le rapport et pour l'Ameliorateur."""

    input: str
    expected: dict[str, Any]
    predicted_raw: str
    predicted: dict[str, Any] | None
    category: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "expected": self.expected,
            "predicted_raw": self.predicted_raw[:600],
            "predicted": self.predicted,
            "category": self.category,
            "detail": self.detail,
        }


@dataclass
class ModelReport:
    """Une ligne du tableau comparatif."""

    name: str
    kind: str
    billing: str = "gpu"
    n: int = 0
    json_valid_rate: float = 0.0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    exact_match: float = 0.0
    per_class: list[dict[str, Any]] = field(default_factory=list)
    field_accuracy: dict[str, float] = field(default_factory=dict)
    latency_mean_s: float = 0.0
    latency_p50_s: float = 0.0
    latency_p95_s: float = 0.0
    cost_per_1k_usd: float = 0.0
    call_failures: int = 0
    error_groups: dict[str, int] = field(default_factory=dict)
    errors: list[ErrorCase] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Score de reference du tableau : F1 macro, exactitude a defaut."""
        return self.macro_f1 or self.accuracy

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "billing": self.billing,
            "n": self.n,
            "json_valid_rate": round(self.json_valid_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "exact_match": round(self.exact_match, 4),
            "per_class": self.per_class,
            "field_accuracy": {k: round(v, 4) for k, v in self.field_accuracy.items()},
            "latency_mean_s": round(self.latency_mean_s, 4),
            "latency_p50_s": round(self.latency_p50_s, 4),
            "latency_p95_s": round(self.latency_p95_s, 4),
            "cost_per_1k_usd": round(self.cost_per_1k_usd, 6),
            "call_failures": self.call_failures,
            "error_groups": self.error_groups,
            "errors": [e.as_dict() for e in self.errors],
        }


@dataclass
class EvaluationResult:
    """Tout ce que produit l'etape d'evaluation."""

    reports: list[ModelReport] = field(default_factory=list)
    primary_field: str | None = None
    target_ratio: float = 0.9
    concurrency: int = 1

    def by_kind(self, kind: str) -> ModelReport | None:
        return next((r for r in self.reports if r.kind == kind), None)

    @property
    def teacher(self) -> ModelReport | None:
        return self.by_kind("teacher")

    @property
    def student(self) -> ModelReport | None:
        return self.by_kind("student")

    @property
    def baseline(self) -> ModelReport | None:
        return self.by_kind("baseline")

    @property
    def quality_ratio(self) -> float:
        if not self.student or not self.teacher:
            return 0.0
        return ratio(self.student.score, self.teacher.score)

    @property
    def speedup(self) -> float:
        if not self.student or not self.teacher:
            return 0.0
        return speedup(self.student.latency_mean_s, self.teacher.latency_mean_s)

    @property
    def cost_factor(self) -> float:
        if not self.student or not self.teacher:
            return 0.0
        return cost_factor(self.teacher.cost_per_1k_usd, self.student.cost_per_1k_usd)

    def target_met(self, min_json_valid: float) -> bool:
        if not self.student:
            return False
        if self.student.json_valid_rate < min_json_valid:
            return False
        if not self.teacher:
            return self.student.score >= self.target_ratio
        return self.quality_ratio >= self.target_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_field": self.primary_field,
            "target_ratio": self.target_ratio,
            "teacher_concurrency": self.concurrency,
            "quality_ratio": round(self.quality_ratio, 4),
            "speedup": round(self.speedup, 2),
            "cost_factor": round(self.cost_factor, 2),
            "models": [r.as_dict() for r in self.reports],
        }


def _categorise(
    expected: dict[str, Any],
    predicted: dict[str, Any] | None,
    raw: str,
    primary: str | None,
    parse_error: str | None,
    strict_fields: Sequence[str],
) -> tuple[str, str] | None:
    """Type d'erreur, ou None si la prediction est correcte sur les champs notes.

    Seuls les champs a valeurs fermees sont juges ici. Un champ libre - un
    resume, par exemple - ne peut pas etre compare par egalite de chaines :
    le compter comme une erreur ferait passer tous les modeles pour nuls, le
    professeur compris.
    """
    if predicted is None:
        if not raw.strip():
            return "reponse_vide", parse_error or "aucune sortie"
        return ("json_invalide" if parse_error == "json" else "schema_invalide"), parse_error or ""
    if primary and str(predicted.get(primary)) != str(expected.get(primary)):
        return (
            f"confusion:{expected.get(primary)}->{predicted.get(primary)}",
            f"{primary} attendu {expected.get(primary)!r}, obtenu {predicted.get(primary)!r}",
        )
    wrong = [k for k in strict_fields if k != primary and predicted.get(k) != expected.get(k)]
    if wrong:
        return f"champ_errone:{wrong[0]}", "champs incorrects : " + ", ".join(wrong)
    return None


def evaluate_model(
    cfg: TaskConfig,
    predictor: Predictor,
    test_set: Sequence[Example],
) -> ModelReport:
    """Passe un modele sur le test set et calcule toutes ses metriques."""
    log = get_logger()
    output_model = build_output_model(cfg.output_schema)
    primary = primary_label_field(cfg.output_schema, cfg.eval.primary_field)
    fields = label_fields(cfg.output_schema, cfg.eval.label_fields)
    # Champs notes strictement : les enums du schema, ou tous les champs si le
    # schema n'en declare aucun.
    strict = fields or cfg.schema_fields

    if test_set and cfg.eval.latency_warmup:
        predictor.warmup(test_set[0].input, cfg.eval.latency_warmup)

    log.info(f"evaluation de [bold]{predictor.name}[/] sur {len(test_set)} exemples")
    predictions = predictor.predict_many([ex.input for ex in test_set])

    report = ModelReport(
        name=predictor.name, kind=predictor.kind, billing=predictor.billing, n=len(test_set)
    )
    y_true: list[Any] = []
    y_pred: list[Any] = []
    latencies: list[float] = []
    costs: list[float] = []
    valid = 0
    exact = 0
    field_hits: Counter[str] = Counter()
    groups: Counter[str] = Counter()

    for example, prediction in zip(test_set, predictions, strict=True):
        if prediction.error:
            report.call_failures += 1
        latencies.append(prediction.latency_s)
        costs.append(prediction.cost_usd)

        parsed: dict[str, Any] | None = None
        parse_error: str | None = None
        try:
            payload = extract_json(prediction.raw)
        except ValueError:
            parse_error = "json"
        else:
            ok, normalised, err = validate_output(output_model, payload)
            if ok and normalised is not None:
                parsed = normalised
                valid += 1
            else:
                parse_error = err or "schema"

        if primary:
            y_true.append(example.output.get(primary))
            y_pred.append(parsed.get(primary) if parsed else None)
        if parsed is not None:
            if all(parsed.get(k) == example.output.get(k) for k in strict):
                exact += 1
            for key in example.output:
                if parsed.get(key) == example.output[key]:
                    field_hits[key] += 1

        outcome = _categorise(example.output, parsed, prediction.raw, primary, parse_error, strict)
        if outcome is not None:
            category, detail = outcome
            groups[category] += 1
            if len(report.errors) < MAX_STORED_ERRORS:
                report.errors.append(
                    ErrorCase(
                        input=example.input,
                        expected=example.output,
                        predicted_raw=prediction.raw,
                        predicted=parsed,
                        category=category,
                        detail=detail,
                    )
                )

    n = max(1, len(test_set))
    report.json_valid_rate = valid / n
    report.exact_match = exact / n
    report.field_accuracy = {k: field_hits[k] / n for k in cfg.schema_fields}
    report.latency_mean_s = mean(latencies)
    report.latency_p50_s = percentile(latencies, 0.5)
    report.latency_p95_s = percentile(latencies, 0.95)
    report.error_groups = dict(groups.most_common())

    if primary:
        cls: ClassificationReport = classification_report(
            y_true, y_pred, enum_values(cfg.output_schema, primary) or None
        )
        report.accuracy = cls.accuracy
        report.macro_f1 = cls.macro_f1
        report.weighted_f1 = cls.weighted_f1
        report.per_class = [c.as_dict() for c in cls.per_class]
    else:
        report.accuracy = report.exact_match

    if predictor.billing == "api":
        report.cost_per_1k_usd = mean(costs) * 1000
    else:
        report.cost_per_1k_usd = local_cost_per_1k(
            report.latency_mean_s, cfg.eval.local_gpu_cost_per_hour
        )

    log.info(
        f"  {predictor.name} : F1 macro {report.macro_f1:.3f} | exactitude {report.accuracy:.3f} | "
        f"JSON valide {report.json_valid_rate:.1%} | {report.latency_mean_s * 1000:.0f} ms | "
        f"{report.cost_per_1k_usd:.4f} USD / 1k"
    )
    log_event("evaluation", "model", **{k: v for k, v in report.as_dict().items() if k != "errors"})
    return report


def evaluate(
    cfg: TaskConfig,
    predictors: Sequence[Predictor],
    test_set: Sequence[Example],
) -> EvaluationResult:
    """Evalue tous les modeles sur le meme test set, dans l'ordre fourni."""
    subset = list(test_set)
    if cfg.eval.max_test_examples:
        subset = subset[: cfg.eval.max_test_examples]
    result = EvaluationResult(
        primary_field=primary_label_field(cfg.output_schema, cfg.eval.primary_field),
        target_ratio=cfg.eval.target_ratio,
        concurrency=cfg.teacher.concurrency,
    )
    for predictor in predictors:
        result.reports.append(evaluate_model(cfg, predictor, subset))
    return result
