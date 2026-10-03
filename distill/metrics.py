"""Metriques du benchmark, calculees a la main.

Pas de scikit-learn : quatre formules valent mieux qu'une dependance de 30 Mo,
et surtout le lecteur du repo peut verifier lui-meme comment le F1 macro est
calcule. C'est un projet dont l'argument principal est un tableau de chiffres ;
ces chiffres doivent etre auditables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class ClassMetrics:
    """Precision, rappel et F1 d'une classe."""

    label: str
    support: int = 0
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "support": self.support,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class ClassificationReport:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list[ClassMetrics]
    n: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "n": self.n,
            "per_class": [c.as_dict() for c in self.per_class],
        }


def classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    labels: Sequence[Any] | None = None,
) -> ClassificationReport:
    """F1 par classe, macro et pondere.

    Une prediction manquante (JSON invalide, champ absent) compte comme une
    erreur, jamais comme une ligne a ignorer : c'est ce qui fait la difference
    entre un benchmark honnete et un benchmark flatteur.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true et y_pred doivent avoir la meme longueur")
    n = len(y_true)
    if n == 0:
        return ClassificationReport(0.0, 0.0, 0.0, [], 0)

    known = list(labels) if labels else sorted({str(v) for v in y_true if v is not None})
    stats = {str(label): ClassMetrics(label=str(label)) for label in known}

    correct = 0
    for truth, pred in zip(y_true, y_pred, strict=True):
        t, p = str(truth), (str(pred) if pred is not None else None)
        stats.setdefault(t, ClassMetrics(label=t)).support += 1
        if p is not None and p == t:
            correct += 1
            stats[t].true_positive += 1
            continue
        stats[t].false_negative += 1
        if p is not None:
            # Une prediction hors vocabulaire est une erreur reelle : on la
            # compte, quitte a creer une classe fantome dans le rapport.
            stats.setdefault(p, ClassMetrics(label=p)).false_positive += 1

    per_class = [stats[label] for label in sorted(stats)]
    scored = [c for c in per_class if c.support > 0]
    macro = sum(c.f1 for c in scored) / len(scored) if scored else 0.0
    total_support = sum(c.support for c in scored) or 1
    weighted = sum(c.f1 * c.support for c in scored) / total_support
    return ClassificationReport(
        accuracy=correct / n,
        macro_f1=macro,
        weighted_f1=weighted,
        per_class=per_class,
        n=n,
    )


def percentile(values: Sequence[float], p: float) -> float:
    """Percentile par interpolation lineaire, sans numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(1.0, p))
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ratio(student: float, teacher: float) -> float:
    """Part du score du professeur atteinte par l'eleve.

    Si le professeur a un score nul, le rapport n'a pas de sens : on retourne 0
    plutot qu'un infini qui donnerait une ligne absurde dans le tableau.
    """
    return student / teacher if teacher > 0 else 0.0


def speedup(student_latency: float, teacher_latency: float) -> float:
    return teacher_latency / student_latency if student_latency > 0 else 0.0


def cost_factor(teacher_cost: float, student_cost: float) -> float:
    return teacher_cost / student_cost if student_cost > 0 else 0.0
