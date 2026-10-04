"""Suivi du cout et du temps, et application des plafonds de task.yaml.

Deux couts coexistent dans Distill et ne se comparent qu'a la fin :
  - le cout API du professeur, facture au token (generation + evaluation) ;
  - le cout d'inference locale de l'eleve, derive du temps GPU occupe.

Le second est ce qui rend l'argument "10x moins cher" verifiable : on mesure la
latence reelle sur la machine de test et on la multiplie par le prix horaire du
GPU declare dans task.yaml.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .errors import BudgetExceeded

# Tarifs API publics en USD par million de tokens (entree, sortie).
# Source : tarification Anthropic, cache du 2026-09-25.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.00, 50.00),
    "claude-mythos-5-1": (10.00, 50.00),
    "claude-fable-5": (10.00, 50.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def price_for(
    model: str, override: tuple[float | None, float | None] = (None, None)
) -> tuple[float, float]:
    """Tarif (entree, sortie) par million de tokens.

    Un professeur auto-heberge via vLLM n'a pas de prix catalogue : il retourne
    (0, 0) et son cout reel est celui du GPU, compte separement.
    """
    in_override, out_override = override
    if in_override is not None and out_override is not None:
        return float(in_override), float(out_override)
    return PRICING_PER_MTOK.get(model, (0.0, 0.0))


def usd(input_tokens: int, output_tokens: int, prices: tuple[float, float]) -> float:
    """Cout en dollars d'un appel."""
    pin, pout = prices
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def local_cost_per_1k(mean_latency_s: float, gpu_cost_per_hour: float) -> float:
    """Cout d'inference locale pour 1 000 requetes, derive du temps GPU occupe."""
    return mean_latency_s * 1000 * gpu_cost_per_hour / 3600


@dataclass
class StageUsage:
    """Consommation cumulee d'une etape du pipeline."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class CostTracker:
    """Compteur partage par tous les agents, sous verrou pour la concurrence."""

    max_cost_usd: float = float("inf")
    max_duration_min: float = float("inf")
    started_at: float = field(default_factory=time.monotonic)
    stages: dict[str, StageUsage] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        stage: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        cached: bool = False,
    ) -> None:
        """Enregistre un appel. Un hit de cache coute zero mais reste compte."""
        with self._lock:
            usage = self.stages.setdefault(stage, StageUsage())
            usage.calls += 1
            if cached:
                usage.cache_hits += 1
                return
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens
            usage.cost_usd += cost_usd

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(s.cost_usd for s in self.stages.values())

    @property
    def total_calls(self) -> int:
        with self._lock:
            return sum(s.calls for s in self.stages.values())

    @property
    def total_cache_hits(self) -> int:
        with self._lock:
            return sum(s.cache_hits for s in self.stages.values())

    @property
    def elapsed_min(self) -> float:
        return (time.monotonic() - self.started_at) / 60

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.max_cost_usd - self.total_usd)

    def check(self) -> None:
        """Leve BudgetExceeded si un plafond est franchi."""
        if self.total_usd > self.max_cost_usd:
            raise BudgetExceeded(
                f"budget depasse : {self.total_usd:.2f} USD > {self.max_cost_usd:.2f} USD "
                "(ajuste budget.max_cost_usd dans task.yaml)"
            )
        if self.elapsed_min > self.max_duration_min:
            raise BudgetExceeded(
                f"duree depassee : {self.elapsed_min:.0f} min > {self.max_duration_min:.0f} min "
                "(ajuste budget.max_duration_min dans task.yaml)"
            )

    def would_exceed(self, projected_usd: float) -> bool:
        """Vrai si un cout projete ferait sauter le plafond."""
        return self.total_usd + projected_usd > self.max_cost_usd

    def summary(self) -> dict[str, object]:
        """Resume serialisable, repris dans le rapport HTML."""
        with self._lock:
            stages = {
                name: {
                    "calls": s.calls,
                    "cache_hits": s.cache_hits,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cost_usd": round(s.cost_usd, 6),
                }
                for name, s in self.stages.items()
            }
            total = sum(s.cost_usd for s in self.stages.values())
        return {
            "total_cost_usd": round(total, 6),
            "elapsed_min": round(self.elapsed_min, 2),
            "stages": stages,
        }


def over_budget_hint(estimate: float, ceiling: float, n_examples: int) -> str | None:
    """Message d'alerte quand l'estimation depasse le plafond, sinon None.

    Le cas est silencieusement destructeur : le run demarre, consomme tout le
    budget, puis s'arrete en pleine generation. Mieux vaut le dire avant.
    """
    if ceiling <= 0 or estimate <= ceiling:
        return None
    tenable = max(50, int(n_examples * ceiling / estimate))
    return (
        f"l'estimation ({estimate:.2f} $) depasse le plafond ({ceiling:.2f} $) : "
        "le run s'arretera en pleine generation. "
        f"Baisse generation.n_examples vers {tenable}, "
        f"monte budget.max_cost_usd au-dessus de {estimate:.2f} $, "
        "ou choisis un professeur moins cher."
    )


def estimate_generation_cost(
    n_examples: int,
    examples_per_call: int,
    model: str,
    *,
    prompt_tokens: int = 1200,
    tokens_per_example: int = 180,
    price_override: tuple[float | None, float | None] = (None, None),
) -> float:
    """Estimation affichee avant le lancement, comme l'exige le cahier des charges.

    Volontairement grossiere mais du bon ordre de grandeur : un prompt fixe par
    appel plus un cout de sortie proportionnel au nombre d'exemples demandes.
    """
    calls = max(1, -(-n_examples // max(1, examples_per_call)))
    prices = price_for(model, price_override)
    return usd(calls * prompt_tokens, n_examples * tokens_per_example, prices)
