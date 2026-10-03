"""Contrat commun aux professeurs (API hebergee ou modele local via vLLM).

Toute la logique transverse vit ici - cache, cout, retry, concurrence - pour
que chaque backend n'ait qu'une seule methode a implementer. Changer de
professeur ne doit jamais changer le comportement du pipeline.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ..cache import DiskCache, cache_key
from ..config import TeacherConfig
from ..costs import CostTracker, price_for, usd
from ..errors import BudgetExceeded, TeacherError
from ..logging_utils import get_logger

MAX_ATTEMPTS = 4


@dataclass(frozen=True)
class LLMResponse:
    """Reponse normalisee, identique quel que soit le backend."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    stop_reason: str | None = None
    latency_s: float = 0.0


@dataclass(frozen=True)
class Prompt:
    """Un appel a faire. Les deux champs suffisent pour tous nos usages."""

    system: str
    user: str


class BaseTeacher(ABC):
    """Professeur : genere les donnees et sert de reference a l'evaluation."""

    provider_id: str = "base"

    def __init__(
        self,
        cfg: TeacherConfig,
        *,
        cache: DiskCache | None = None,
        tracker: CostTracker | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = cfg.model
        self.cache = cache or DiskCache(".distill-cache", enabled=False)
        self.tracker = tracker or CostTracker()
        self.prices = price_for(cfg.model, (cfg.price_input_per_mtok, cfg.price_output_per_mtok))
        self.log = get_logger()

    # --- a implementer par chaque backend ---

    @abstractmethod
    def _call(self, prompt: Prompt, max_tokens: int) -> LLMResponse:
        """Un seul appel reseau, sans cache ni retry."""

    def _is_retryable(self, exc: Exception) -> bool:
        """Par defaut on ne retente pas : chaque backend affine."""
        return isinstance(exc, TimeoutError)

    # --- logique commune ---

    def complete(
        self, prompt: Prompt, *, stage: str = "teacher", max_tokens: int | None = None
    ) -> LLMResponse:
        """Un appel, avec cache, plafond de budget, retry et comptabilite."""
        budget = max_tokens or self.cfg.max_tokens
        key = cache_key(
            self.provider_id,
            self.model,
            prompt.system,
            prompt.user,
            budget,
            self.cfg.effort,
            self.cfg.temperature,
        )
        hit = self.cache.get(key)
        if hit is not None:
            self.tracker.record(stage, cached=True)
            return LLMResponse(
                text=str(hit.get("text", "")),
                input_tokens=int(hit.get("input_tokens", 0)),
                output_tokens=int(hit.get("output_tokens", 0)),
                cached=True,
                stop_reason=hit.get("stop_reason"),
            )

        self.tracker.check()
        response = self._call_with_retry(prompt, budget)
        cost = usd(response.input_tokens, response.output_tokens, self.prices)
        self.tracker.record(
            stage,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
        )
        self.cache.set(
            key,
            {
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "stop_reason": response.stop_reason,
                "model": self.model,
            },
        )
        return response

    def _call_with_retry(self, prompt: Prompt, max_tokens: int) -> LLMResponse:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            started = time.monotonic()
            try:
                response = self._call(prompt, max_tokens)
            except Exception as exc:  # re-classifie juste apres via _is_retryable
                last = exc
                if not self._is_retryable(exc) or attempt == MAX_ATTEMPTS:
                    raise TeacherError(f"{self.provider_id}/{self.model} : {exc}") from exc
                # Backoff exponentiel avec gigue : evite de synchroniser les
                # threads sur un 429 et de reprendre tous en meme temps.
                delay = min(30.0, 2.0**attempt) * (0.5 + random.random())
                self.log.warning(
                    f"[yellow]professeur[/] tentative {attempt}/{MAX_ATTEMPTS} echouee "
                    f"({type(exc).__name__}), nouvelle tentative dans {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            return LLMResponse(
                text=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                stop_reason=response.stop_reason,
                latency_s=time.monotonic() - started,
            )
        raise TeacherError(
            f"{self.provider_id}/{self.model} : echec apres {MAX_ATTEMPTS} tentatives ({last})"
        )

    def complete_many(
        self,
        prompts: Sequence[Prompt],
        *,
        stage: str = "teacher",
        max_tokens: int | None = None,
    ) -> list[LLMResponse | None]:
        """Appels paralleles. Un echec isole renvoie None plutot que de tout perdre.

        Un depassement de budget, lui, arrete tout : c'est un plafond, pas un
        incident. Les resultats deja obtenus restent utilisables.
        """
        results: list[LLMResponse | None] = [None] * len(prompts)
        if not prompts:
            return results
        workers = max(1, min(self.cfg.concurrency, len(prompts)))
        stop = False

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="teacher") as pool:
            futures = {
                pool.submit(self.complete, prompt, stage=stage, max_tokens=max_tokens): idx
                for idx, prompt in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                if stop:
                    future.cancel()
                    continue
                try:
                    results[idx] = future.result()
                except BudgetExceeded:
                    stop = True
                    self.log.warning("[yellow]budget atteint[/] arret de la generation en cours")
                    for pending in futures:
                        pending.cancel()
                except TeacherError as exc:
                    self.log.warning(f"[yellow]appel professeur ignore[/] {exc}")
        return results

    def close(self) -> None:
        """Libere les ressources reseau. Surcharge si le backend en tient."""
        return None
