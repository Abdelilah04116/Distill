"""Interface commune aux modeles compares pendant l'evaluation.

Professeur, eleve fine-tune et eleve brut passent par le meme objet : c'est la
seule facon de garantir qu'ils recoivent exactement les memes entrees et que
les chiffres du tableau final sont comparables.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass
class Prediction:
    """Sortie brute d'un modele sur une entree, avant validation du schema."""

    raw: str
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


class Predictor(ABC):
    """Un modele interrogeable une entree a la fois."""

    #: nom affiche dans le tableau de resultats
    name: str = "modele"
    #: "teacher" | "student" | "baseline", pour le style du rapport
    kind: str = "student"
    #: cout facture au token, ou derive du temps GPU pour un modele local
    billing: str = "api"

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Une inference."""

    def predict_many(
        self,
        texts: Sequence[str],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[Prediction]:
        """Par defaut, sequentiel : c'est ce qui donne une latence honnete."""
        out: list[Prediction] = []
        for i, text in enumerate(texts, start=1):
            out.append(self.predict(text))
            if on_progress:
                on_progress(i)
        return out

    def warmup(self, text: str, rounds: int = 2) -> None:
        """Absorbe le cout du premier appel (chargement CUDA, cache du serveur).

        Sans ca, la latence moyenne du premier modele evalue est systematiquement
        surestimee et la comparaison est faussee.
        """
        for _ in range(max(0, rounds)):
            try:
                self.predict(text)
            except Exception:  # noqa: BLE001 - un echauffement rate n'est pas fatal
                return

    def close(self) -> None:
        return None

    def describe(self) -> dict[str, str]:
        return {"name": self.name, "kind": self.kind, "billing": self.billing}
