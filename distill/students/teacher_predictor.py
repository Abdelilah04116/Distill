"""Le professeur vu comme un modele a evaluer.

Il recoit le meme prompt d'inference que l'eleve - pas le prompt de generation.
Comparer un professeur avantage par un prompt plus riche a un eleve contraint
donnerait un benchmark flatteur et faux.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..config import TaskConfig
from ..costs import usd
from ..prompts import system_prompt, user_prompt
from ..teachers import BaseTeacher, Prompt
from .base import Prediction, Predictor


class TeacherPredictor(Predictor):
    """Adaptateur professeur -> Predictor."""

    kind = "teacher"

    def __init__(self, cfg: TaskConfig, teacher: BaseTeacher, *, stage: str = "evaluation") -> None:
        self.cfg = cfg
        self.teacher = teacher
        self.stage = stage
        self.name = f"Professeur {teacher.model}"
        self.billing = "api" if not cfg.teacher.is_local else "gpu"
        self._system = system_prompt(cfg)

    def _prompt(self, text: str) -> Prompt:
        return Prompt(system=self._system, user=user_prompt(text))

    def predict(self, text: str) -> Prediction:
        try:
            response = self.teacher.complete(self._prompt(text), stage=self.stage, max_tokens=1024)
        except Exception as exc:  # noqa: BLE001 - un echec compte comme une erreur, pas un crash
            return Prediction(raw="", error=str(exc))
        return Prediction(
            raw=response.text,
            latency_s=response.latency_s,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=usd(response.input_tokens, response.output_tokens, self.teacher.prices),
        )

    def predict_many(
        self,
        texts: Sequence[str],
        *,
        on_progress: Callable[[int], None] | None = None,
    ) -> list[Prediction]:
        """Appels paralleles : 200 appels sequentiels couteraient des minutes.

        La latence rapportee reste celle mesuree par appel ; le rapport precise
        la concurrence utilisee pour que le chiffre soit interpretable.
        """
        responses = self.teacher.complete_many(
            [self._prompt(t) for t in texts], stage=self.stage, max_tokens=1024
        )
        out: list[Prediction] = []
        for i, response in enumerate(responses, start=1):
            if response is None:
                out.append(Prediction(raw="", error="appel professeur echoue"))
            else:
                out.append(
                    Prediction(
                        raw=response.text,
                        latency_s=response.latency_s,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cost_usd=usd(
                            response.input_tokens, response.output_tokens, self.teacher.prices
                        ),
                    )
                )
            if on_progress:
                on_progress(i)
        return out
