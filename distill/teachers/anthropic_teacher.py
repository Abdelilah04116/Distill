"""Professeur Anthropic via le SDK officiel.

Points d'attention specifiques aux modeles recents, qui expliquent les quelques
branches ci-dessous :
  - les parametres d'echantillonnage (temperature, top_p) sont refuses par les
    modeles Claude 5 : la profondeur se regle avec `effort`, pas la temperature ;
  - un appel peut revenir en HTTP 200 avec stop_reason "refusal" : il faut le
    tester avant de lire le contenu ;
  - le repli serveur (`fallbacks`) evite qu'un refus isole fasse echouer une
    generation de plusieurs milliers d'exemples.
"""

from __future__ import annotations

from typing import Any

from ..config import TeacherConfig
from .base import BaseTeacher, LLMResponse, Prompt

# Modeles qui rejettent temperature / top_p / top_k (erreur 400).
_NO_SAMPLING = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)

# Au-dela de ce seuil, le SDK demande du streaming pour ne pas taper le timeout HTTP.
_STREAM_ABOVE = 16_000

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicTeacher(BaseTeacher):
    """Backend Anthropic (API hebergee)."""

    provider_id = "anthropic"

    def __init__(self, cfg: TeacherConfig, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depend de l'installation
            raise RuntimeError(
                "le paquet 'anthropic' est requis pour provider: anthropic (pip install anthropic)"
            ) from exc
        self._sdk = anthropic
        import os

        api_key = os.environ.get(self.cfg.api_key_env)
        # Sans cle explicite, le SDK retombe sur le profil OAuth local
        # (`ant auth login`) : ne pas forcer api_key=None a la main.
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    @property
    def _supports_sampling(self) -> bool:
        return not self.model.startswith(_NO_SAMPLING)

    def _request_kwargs(self, prompt: Prompt, max_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": prompt.system,
            "messages": [{"role": "user", "content": prompt.user}],
            "output_config": {"effort": self.cfg.effort},
        }
        if self.cfg.temperature is not None and self._supports_sampling:
            kwargs["temperature"] = self.cfg.temperature
        return kwargs

    def _call(self, prompt: Prompt, max_tokens: int) -> LLMResponse:
        kwargs = self._request_kwargs(prompt, max_tokens)
        use_fallback = self.cfg.server_side_fallback
        if use_fallback:
            kwargs["betas"] = [_FALLBACK_BETA]
            kwargs["fallbacks"] = "default"

        namespace = self._client.beta.messages if use_fallback else self._client.messages
        if max_tokens > _STREAM_ABOVE:
            with namespace.stream(**kwargs) as stream:  # type: ignore[call-overload]
                message = stream.get_final_message()
        else:
            message = namespace.create(**kwargs)  # type: ignore[call-overload]

        if getattr(message, "stop_reason", None) == "refusal":
            details = getattr(message, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise RuntimeError(
                f"le professeur a refuse la requete (categorie : {category or 'inconnue'}). "
                "Reformule la description de la tache ou change de professeur."
            )

        # Les blocs de reflexion et d'outils ne portent pas de .text : on ne
        # garde que les blocs de texte, sans supposer leur type exact.
        text = "".join(
            str(getattr(block, "text", ""))
            for block in message.content
            if getattr(block, "type", None) == "text"
        )
        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop_reason=getattr(message, "stop_reason", None),
        )

    def _is_retryable(self, exc: Exception) -> bool:
        sdk = self._sdk
        if isinstance(exc, (sdk.RateLimitError, sdk.APIConnectionError, sdk.APITimeoutError)):
            return True
        if isinstance(exc, sdk.APIStatusError):
            status = int(getattr(exc, "status_code", 0) or 0)
            return status >= 500 or status in (408, 409)
        return False

    def close(self) -> None:
        self._client.close()
