"""Le prompt d'inference, defini une seule fois.

Point critique du pipeline : le prompt utilise a l'entrainement, celui de
l'evaluation, celui du Modelfile Ollama et celui de l'API FastAPI doivent etre
rigoureusement identiques. Une virgule de difference et l'eleve perd plusieurs
points sans explication visible.

D'ou ce module unique, importe partout.
"""

from __future__ import annotations

import json
from typing import Any

from .config import TaskConfig
from .schema import schema_hint

SYSTEM_TEMPLATE = """{description}

Tu reponds UNIQUEMENT par un objet JSON valide, sans texte avant ni apres, respectant exactement ce schema :
{schema}"""


def system_prompt(cfg: TaskConfig) -> str:
    """Prompt systeme commun a l'entrainement, l'evaluation et le service."""
    if cfg.export.system_prompt:
        return cfg.export.system_prompt
    return SYSTEM_TEMPLATE.format(
        description=cfg.description.strip(),
        schema=schema_hint(cfg.output_schema),
    )


def user_prompt(text: str) -> str:
    """Prompt utilisateur : l'entree brute, sans habillage."""
    return text.strip()


def chat_messages(cfg: TaskConfig, text: str) -> list[dict[str, str]]:
    """Messages au format chat, pour les tokenizers et les API."""
    return [
        {"role": "system", "content": system_prompt(cfg)},
        {"role": "user", "content": user_prompt(text)},
    ]


def target_text(output: dict[str, Any]) -> str:
    """Sortie attendue, serialisee de maniere stable.

    Cles triees et separateurs compacts : le modele apprend une seule forme de
    JSON, ce qui fait monter le taux de JSON valide.
    """
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def training_example(cfg: TaskConfig, text: str, output: dict[str, Any]) -> dict[str, Any]:
    """Un exemple au format conversationnel attendu par TRL / Unsloth."""
    return {
        "messages": [
            *chat_messages(cfg, text),
            {"role": "assistant", "content": target_text(output)},
        ]
    }
