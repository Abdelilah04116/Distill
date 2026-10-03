"""Fixtures communes.

Aucun test ne touche au reseau. Le professeur est remplace par un double
deterministe qui reconnait le type de prompt qu'on lui envoie (generation,
jugement, inference) et repond en consequence. C'est ce qui permet de tester
les agents de bout en bout sans cle API et sans GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from distill.cache import DiskCache
from distill.config import TaskConfig, TeacherConfig
from distill.costs import CostTracker
from distill.data import Example
from distill.students.base import Prediction, Predictor
from distill.teachers.base import BaseTeacher, LLMResponse, Prompt

CATEGORIES = ["reclamation", "facturation", "technique", "commercial", "autre"]
URGENCES = ["basse", "moyenne", "haute"]

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["categorie", "urgence", "resume"],
    "properties": {
        "categorie": {"type": "string", "enum": CATEGORIES},
        "urgence": {"type": "string", "enum": URGENCES},
        "resume": {"type": "string", "description": "Une ligne."},
    },
}

SEED_ROWS = [
    {
        "input": "Ma commande n'est jamais arrivee, cela fait trois semaines.",
        "output": {
            "categorie": "reclamation",
            "urgence": "haute",
            "resume": "Commande non livree depuis trois semaines.",
        },
    },
    {
        "input": "Pouvez-vous m'envoyer la facture de janvier ?",
        "output": {
            "categorie": "facturation",
            "urgence": "moyenne",
            "resume": "Demande de la facture de janvier.",
        },
    },
    {
        "input": "Le site affiche une erreur 500 depuis ce matin.",
        "output": {
            "categorie": "technique",
            "urgence": "haute",
            "resume": "Erreur 500 sur le site depuis le matin.",
        },
    },
    {
        "input": "Quel est le tarif pour dix licences ?",
        "output": {
            "categorie": "commercial",
            "urgence": "basse",
            "resume": "Demande de tarif pour dix licences.",
        },
    },
    {
        "input": "Merci beaucoup pour votre aide.",
        "output": {"categorie": "autre", "urgence": "basse", "resume": "Remerciement."},
    },
]

TEST_ROWS = [
    {
        "input": "Je n'ai toujours pas recu mon colis commande le 2.",
        "output": {
            "categorie": "reclamation",
            "urgence": "haute",
            "resume": "Colis du 2 non recu.",
        },
    },
    {
        "input": "Merci de m'adresser le duplicata de la facture de mars.",
        "output": {
            "categorie": "facturation",
            "urgence": "basse",
            "resume": "Duplicata de la facture de mars.",
        },
    },
    {
        "input": "L'application plante au demarrage depuis la mise a jour.",
        "output": {
            "categorie": "technique",
            "urgence": "haute",
            "resume": "Plantage au demarrage apres mise a jour.",
        },
    },
    {
        "input": "Proposez-vous une remise pour les associations ?",
        "output": {
            "categorie": "commercial",
            "urgence": "basse",
            "resume": "Question sur les remises associations.",
        },
    },
    {
        "input": "Bonne annee a toute l'equipe.",
        "output": {"categorie": "autre", "urgence": "basse", "resume": "Voeux de bonne annee."},
    },
]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    """Un dossier de tache complet : task.yaml + seeds + test set."""
    _write_jsonl(tmp_path / "seeds.jsonl", SEED_ROWS)
    _write_jsonl(tmp_path / "test_set.jsonl", TEST_ROWS)
    payload = {
        "name": "triage-test",
        "description": "Tu tries les emails du service client.",
        "language": "francais",
        "output_schema": OUTPUT_SCHEMA,
        "seeds": "seeds.jsonl",
        "test_set": "test_set.jsonl",
        "teacher": {"model": "claude-opus-5-5", "concurrency": 2},
        "generation": {"n_examples": 60, "examples_per_call": 5},
        "critic": {"judge_batch_size": 3},
        "eval": {"primary_field": "categorie", "label_fields": ["categorie", "urgence"]},
        "budget": {"max_examples": 200, "max_cost_usd": 1.0, "max_duration_min": 10},
    }
    (tmp_path / "task.yaml").write_text(
        yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def cfg(task_dir: Path) -> TaskConfig:
    return TaskConfig.load(task_dir / "task.yaml")


@pytest.fixture
def seeds() -> list[Example]:
    return [Example(input=r["input"], output=r["output"], source="seed") for r in SEED_ROWS]


@pytest.fixture
def test_examples() -> list[Example]:
    return [Example(input=r["input"], output=r["output"], source="test") for r in TEST_ROWS]


def classify(text: str) -> dict[str, Any]:
    """Regle de classification jouet, suffisante et deterministe."""
    lowered = text.lower()
    if "factur" in lowered or "prelevement" in lowered:
        categorie = "facturation"
    elif "erreur" in lowered or "plante" in lowered or "bug" in lowered:
        categorie = "technique"
    elif "tarif" in lowered or "remise" in lowered or "devis" in lowered:
        categorie = "commercial"
    elif "colis" in lowered or "commande" in lowered or "livr" in lowered:
        categorie = "reclamation"
    else:
        categorie = "autre"
    urgence = "haute" if categorie in {"reclamation", "technique"} else "basse"
    return {"categorie": categorie, "urgence": urgence, "resume": text[:40]}


class FakeTeacher(BaseTeacher):
    """Professeur deterministe. Reconnait le prompt a son systeme."""

    provider_id = "fake"

    def __init__(self, cfg: TeacherConfig | None = None, **kwargs: Any) -> None:
        super().__init__(cfg or TeacherConfig(concurrency=2), **kwargs)
        self.calls: list[Prompt] = []
        #: force un lot de reponses invalides, pour tester la robustesse
        self.break_json = False
        #: note attribuee par le juge
        self.judge_score = 5

    def _call(self, prompt: Prompt, max_tokens: int) -> LLMResponse:
        self.calls.append(prompt)
        if self.break_json:
            return LLMResponse(text="desole, je ne peux pas", input_tokens=10, output_tokens=5)
        if "generateur de donnees" in prompt.system:
            return LLMResponse(text=self._generate(prompt), input_tokens=500, output_tokens=400)
        if "relecteur exigeant" in prompt.system:
            return LLMResponse(text=self._judge(prompt), input_tokens=200, output_tokens=80)
        return LLMResponse(
            text=json.dumps(classify(prompt.user), ensure_ascii=False),
            input_tokens=120,
            output_tokens=40,
        )

    def _generate(self, prompt: Prompt) -> str:
        count = 5
        for line in prompt.user.splitlines():
            if line.startswith("Produis exactement"):
                count = int(line.split()[2])
        base = [row["input"] for row in SEED_ROWS]
        rows = []
        for i in range(count):
            text = f"{base[i % len(base)]} (variante {len(self.calls)}-{i})"
            rows.append({"input": text, "output": classify(text)})
        # Une sortie hors schema par lot : le pipeline doit la rejeter sans broncher.
        rows.append(
            {
                "input": "entree invalide",
                "output": {"categorie": "inconnue", "urgence": "basse", "resume": "x"},
            }
        )
        return "```json\n" + json.dumps(rows, ensure_ascii=False) + "\n```"

    def _judge(self, prompt: Prompt) -> str:
        count = sum(1 for line in prompt.user.splitlines() if line.startswith('{"index"'))
        return json.dumps(
            [{"index": i, "score": self.judge_score, "reason": "ok"} for i in range(count)],
            ensure_ascii=False,
        )


@pytest.fixture
def teacher(tmp_path: Path) -> FakeTeacher:
    return FakeTeacher(
        cache=DiskCache(tmp_path / "cache", enabled=False),
        tracker=CostTracker(max_cost_usd=10.0),
    )


class StubPredictor(Predictor):
    """Modele factice dont on controle exactement les reponses."""

    def __init__(
        self, name: str, kind: str = "student", *, mode: str = "perfect", latency: float = 0.01
    ) -> None:
        self.name = name
        self.kind = kind
        self.billing = "gpu" if kind != "teacher" else "api"
        self.mode = mode
        self.latency = latency
        self.seen: list[str] = []

    def predict(self, text: str) -> Prediction:
        self.seen.append(text)
        expected = classify(text)
        if self.mode == "perfect":
            raw = json.dumps(expected, ensure_ascii=False)
        elif self.mode == "broken":
            raw = "je ne sais pas repondre"
        elif self.mode == "off_schema":
            raw = json.dumps(
                {"categorie": "inventee", "urgence": "basse", "resume": "x"}, ensure_ascii=False
            )
        elif self.mode == "confused":
            raw = json.dumps({**expected, "categorie": "autre"}, ensure_ascii=False)
        else:
            raise ValueError(self.mode)
        return Prediction(
            raw=raw, latency_s=self.latency, input_tokens=100, output_tokens=30, cost_usd=0.001
        )
