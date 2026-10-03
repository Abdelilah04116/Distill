"""F1 - Definition de la tache : lecture et validation de task.yaml.

Tout le pipeline est pilote par un seul fichier. Les chemins relatifs sont
resolus par rapport au repertoire du task.yaml, pas au repertoire courant,
pour qu'un run soit reproductible depuis n'importe ou.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class StrictModel(BaseModel):
    """Refuse les cles inconnues : une faute de frappe dans task.yaml doit crier."""

    model_config = ConfigDict(extra="forbid")


class TeacherConfig(StrictModel):
    """Le professeur ne sert qu'a generer et evaluer, jamais en production."""

    provider: Literal["anthropic", "openai_compatible"] = "anthropic"
    model: str = "claude-opus-5-5"
    base_url: str | None = None
    api_key_env: str = "ANTHROPIC_API_KEY"
    effort: Effort = "medium"
    temperature: float | None = None
    max_tokens: int = Field(default=8000, ge=256, le=64000)
    concurrency: int = Field(default=4, ge=1, le=32)
    use_cache: bool = True
    server_side_fallback: bool = True
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None

    @property
    def is_local(self) -> bool:
        """Vrai si aucune donnee ne sort de la machine (mode RGPD strict)."""
        if self.provider != "openai_compatible" or not self.base_url:
            return False
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        return host in {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}


class StudentConfig(StrictModel):
    """L'eleve : 0.5B a 3B, entraine en LoRA sur un seul GPU."""

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = Field(default=2048, ge=256, le=32768)
    load_in_4bit: bool = True
    dtype: Literal["auto", "bfloat16", "float16"] = "auto"


class BudgetConfig(StrictModel):
    """Plafonds durs : depasses, le run s'arrete proprement avec son etat sauve."""

    max_examples: int = Field(default=3000, ge=50)
    max_cost_usd: float = Field(default=5.0, ge=0.0)
    max_duration_min: int = Field(default=120, ge=1)


class GenerationConfig(StrictModel):
    """F2 - Agent Generateur."""

    n_examples: int = Field(default=3000, ge=50)
    examples_per_call: int = Field(default=10, ge=1, le=50)
    seeds_per_prompt: int = Field(default=4, ge=1, le=20)
    personas: list[str] = Field(default_factory=list)
    tones: list[str] = Field(default_factory=lambda: ["neutre", "enerve", "poli", "presse"])
    lengths: list[str] = Field(default_factory=lambda: ["tres court", "moyen", "long"])
    noise_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    edge_case_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class CriticConfig(StrictModel):
    """F3 - Agent Critique."""

    judge_model: str | None = None
    judge_threshold: int = Field(default=4, ge=1, le=5)
    judge_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)
    judge_batch_size: int = Field(default=5, ge=1, le=20)
    dedup_threshold: float = Field(default=0.92, ge=0.5, le=1.0)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    use_qdrant: bool = True
    test_split: float = Field(default=0.15, ge=0.0, le=0.5)
    min_class_ratio: float = Field(default=0.05, ge=0.0, le=1.0)


class TrainingConfig(StrictModel):
    """F4 - Entrainement LoRA / QLoRA."""

    lora_r: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.0, ge=0.0, le=0.5)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    epochs: float = Field(default=2.0, gt=0.0)
    batch_size: int = Field(default=2, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_steps: int | None = None
    save_steps: int = Field(default=50, ge=1)
    use_unsloth: bool = True
    resume_from_checkpoint: bool = True


class EvalConfig(StrictModel):
    """F5 - Agent Evaluateur et criteres de reussite."""

    target_ratio: float = Field(default=0.90, ge=0.0, le=1.5)
    min_json_valid: float = Field(default=0.95, ge=0.0, le=1.0)
    primary_field: str | None = None
    label_fields: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=3, ge=1, le=10)
    baseline: bool = True
    local_gpu_cost_per_hour: float = Field(default=0.60, ge=0.0)
    latency_warmup: int = Field(default=2, ge=0)
    max_test_examples: int | None = None


class ExportConfig(StrictModel):
    """F7 - Export et service."""

    quantizations: list[str] = Field(default_factory=lambda: ["Q4_K_M", "Q8_0"])
    ollama_model_name: str | None = None
    llama_cpp_dir: Path | None = None
    system_prompt: str | None = None
    push_to_hub: bool = False
    hub_repo: str | None = None


class TaskConfig(StrictModel):
    """Le contenu complet de task.yaml."""

    name: str
    description: str
    language: str = "fr"
    seed: int = 42

    output_schema: dict[str, Any]
    input_key: str = "input"
    output_key: str = "output"

    seeds: Path | None = None
    seed_examples: list[dict[str, Any]] = Field(default_factory=list)
    test_set: Path | None = None

    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    critic: CriticConfig = Field(default_factory=CriticConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    source_path: Path | None = Field(default=None, exclude=True)

    @field_validator("name")
    @classmethod
    def _slug(cls, v: str) -> str:
        cleaned = v.strip().lower().replace(" ", "-").replace("_", "-")
        if not cleaned or not all(c.isalnum() or c == "-" for c in cleaned):
            raise ValueError("name doit etre un slug alphanumerique (ex: email-triage)")
        return cleaned

    @field_validator("output_schema")
    @classmethod
    def _schema_is_object(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type") != "object" or not v.get("properties"):
            raise ValueError("output_schema doit etre un objet JSON schema avec des 'properties'")
        return v

    @model_validator(mode="after")
    def _has_seeds(self) -> TaskConfig:
        if self.seeds is None and not self.seed_examples:
            raise ValueError("fournis 'seeds' (chemin jsonl) ou 'seed_examples' (liste inline)")
        if self.generation.n_examples > self.budget.max_examples:
            self.generation.n_examples = self.budget.max_examples
        return self

    @property
    def model_slug(self) -> str:
        """Nom du modele Ollama produit, ex: distill-email-triage."""
        return self.export.ollama_model_name or f"distill-{self.name}"

    @property
    def schema_fields(self) -> list[str]:
        return list(self.output_schema.get("properties", {}).keys())

    def resolve(self, base: Path) -> TaskConfig:
        """Rend les chemins relatifs absolus par rapport au dossier du task.yaml."""
        if self.seeds is not None and not self.seeds.is_absolute():
            self.seeds = (base / self.seeds).resolve()
        if self.test_set is not None and not self.test_set.is_absolute():
            self.test_set = (base / self.test_set).resolve()
        if self.export.llama_cpp_dir is not None and not self.export.llama_cpp_dir.is_absolute():
            self.export.llama_cpp_dir = (base / self.export.llama_cpp_dir).resolve()
        return self

    @classmethod
    def load(cls, path: str | Path) -> TaskConfig:
        """Charge et valide un task.yaml."""
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise ConfigError(f"task.yaml introuvable : {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"YAML invalide dans {p} : {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"{p} doit contenir un mapping YAML a la racine")
        try:
            cfg = cls.model_validate(raw)
        except Exception as exc:
            raise ConfigError(f"task.yaml invalide ({p}) :\n{exc}") from exc
        cfg.source_path = p
        return cfg.resolve(p.parent)
