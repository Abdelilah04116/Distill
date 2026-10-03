"""Arborescence d'un run.

Un run est un dossier autonome : config figee, datasets, checkpoints, metriques
et rapport. C'est ce qui rend un resultat reproductible et rejouable, et ce qui
permet la reprise apres crash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def new_run_id() -> str:
    """Identifiant trie chronologiquement, ex: 20261003-141205."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass(frozen=True)
class RunPaths:
    """Chemins d'un run, tous sous runs/<tache>/<run_id>/."""

    root: Path
    task_name: str
    run_id: str

    # --- sous-dossiers ---
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def export_dir(self) -> Path:
        return self.root / "export"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # --- fichiers ---
    @property
    def config_snapshot(self) -> Path:
        return self.root / "task.resolved.yaml"

    @property
    def raw_examples(self) -> Path:
        return self.data / "generated.jsonl"

    @property
    def train_set(self) -> Path:
        return self.data / "train.jsonl"

    @property
    def test_set(self) -> Path:
        return self.data / "test.jsonl"

    @property
    def rejected(self) -> Path:
        return self.data / "rejected.jsonl"

    @property
    def adapter_dir(self) -> Path:
        return self.train_dir / "adapter"

    @property
    def merged_dir(self) -> Path:
        return self.export_dir / "merged"

    @property
    def loss_curve(self) -> Path:
        return self.train_dir / "loss.jsonl"

    @property
    def metrics(self) -> Path:
        return self.eval_dir / "metrics.json"

    @property
    def errors(self) -> Path:
        return self.eval_dir / "errors.jsonl"

    @property
    def report(self) -> Path:
        return self.root / "report.html"

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def checkpoint_db(self) -> Path:
        return self.root / "graph.sqlite"

    @property
    def events(self) -> Path:
        return self.logs / "events.jsonl"

    @property
    def modelfile(self) -> Path:
        return self.export_dir / "Modelfile"

    def gguf(self, quant: str) -> Path:
        return self.export_dir / f"{self.task_name}-{quant}.gguf"

    def iteration_dir(self, iteration: int) -> Path:
        return self.root / f"iteration-{iteration}"

    def ensure(self) -> RunPaths:
        """Cree l'arborescence si elle n'existe pas."""
        for folder in (
            self.root,
            self.data,
            self.train_dir,
            self.eval_dir,
            self.export_dir,
            self.logs,
        ):
            folder.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def create(cls, base: Path | str, task_name: str, run_id: str | None = None) -> RunPaths:
        """Nouveau run (ou reprise si run_id est fourni)."""
        base_path = Path(base).expanduser().resolve()
        rid = run_id or new_run_id()
        paths = cls(root=base_path / task_name / rid, task_name=task_name, run_id=rid).ensure()
        (base_path / task_name / "latest.txt").write_text(rid, encoding="utf-8")
        return paths

    @classmethod
    def latest(cls, base: Path | str, task_name: str) -> RunPaths | None:
        """Dernier run connu pour cette tache, utilise par `distill resume`."""
        base_path = Path(base).expanduser().resolve()
        pointer = base_path / task_name / "latest.txt"
        if pointer.is_file():
            rid = pointer.read_text(encoding="utf-8").strip()
            root = base_path / task_name / rid
            if root.is_dir():
                return cls(root=root, task_name=task_name, run_id=rid)
        task_dir = base_path / task_name
        if not task_dir.is_dir():
            return None
        candidates = sorted((d for d in task_dir.iterdir() if d.is_dir()), reverse=True)
        if not candidates:
            return None
        return cls(root=candidates[0], task_name=task_name, run_id=candidates[0].name)


def cache_dir(base: Path | str) -> Path:
    """Cache des reponses du professeur, partage entre tous les runs.

    Partage volontairement : relancer un run apres un crash ou ajuster un
    hyperparametre ne doit pas re-payer la generation.
    """
    path = Path(base).expanduser().resolve() / "_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: object) -> None:
    """Ecriture atomique : un crash ne laisse pas un JSON tronque."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path, default: object = None) -> object:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
