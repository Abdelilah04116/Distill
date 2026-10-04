"""Tests du parcours « je clone et j'essaie ».

Chacun de ces tests correspond a un defaut rencontre en rejouant pour de vrai
l'installation depuis un clone neuf. Ils sont ici pour que ces defauts ne
reviennent pas : un message d'erreur trompeur coute plus cher a l'adoption
qu'un bug fonctionnel, parce que personne ne le signale.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from rich.console import Console
from typer.testing import CliRunner

from distill.cli import app
from distill.config import TaskConfig, TeacherConfig
from distill.costs import estimate_generation_cost, over_budget_hint
from distill.errors import ConfigError, DependencyError, DistillError

runner = CliRunner()
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "email-triage"


# --------------------------------------------------- coherence des exemples livres


@pytest.mark.parametrize("task_file", ["task.yaml", "task.local.yaml"])
def test_shipped_examples_are_valid(task_file: str) -> None:
    cfg = TaskConfig.load(EXAMPLES / task_file)
    assert cfg.seeds is not None and cfg.seeds.is_file()
    assert cfg.test_set is not None and cfg.test_set.is_file()


@pytest.mark.parametrize("task_file", ["task.yaml", "task.local.yaml"])
def test_shipped_examples_fit_inside_their_own_ceiling(task_file: str) -> None:
    """Un exemple dont l'estimation depasse le plafond s'arrete en pleine
    generation. C'est la premiere chose que voit quelqu'un qui essaie."""
    cfg = TaskConfig.load(EXAMPLES / task_file)
    estimate = estimate_generation_cost(
        cfg.generation.n_examples,
        cfg.generation.examples_per_call,
        cfg.teacher.model,
        price_override=(cfg.teacher.price_input_per_mtok, cfg.teacher.price_output_per_mtok),
    )
    assert estimate <= cfg.budget.max_cost_usd, (
        f"{task_file} : estimation {estimate:.2f} $ > plafond {cfg.budget.max_cost_usd:.2f} $"
    )


def test_the_local_example_costs_nothing_and_stays_offline() -> None:
    cfg = TaskConfig.load(EXAMPLES / "task.local.yaml")
    assert cfg.teacher.is_local
    assert cfg.budget.max_cost_usd == 0.0


# --------------------------------------------------- alerte de depassement


def test_over_budget_hint_stays_quiet_when_everything_fits() -> None:
    assert over_budget_hint(2.0, 5.0, 3000) is None
    assert over_budget_hint(5.0, 5.0, 3000) is None


def test_over_budget_hint_proposes_a_tenable_volume() -> None:
    hint = over_budget_hint(12.0, 5.0, 3000)
    assert hint is not None
    assert "depasse le plafond" in hint
    assert "1250" in hint, "le conseil doit donner un nombre d'exemples qui tient"


def test_cli_estimate_warns_when_the_ceiling_is_too_low(task_dir: Path) -> None:
    import yaml

    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    raw["generation"]["n_examples"] = 3000
    raw["budget"] = {"max_examples": 5000, "max_cost_usd": 0.5, "max_duration_min": 10}
    (task_dir / "cher.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    result = runner.invoke(app, ["estimate", "--task", str(task_dir / "cher.yaml")])
    assert result.exit_code == 0
    assert "Attention" in result.stdout
    assert "depasse le plafond" in result.stdout


# --------------------------------------------------- messages lisibles


def test_install_hints_keep_their_extras() -> None:
    """Rich interprete les crochets : sans echappement, 'distill-slm[train]'
    s'affiche 'distill-slm' et le conseil devient faux."""
    import io

    from rich.markup import escape

    buffer = Console(record=True, width=120, file=io.StringIO())
    buffer.print("[bold red]Erreur[/] " + escape("pip install 'distill-slm[train]'"))
    assert "distill-slm[train]" in buffer.export_text()


def test_die_reports_once_and_preserves_brackets(capsys: pytest.CaptureFixture[str]) -> None:
    from distill.logging_utils import die

    with pytest.raises(SystemExit):
        die("uvicorn est requis : pip install 'distill-slm[serve]'")
    out = capsys.readouterr().out
    assert "distill-slm[serve]" in out
    assert out.count("Erreur") == 1, "le message ne doit pas etre affiche deux fois"


# --------------------------------------------------- echecs rapides et explicites


def test_missing_credentials_fail_before_any_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sans cle, le run echouait trois etapes plus loin en accusant le Critique."""
    pytest.importorskip("anthropic")
    from distill.teachers.anthropic_teacher import AnthropicTeacher

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(AnthropicTeacher, "_has_local_profile", staticmethod(lambda: False))

    with pytest.raises(ConfigError) as exc:
        AnthropicTeacher(TeacherConfig())
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "professeur local" in message, "l'erreur doit proposer l'issue de secours gratuite"


def test_an_unreachable_local_teacher_fails_in_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    from distill.teachers.openai_compatible import OpenAICompatibleTeacher

    def refuse(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("connexion refusee")

    monkeypatch.setattr(httpx.Client, "get", refuse)
    cfg = TeacherConfig(
        provider="openai_compatible", base_url="http://localhost:11434/v1", model="qwen2.5:7b"
    )

    with pytest.raises(ConfigError) as exc:
        OpenAICompatibleTeacher(cfg)
    assert "aucun serveur ne repond" in str(exc.value)
    assert "ollama serve" in str(exc.value)


def test_a_server_without_models_endpoint_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tous les serveurs compatibles OpenAI n'exposent pas /models."""
    from distill.teachers.openai_compatible import OpenAICompatibleTeacher

    def http_error(*args: object, **kwargs: object) -> None:
        raise httpx.HTTPStatusError("404", request=None, response=None)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx.Client, "get", http_error)
    cfg = TeacherConfig(
        provider="openai_compatible", base_url="http://localhost:8000/v1", model="qwen"
    )
    assert OpenAICompatibleTeacher(cfg).model == "qwen"


def test_missing_training_dependencies_are_a_distill_error() -> None:
    """Un extra absent ne doit pas sortir en trace Python brute."""
    assert issubclass(DependencyError, DistillError)
