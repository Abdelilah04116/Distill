"""Tests du socle : config, schema, donnees, metriques, couts, cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from distill.cache import DiskCache, cache_key
from distill.config import TaskConfig
from distill.costs import (
    BudgetExceeded,
    CostTracker,
    estimate_generation_cost,
    local_cost_per_1k,
    price_for,
    usd,
)
from distill.data import (
    Example,
    class_distribution,
    dedup_exact,
    load_examples,
    remove_overlap,
    save_examples,
    stratified_split,
    underrepresented,
)
from distill.errors import ConfigError
from distill.metrics import classification_report, percentile, ratio, speedup
from distill.prompts import chat_messages, system_prompt, target_text
from distill.schema import (
    build_output_model,
    enum_values,
    extract_json,
    label_fields,
    primary_label_field,
    schema_hint,
    validate_output,
)

# --------------------------------------------------------------------- config


def test_load_resolves_paths_relative_to_the_yaml(cfg: TaskConfig, task_dir: Path) -> None:
    assert cfg.seeds == (task_dir / "seeds.jsonl").resolve()
    assert cfg.test_set == (task_dir / "test_set.jsonl").resolve()
    assert cfg.model_slug == "distill-triage-test"


def test_unknown_key_is_rejected(task_dir: Path) -> None:
    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    raw["teacher"]["temperatur"] = 0.5  # faute de frappe volontaire
    (task_dir / "bad.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalide"):
        TaskConfig.load(task_dir / "bad.yaml")


def test_missing_seeds_is_rejected(task_dir: Path) -> None:
    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    del raw["seeds"]
    (task_dir / "noseed.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError):
        TaskConfig.load(task_dir / "noseed.yaml")


def test_generation_is_clamped_to_the_budget(task_dir: Path) -> None:
    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    raw["generation"]["n_examples"] = 100_000
    raw["budget"]["max_examples"] = 500
    (task_dir / "big.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert TaskConfig.load(task_dir / "big.yaml").generation.n_examples == 500


def test_local_teacher_is_detected(task_dir: Path) -> None:
    raw = yaml.safe_load((task_dir / "task.yaml").read_text(encoding="utf-8"))
    raw["teacher"] = {
        "provider": "openai_compatible",
        "base_url": "http://localhost:8000/v1",
        "model": "qwen",
    }
    (task_dir / "local.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert TaskConfig.load(task_dir / "local.yaml").teacher.is_local is True
    assert TaskConfig.load(task_dir / "task.yaml").teacher.is_local is False


# --------------------------------------------------------------------- schema


def test_model_enforces_enums_and_extra_fields(cfg: TaskConfig) -> None:
    model = build_output_model(cfg.output_schema)
    ok, data, _ = validate_output(
        model, {"categorie": "technique", "urgence": "haute", "resume": "x"}
    )
    assert ok and data is not None and data["categorie"] == "technique"

    ok, _, error = validate_output(
        model, {"categorie": "inexistante", "urgence": "haute", "resume": "x"}
    )
    assert not ok and error is not None and "categorie" in error

    ok, _, _ = validate_output(
        model, {"categorie": "technique", "urgence": "haute", "resume": "x", "extra": 1}
    )
    assert not ok, "additionalProperties: false doit refuser les champs en trop"


def test_optional_and_nested_and_array_fields() -> None:
    schema = {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "auteur": {
                "type": "object",
                "properties": {"nom": {"type": "string"}},
                "required": ["nom"],
            },
        },
    }
    model = build_output_model(schema)
    ok, data, _ = validate_output(model, {"id": 3, "tags": ["a", "b"], "auteur": {"nom": "Ourti"}})
    assert ok and data is not None and data["auteur"]["nom"] == "Ourti"
    ok, data, _ = validate_output(model, {"id": 3})
    assert ok and data is not None and data["tags"] is None


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Voici le resultat :\n{"a": 1}\nJ\'espere que cela convient.',
        '  \n{"a": 1}  ',
        '{"a": 1} et du bavardage apres',
    ],
)
def test_extract_json_survives_le_bavardage_des_llm(raw: str) -> None:
    assert extract_json(raw) == {"a": 1}


def test_extract_json_handles_nested_braces_in_strings() -> None:
    assert extract_json('prefixe {"a": "accolade } dedans", "b": {"c": 1}} suffixe') == {
        "a": "accolade } dedans",
        "b": {"c": 1},
    }


@pytest.mark.parametrize("raw", ["", "   ", "pas de json ici"])
def test_extract_json_refuses_what_is_not_json(raw: str) -> None:
    with pytest.raises(ValueError):
        extract_json(raw)


def test_schema_helpers(cfg: TaskConfig) -> None:
    assert primary_label_field(cfg.output_schema) == "categorie"
    assert primary_label_field(cfg.output_schema, "urgence") == "urgence"
    assert label_fields(cfg.output_schema) == ["categorie", "urgence"]
    assert "reclamation" in enum_values(cfg.output_schema, "categorie")
    assert "categorie" in schema_hint(cfg.output_schema)


# --------------------------------------------------------------------- donnees


def test_jsonl_roundtrip(tmp_path: Path, seeds: list[Example]) -> None:
    path = tmp_path / "out.jsonl"
    assert save_examples(path, seeds) == len(seeds)
    reloaded = load_examples(path)
    assert [e.input for e in reloaded] == [e.input for e in seeds]


def test_malformed_jsonl_names_the_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": "ok", "output": {}}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        load_examples(path)


def test_dedup_exact_ignores_case_and_spacing() -> None:
    examples = [
        Example(input="Bonjour le monde", output={"a": 1}),
        Example(input="  bonjour   LE monde ", output={"a": 1}),
        Example(input="autre chose", output={"a": 2}),
    ]
    kept, removed = dedup_exact(examples)
    assert len(kept) == 2 and removed == 1


def test_remove_overlap_protects_the_test_set() -> None:
    test = [Example(input="Le site est en panne", output={"a": 1})]
    candidates = [
        Example(input="le site est en panne", output={"a": 1}),
        Example(input="tout va bien", output={"a": 2}),
    ]
    kept, removed = remove_overlap(candidates, test)
    assert removed == 1 and [e.input for e in kept] == ["tout va bien"]


def test_stratified_split_keeps_every_class_in_train_and_test() -> None:
    examples = [
        Example(input=f"texte {i}", output={"categorie": "a" if i % 2 else "b"}) for i in range(40)
    ]
    train, test = stratified_split(examples, test_ratio=0.25, field="categorie", seed=1)
    assert len(test) == 10 and len(train) == 30
    assert set(class_distribution(test, "categorie")) == {"a", "b"}
    assert set(class_distribution(train, "categorie")) == {"a", "b"}


def test_stratified_split_is_reproducible() -> None:
    examples = [Example(input=f"t{i}", output={"c": str(i % 3)}) for i in range(30)]
    a = stratified_split(examples, test_ratio=0.2, field="c", seed=7)[1]
    b = stratified_split(examples, test_ratio=0.2, field="c", seed=7)[1]
    assert [e.input for e in a] == [e.input for e in b]


def test_underrepresented_flags_the_rare_classes() -> None:
    examples = [Example(input=f"t{i}", output={"c": "a"}) for i in range(19)]
    examples.append(Example(input="rare", output={"c": "b"}))
    flagged = underrepresented(examples, "c", ["a", "b", "c"], min_ratio=0.1)
    assert flagged == ["b", "c"]


# --------------------------------------------------------------------- metriques


def test_classification_report_counts_missing_predictions_as_errors() -> None:
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", None, "b", "a"]
    report = classification_report(y_true, y_pred, ["a", "b"])
    assert report.n == 4
    assert report.accuracy == 0.5
    per_class = {c.label: c for c in report.per_class}
    assert per_class["a"].support == 2
    assert per_class["a"].recall == 0.5
    assert 0.0 < report.macro_f1 < 1.0


def test_classification_report_is_perfect_when_everything_matches() -> None:
    report = classification_report(["a", "b"], ["a", "b"], ["a", "b"])
    assert report.accuracy == 1.0 and report.macro_f1 == 1.0


def test_out_of_vocabulary_prediction_is_counted() -> None:
    report = classification_report(["a"], ["zzz"], ["a"])
    assert report.accuracy == 0.0
    assert any(c.label == "zzz" and c.false_positive == 1 for c in report.per_class)


def test_percentile_and_ratios() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert percentile([], 0.9) == 0.0
    assert ratio(0.8, 0.9) == pytest.approx(0.888, abs=1e-3)
    assert ratio(0.8, 0.0) == 0.0
    assert speedup(0.1, 1.0) == pytest.approx(10.0)
    assert speedup(0.0, 1.0) == 0.0


# --------------------------------------------------------------------- couts


def test_known_models_have_a_price() -> None:
    assert price_for("claude-opus-5-5") == (4.0, 20.0)
    assert price_for("claude-haiku-4-5") == (1.0, 5.0)
    # Un modele auto-heberge n'a pas de prix catalogue.
    assert price_for("Qwen/Qwen2.5-32B-Instruct") == (0.0, 0.0)
    assert price_for("peu-importe", (2.0, 4.0)) == (2.0, 4.0)


def test_cost_formula() -> None:
    assert usd(1_000_000, 0, (3.0, 15.0)) == pytest.approx(3.0)
    assert usd(0, 500_000, (3.0, 15.0)) == pytest.approx(7.5)


def test_tracker_raises_when_the_ceiling_is_crossed() -> None:
    tracker = CostTracker(max_cost_usd=0.01)
    tracker.record("generation", input_tokens=10, output_tokens=5, cost_usd=0.009)
    tracker.check()
    tracker.record("generation", cost_usd=0.009)
    with pytest.raises(BudgetExceeded, match="budget depasse"):
        tracker.check()


def test_tracker_counts_cache_hits_without_charging_them() -> None:
    tracker = CostTracker()
    tracker.record("generation", cached=True)
    tracker.record("generation", input_tokens=100, output_tokens=10, cost_usd=0.5)
    assert tracker.total_calls == 2
    assert tracker.total_cache_hits == 1
    assert tracker.total_usd == pytest.approx(0.5)
    assert tracker.summary()["stages"]["generation"]["cache_hits"] == 1


def test_local_cost_is_derived_from_gpu_time() -> None:
    # 100 ms par requete, GPU a 3,60 $/h = 0,001 $/s -> 0,10 $ pour 1000 requetes
    assert local_cost_per_1k(0.1, 3.6) == pytest.approx(0.1)


def test_estimate_is_in_the_right_order_of_magnitude() -> None:
    estimate = estimate_generation_cost(3000, 10, "claude-opus-5-5")
    assert 0.5 < estimate < 20


# --------------------------------------------------------------------- cache


def test_cache_roundtrip_and_stats(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "c")
    key = cache_key("modele", "systeme", "utilisateur")
    assert cache.get(key) is None
    cache.set(key, {"text": "bonjour"})
    assert cache.get(key) == {"text": "bonjour"}
    assert cache.hits == 1 and cache.misses == 1
    assert cache.hit_rate == 0.5
    assert cache.clear() == 1
    assert cache.get(key) is None


def test_cache_key_is_order_independent_but_content_sensitive() -> None:
    assert cache_key("a", {"x": 1, "y": 2}) == cache_key("a", {"y": 2, "x": 1})
    assert cache_key("a", "b") != cache_key("a", "c")


def test_disabled_cache_never_stores(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "off", enabled=False)
    cache.set("k", {"text": "x"})
    assert cache.get("k") is None


# --------------------------------------------------------------------- prompts


def test_the_inference_prompt_is_identical_everywhere(cfg: TaskConfig) -> None:
    """Le prompt d'entrainement, d'evaluation et de service doit etre le meme."""
    messages = chat_messages(cfg, "un email")
    assert messages[0]["content"] == system_prompt(cfg)
    assert messages[1]["content"] == "un email"
    assert cfg.description.strip().split("\n")[0] in system_prompt(cfg)


def test_target_text_is_stable(cfg: TaskConfig) -> None:
    a = target_text({"urgence": "haute", "categorie": "technique"})
    b = target_text({"categorie": "technique", "urgence": "haute"})
    assert a == b, "les cles doivent etre triees pour que le modele voie une seule forme"
    assert json.loads(a)["categorie"] == "technique"
