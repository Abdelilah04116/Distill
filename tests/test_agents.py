"""Tests des agents : generateur, critique, deduplication, evaluateur, ameliorateur."""

from __future__ import annotations

import random

import pytest

from distill.agents.critic import critique, judge
from distill.agents.evaluator import evaluate, evaluate_model
from distill.agents.generator import build_prompts, generate
from distill.agents.improver import build_plan, improve
from distill.config import TaskConfig
from distill.data import Example
from distill.dedup import HashingEmbedder, deduplicate
from distill.schema import build_output_model, validate_output

from .conftest import FakeTeacher, StubPredictor

# --------------------------------------------------------------------- generateur


def test_prompts_cover_the_requested_volume(cfg: TaskConfig, seeds: list[Example]) -> None:
    prompts = build_prompts(cfg, seeds, n_examples=50, rng=random.Random(0))
    assert len(prompts) == 10  # 50 exemples / 5 par appel
    prompt, persona = prompts[0]
    assert "generateur de donnees" in prompt.system
    assert "Produis exactement 5" in prompt.user
    assert persona


def test_prompts_vary_the_personas(cfg: TaskConfig, seeds: list[Example]) -> None:
    personas = {
        persona for _, persona in build_prompts(cfg, seeds, n_examples=200, rng=random.Random(3))
    }
    assert len(personas) > 1, "le persona sampling doit produire de la variete"


def test_targeted_directives_reach_the_prompt(cfg: TaskConfig, seeds: list[Example]) -> None:
    prompts = build_prompts(
        cfg, seeds, n_examples=5, focus=["confusion entre A et B"], rng=random.Random(0)
    )
    assert (
        "CIBLE PRIORITAIRE : confusion entre A et B" in prompts[0][1]
        or "CIBLE PRIORITAIRE" in prompts[0][0].user
    )


def test_generate_keeps_only_schema_valid_examples(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    result = generate(cfg, teacher, seeds, n_examples=20)
    model = build_output_model(cfg.output_schema)
    assert result.examples
    assert all(validate_output(model, e.output)[0] for e in result.examples)
    # Le faux professeur glisse une sortie hors schema par lot : elle doit etre comptee.
    assert result.stats.schema_violations == result.stats.calls
    assert result.stats.kept == len(result.examples)


def test_generate_survives_an_unusable_teacher(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    teacher.break_json = True
    result = generate(cfg, teacher, seeds, n_examples=20)
    assert result.examples == []
    assert result.stats.invalid_json == result.stats.calls


def test_generation_is_accounted_for(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    generate(cfg, teacher, seeds, n_examples=20)
    assert teacher.tracker.total_usd > 0
    assert teacher.tracker.summary()["stages"]["generation"]["calls"] == 4


# --------------------------------------------------------------------- dedup


def test_semantic_dedup_removes_near_duplicates() -> None:
    examples = [
        Example(input="Le site est en panne depuis ce matin", output={"c": "a"}),
        Example(input="Le site est en panne depuis ce matin.", output={"c": "a"}),
        Example(input="Je voudrais un devis pour dix licences", output={"c": "b"}),
    ]
    result = deduplicate(examples, threshold=0.9, embedder=HashingEmbedder(), use_qdrant=False)
    assert len(result.kept) == 2
    assert len(result.removed) == 1


def test_semantic_dedup_blocks_test_set_leaks() -> None:
    protected = [Example(input="Ma facture de mars est fausse", output={"c": "a"})]
    candidates = [
        Example(input="Ma facture de mars est fausse !!", output={"c": "a"}),
        Example(input="Comment reinitialiser mon mot de passe", output={"c": "b"}),
    ]
    result = deduplicate(
        candidates, threshold=0.9, embedder=HashingEmbedder(), protected=protected, use_qdrant=False
    )
    assert len(result.leaked) == 1
    assert [e.input for e in result.kept] == ["Comment reinitialiser mon mot de passe"]


def test_dedup_on_empty_input_is_a_noop() -> None:
    result = deduplicate([], embedder=HashingEmbedder(), use_qdrant=False)
    assert result.kept == [] and result.stats["kept"] == 0


# --------------------------------------------------------------------- critique


def test_judge_rejects_under_the_threshold(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    teacher.judge_score = 2
    accepted, rejected, mean = judge(cfg, teacher, seeds)
    assert accepted == [] and len(rejected) == len(seeds)
    assert mean == pytest.approx(2.0)
    assert all(e.judge_score == 2 for e in rejected)


def test_judge_keeps_examples_it_could_not_score(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    """Une panne du juge ne doit pas detruire des donnees valides."""
    teacher.break_json = True
    accepted, rejected, _ = judge(cfg, teacher, seeds)
    assert len(accepted) == len(seeds) and rejected == []


def test_critique_produces_clean_train_and_keeps_the_manual_test_set(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example], test_examples: list[Example]
) -> None:
    generated = generate(cfg, teacher, seeds, n_examples=40).examples
    duplicated = generated + generated[:5] + [test_examples[0]]
    result = critique(cfg, teacher, duplicated, held_out=test_examples)

    assert result.stats.exact_duplicates >= 5
    assert result.stats.test_leaks >= 1
    assert result.test == test_examples, "un test set ecrit a la main ne doit jamais etre remplace"
    assert all(e.input != test_examples[0].input for e in result.train)
    assert result.stats.kept == len(result.train)


def test_critique_warns_when_it_has_to_invent_a_test_set(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example]
) -> None:
    generated = generate(cfg, teacher, seeds, n_examples=40).examples
    result = critique(cfg, teacher, generated, held_out=[])
    assert result.test, "un test set doit etre decoupe faute de mieux"
    assert any("ecrit a la main" in w for w in result.warnings)


def test_critique_flags_class_imbalance(cfg: TaskConfig, teacher: FakeTeacher) -> None:
    skewed = [
        Example(
            input=f"Ma facture numero {i} est fausse",
            output={"categorie": "facturation", "urgence": "basse", "resume": "x"},
        )
        for i in range(30)
    ]
    result = critique(cfg, teacher, skewed, held_out=[])
    assert result.stats.underrepresented
    assert any("sous-representees" in w for w in result.warnings)


# --------------------------------------------------------------------- evaluateur


def test_a_perfect_model_scores_one(cfg: TaskConfig, test_examples: list[Example]) -> None:
    report = evaluate_model(cfg, StubPredictor("parfait"), test_examples)
    assert report.json_valid_rate == 1.0
    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0
    assert report.exact_match == 1.0
    assert report.error_groups == {}
    assert report.latency_mean_s > 0


def test_unparsable_output_is_an_error_not_an_abstention(
    cfg: TaskConfig, test_examples: list[Example]
) -> None:
    report = evaluate_model(cfg, StubPredictor("casse", mode="broken"), test_examples)
    assert report.json_valid_rate == 0.0
    assert report.accuracy == 0.0
    assert report.n == len(test_examples)
    assert report.error_groups.get("json_invalide") == len(test_examples)


def test_off_schema_output_is_rejected(cfg: TaskConfig, test_examples: list[Example]) -> None:
    report = evaluate_model(cfg, StubPredictor("hors-schema", mode="off_schema"), test_examples)
    assert report.json_valid_rate == 0.0
    assert "schema_invalide" in report.error_groups


def test_confusions_are_grouped_by_pair(cfg: TaskConfig, test_examples: list[Example]) -> None:
    report = evaluate_model(cfg, StubPredictor("confus", mode="confused"), test_examples)
    assert any(key.startswith("confusion:") for key in report.error_groups)
    assert report.errors and report.errors[0].category.startswith("confusion:")


def test_free_text_fields_do_not_count_as_errors(cfg: TaskConfig) -> None:
    """Un resume different ne doit pas faire chuter le score : il n'est pas notable."""
    examples = [
        Example(
            input="Le site affiche une erreur 500",
            output={
                "categorie": "technique",
                "urgence": "haute",
                "resume": "un resume ecrit autrement",
            },
        )
    ]
    report = evaluate_model(cfg, StubPredictor("parfait"), examples)
    assert report.accuracy == 1.0
    assert report.exact_match == 1.0
    assert report.field_accuracy["resume"] == 0.0  # visible, mais sans effet sur le score


def test_comparison_computes_ratio_speedup_and_cost(
    cfg: TaskConfig, test_examples: list[Example]
) -> None:
    student = StubPredictor("eleve", kind="student", mode="confused", latency=0.01)
    teacher_model = StubPredictor("professeur", kind="teacher", mode="perfect", latency=0.5)
    baseline = StubPredictor("brut", kind="baseline", mode="broken", latency=0.01)

    result = evaluate(cfg, [teacher_model, baseline, student], test_examples)
    assert result.teacher and result.student and result.baseline
    assert 0.0 < result.quality_ratio < 1.0
    assert result.speedup > 10
    assert result.cost_factor > 1, "le cout API du professeur doit depasser le cout GPU de l'eleve"
    assert result.target_met(cfg.eval.min_json_valid) is False
    assert result.as_dict()["models"][0]["kind"] == "teacher"


def test_target_is_met_when_the_student_matches_the_teacher(
    cfg: TaskConfig, test_examples: list[Example]
) -> None:
    result = evaluate(
        cfg,
        [StubPredictor("professeur", kind="teacher"), StubPredictor("eleve", kind="student")],
        test_examples,
    )
    assert result.quality_ratio == pytest.approx(1.0)
    assert result.target_met(cfg.eval.min_json_valid) is True


# --------------------------------------------------------------------- ameliorateur


def test_plan_turns_errors_into_actionable_directives(
    cfg: TaskConfig, test_examples: list[Example]
) -> None:
    report = evaluate_model(cfg, StubPredictor("confus", mode="confused"), test_examples)
    plan = build_plan(report, "categorie", 100)
    assert plan.directives
    assert any("confond" in d for d in plan.directives)
    assert plan.n_targeted == 100


def test_plan_handles_json_failures(cfg: TaskConfig, test_examples: list[Example]) -> None:
    report = evaluate_model(cfg, StubPredictor("casse", mode="broken"), test_examples)
    plan = build_plan(report, "categorie", 50)
    assert any("hors format" in d for d in plan.directives)


def test_improve_generates_targeted_examples(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example], test_examples: list[Example]
) -> None:
    report = evaluate_model(cfg, StubPredictor("confus", mode="confused"), test_examples)
    result, plan = improve(
        cfg, teacher, seeds, report, primary_field="categorie", iteration=1, n_targeted=10
    )
    assert plan.directives
    assert result.examples
    assert all(e.source == "targeted" for e in result.examples)


def test_improve_does_nothing_without_exploitable_errors(
    cfg: TaskConfig, teacher: FakeTeacher, seeds: list[Example], test_examples: list[Example]
) -> None:
    report = evaluate_model(cfg, StubPredictor("parfait"), test_examples)
    result, plan = improve(cfg, teacher, seeds, report, primary_field="categorie", iteration=1)
    assert plan.directives == [] and result.examples == []
