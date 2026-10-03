"""Publication du modele sur Hugging Face, carte de modele comprise.

La carte n'est pas un gabarit a trous : elle est remplie avec les chiffres du
run, lus dans metrics.json. Publier un modele sans ses limites et sans ses cas
d'echec, c'est exactement ce que le protocole d'evaluation cherche a eviter.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import TaskConfig
from ..errors import ExportError
from ..logging_utils import get_logger
from ..paths import RunPaths
from ..prompts import system_prompt

CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
library_name: gguf
pipeline_tag: text-generation
language:
{languages}
tags:
  - distill
  - slm
  - lora
  - gguf
  - ollama
  - {task_tag}
---

# {model_name}

{description}

Petit modele specialise, distille depuis **{teacher}** avec
[Distill](https://github.com/abdelilah-ourti/distill). Il fait une seule tache,
en local, et rend du JSON conforme a un schema fixe.

## Resultats mesures

{results_table}

{verdict}

Protocole : meme test set de {n_test} exemples, memes entrees, meme prompt
d'inference pour tous les modeles. Une sortie illisible compte comme une
erreur, pas comme une ligne ignoree. Le cout local est derive du temps GPU
reellement occupe, pas suppose nul.

## Utilisation

### Ollama

```bash
ollama run {ollama_name}
```

### llama.cpp

```bash
llama-cli -m {gguf_name} -p "<votre entree>"
```

### Prompt systeme

Le modele a ete entraine avec ce prompt systeme exact. L'utiliser tel quel fait
la difference entre les chiffres ci-dessus et des resultats degrades.

```
{system_prompt}
```

### Schema de sortie

```json
{output_schema}
```

## Entrainement

| | |
|---|---|
| Modele de base | `{base_model}` |
| Methode | LoRA{quant_note} |
| Exemples d'entrainement | {n_train} |
| Donnees | synthetiques, generees par {teacher}, dedupliquees et notees par un juge LLM |
| Test set | {n_test} exemples, isoles **avant** toute generation |
| Quantifications | {quantizations} |

## Limites

{limits}

## Ou le modele echoue

{failures}

Les erreurs completes du run sont dans `eval/errors.jsonl` et dans le rapport
HTML joint.

## Licence et provenance

Code Apache 2.0. Le modele de base `{base_model}` reste soumis a sa propre
licence. Les donnees d'entrainement sont synthetiques : verifiez les conditions
d'utilisation de votre fournisseur de professeur avant toute reutilisation
commerciale.

Produit par **Abdelilah OURTI**, AI Engineer — [neurautech.com](https://neurautech.com).
"""


@dataclass
class PublishPlan:
    """Ce qui partira sur le Hub, decide avant d'envoyer quoi que ce soit."""

    repo: str
    files: list[Path]
    card: Path
    private: bool

    def describe(self) -> str:
        lines = [f"depot : {self.repo}" + (" (prive)" if self.private else "")]
        total = 0
        for path in self.files:
            size = path.stat().st_size if path.is_file() else 0
            total += size
            lines.append(f"  {path.name:<48} {size / 1e6:>8.1f} Mo")
        lines.append(f"  {'total':<48} {total / 1e6:>8.1f} Mo")
        return "\n".join(lines)


def _results_table(models: Sequence[dict[str, Any]]) -> str:
    header = (
        "| Modele | Exactitude | F1 macro | JSON valide | Latence moy. | Cout / 1 000 req. |\n"
        "|---|---|---|---|---|---|"
    )
    rows = [
        "| {name} | {acc:.1%} | {f1:.1%} | {js:.1%} | {lat:.0f} ms | {cost:.4f} $ |".format(
            name=f"**{m['name']}**" if m.get("kind") == "student" else m["name"],
            acc=m.get("accuracy", 0.0),
            f1=m.get("macro_f1", 0.0),
            js=m.get("json_valid_rate", 0.0),
            lat=m.get("latency_mean_s", 0.0) * 1000,
            cost=m.get("cost_per_1k_usd", 0.0),
        )
        for m in models
    ]
    return "\n".join([header, *rows])


def _verdict(metrics: dict[str, Any]) -> str:
    ratio = metrics.get("quality_ratio", 0.0)
    if not ratio:
        return (
            "Aucun professeur dans ce run : les chiffres ci-dessus sont absolus, sans comparaison."
        )
    return (
        f"L'eleve atteint **{ratio:.1%}** du score du professeur, repond "
        f"**{metrics.get('speedup', 0):.1f}x** plus vite et coute "
        f"**{metrics.get('cost_factor', 0):.1f}x** moins cher pour 1 000 requetes."
    )


def _failures(models: Sequence[dict[str, Any]]) -> str:
    student = next((m for m in models if m.get("kind") == "student"), None)
    groups = (student or {}).get("error_groups") or {}
    if not groups:
        return "Aucune erreur enregistree sur ce test set. Elargissez-le avant d'en conclure quoi que ce soit."
    lines = ["Repartition des erreurs sur le test set :", ""]
    for category, count in list(groups.items())[:8]:
        lines.append(f"- `{category}` : {count}")
    return "\n".join(lines)


def _limits(cfg: TaskConfig, models: Sequence[dict[str, Any]]) -> str:
    student = next((m for m in models if m.get("kind") == "student"), None)
    limits = [
        f"- Le modele ne sait faire que cette tache : {cfg.description.strip().splitlines()[0]}",
        "- Les donnees d'entrainement sont synthetiques : les formulations rares du monde reel "
        "peuvent le mettre en difficulte.",
        f"- Langue d'entrainement : {cfg.language}. Hors de cette langue, le comportement n'est pas garanti.",
    ]
    if student:
        json_rate = student.get("json_valid_rate", 1.0)
        if json_rate < 1.0:
            limits.append(
                f"- {1 - json_rate:.1%} des reponses ne respectent pas le schema : validez la sortie "
                "cote appelant, ne la consommez pas aveuglement."
            )
    free_fields = [
        f for f in cfg.schema_fields if not (cfg.output_schema["properties"][f].get("enum"))
    ]
    if free_fields:
        limits.append(
            f"- Les champs libres ({', '.join(free_fields)}) ne sont pas notes par egalite de chaines : "
            "leur qualite n'est pas mesuree par les chiffres ci-dessus."
        )
    return "\n".join(limits)


def build_model_card(cfg: TaskConfig, metrics: dict[str, Any], *, gguf_name: str = "") -> str:
    """Compose la carte de modele a partir des metriques du run."""
    models = metrics.get("models", [])
    student: dict[str, Any] = next((m for m in models if m.get("kind") == "student"), {})
    n_test = student.get("n", 0)
    languages = {
        "francais": "fr",
        "français": "fr",
        "anglais": "en",
        "english": "en",
        "french": "fr",
    }
    lang = languages.get(cfg.language.lower(), cfg.language.lower()[:2])

    return CARD_TEMPLATE.format(
        base_model=cfg.student.model,
        languages=f"  - {lang}",
        task_tag=cfg.name,
        model_name=cfg.model_slug,
        description=cfg.description.strip(),
        teacher=cfg.teacher.model,
        results_table=_results_table(models),
        verdict=_verdict(metrics),
        n_test=n_test,
        ollama_name=cfg.model_slug,
        gguf_name=gguf_name or f"{cfg.model_slug}-Q4_K_M.gguf",
        system_prompt=system_prompt(cfg),
        output_schema=json.dumps(cfg.output_schema, ensure_ascii=False, indent=2),
        quant_note=" / QLoRA 4 bits" if cfg.student.load_in_4bit else "",
        n_train=metrics.get("n_train", "voir le rapport"),
        quantizations=", ".join(cfg.export.quantizations),
        limits=_limits(cfg, models),
        failures=_failures(models),
    )


def plan_publication(
    cfg: TaskConfig, paths: RunPaths, repo: str, *, private: bool = False
) -> PublishPlan:
    """Prepare la carte et la liste des fichiers, sans rien envoyer."""
    if not paths.metrics.is_file():
        raise ExportError(
            f"aucune metrique dans {paths.root} : publier un modele sans ses chiffres n'a pas de sens"
        )
    metrics = json.loads(paths.metrics.read_text(encoding="utf-8"))

    gguf_files = sorted(paths.export_dir.glob("*.gguf"))
    # Le f16 pese plusieurs Go pour rien une fois les versions quantifiees la.
    shipped = [f for f in gguf_files if "-f16" not in f.name] or gguf_files
    if not shipped:
        raise ExportError(
            f"aucun fichier GGUF dans {paths.export_dir} : lance d'abord `distill export`"
        )

    card = paths.export_dir / "README.md"
    card.write_text(
        build_model_card(cfg, metrics, gguf_name=shipped[0].name),
        encoding="utf-8",
    )

    files = [*shipped, card]
    for extra in (paths.modelfile, paths.report, paths.metrics, paths.test_set):
        if extra.is_file():
            files.append(extra)
    return PublishPlan(repo=repo, files=files, card=card, private=private)


def publish(plan: PublishPlan, *, token: str | None = None) -> str:
    """Envoie le modele sur le Hub et retourne l'URL du depot."""
    log = get_logger()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ExportError("huggingface_hub est requis : pip install 'distill-slm[hub]'") from exc

    api = HfApi(token=token)
    api.create_repo(plan.repo, exist_ok=True, private=plan.private, repo_type="model")
    for path in plan.files:
        # La carte doit s'appeler README.md a la racine ; le reste garde son nom.
        name = "README.md" if path == plan.card else path.name
        log.info(f"envoi de {name} ({path.stat().st_size / 1e6:.1f} Mo)")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=plan.repo,
            repo_type="model",
        )
    url = f"https://huggingface.co/{plan.repo}"
    log.info(f"[bold green]modele publie[/] : {url}")
    return url
