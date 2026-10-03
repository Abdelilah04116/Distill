<div align="center">

# Distill

**Ton propre modèle, spécialisé, local et 10× moins cher — sans écrire une ligne de code d'entraînement.**

[![Licence](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/abdelilah-ourti/distill/actions/workflows/ci.yml/badge.svg)](https://github.com/abdelilah-ourti/distill/actions/workflows/ci.yml)

Une description de tâche, une vingtaine d'exemples, une commande.
En sortie : un SLM de 0,5 à 3 milliards de paramètres, évalué, quantifié, et prêt à tourner en local avec Ollama.

</div>

---

## Le problème

Les entreprises paient des LLM géants pour des tâches étroites et répétitives : classer des tickets, extraire des champs, router des demandes. C'est cher, c'est lent, et c'est souvent incompatible avec le RGPD dès que les données partent dans le cloud.

**Distill utilise le gros LLM une seule fois, comme professeur**, pour former un élève qui fait la même tâche en local, pour une fraction du coût.

```bash
distill run --task examples/email-triage/task.yaml
# ... génération, critique, entraînement, évaluation, export ...
ollama run distill-email-triage
```

## Ce que ça fait, étape par étape

```
task.yaml ──▶ Générateur ──▶ Critique ──▶ Entraînement ──▶ Évaluateur
 description      2 à 5k      dédup +        LoRA /          SLM vs
 + 20 seeds      exemples    LLM juge        Unsloth        professeur
                    ▲                                           │
                    │                                           ▼
                    └──── Améliorateur ◀──── non ───  Cible atteinte ?
                          données ciblées                        │ oui
                          (max 3 itérations)                     ▼
                                                    Export : GGUF · Ollama · FastAPI
```

| Agent | Rôle | Fichier |
|---|---|---|
| **Générateur** | 2 000 à 5 000 exemples synthétiques, variés par persona, ton, longueur, fautes, cas limites | [generator.py](distill/agents/generator.py) |
| **Critique** | dédup exacte et sémantique, LLM-as-judge (note 1–5), équilibre des classes, isolation du test set | [critic.py](distill/agents/critic.py) |
| **Entraînement** | LoRA / QLoRA sur un seul GPU, courbe de loss en temps réel, reprise après interruption | [lora_trainer.py](distill/training/lora_trainer.py) |
| **Évaluateur** | élève, professeur et baseline sur le même test set, F1 par classe, latence, coût | [evaluator.py](distill/agents/evaluator.py) |
| **Améliorateur** | lit les erreurs réelles, génère des exemples ciblés, relance (3 fois maximum) | [improver.py](distill/agents/improver.py) |
| **Export** | fusion LoRA, GGUF (Q4_K_M, Q8_0), Modelfile Ollama, API FastAPI, rapport HTML | [export/](distill/export/) |

L'orchestration est un graphe LangGraph avec état persisté ([graph.py](distill/graph.py)) : un run interrompu reprend là où il s'est arrêté. Si LangGraph n'est pas installé, le même enchaînement tourne en séquentiel.

## Installation

```bash
pip install distill-slm              # noyau : CLI, agents, évaluation, rapport
pip install "distill-slm[all]"       # + LangGraph, Qdrant, embeddings, FastAPI
pip install "distill-slm[train]"     # + torch, transformers, peft, trl (GPU)
```

Pour l'entraînement, Unsloth est optionnel mais recommandé — il est deux fois plus rapide et tient dans moins de VRAM :

```bash
pip install unsloth
```

Pour l'export GGUF, il faut llama.cpp compilé :

```bash
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
cmake -B build && cmake --build build --config Release
export LLAMA_CPP_DIR=$PWD
```

Et une clé pour le professeur :

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Démarrage rapide

```bash
# 1. Vérifier la tâche et les exemples avant de dépenser quoi que ce soit
distill validate --task examples/email-triage/task.yaml
distill estimate --task examples/email-triage/task.yaml

# 2. Lancer le pipeline complet
distill run --task examples/email-triage/task.yaml

# 3. Essayer le modèle
ollama run distill-email-triage

# 4. Le servir en HTTP
distill serve --task examples/email-triage/task.yaml
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"input":"Bonjour, je n ai pas recu ma facture de janvier."}'
```

Autres commandes utiles :

```bash
distill run --task t.yaml --dry-run          # plan + estimation de coût, sans rien lancer
distill run --task t.yaml --resume           # reprend le dernier run
distill run --task t.yaml --skip-generation  # réutilise les exemples déjà produits
distill watch --task t.yaml                  # suivi en direct : étapes, coût, courbe de loss
distill eval --task t.yaml --backend ollama  # réévalue le modèle quantifié réellement livré
distill report --task t.yaml                 # régénère le rapport HTML
distill cache --clear                        # vide le cache des appels au professeur
```

## Le fichier `task.yaml`

Tout le pipeline est piloté par un seul fichier. Le minimum vital :

```yaml
name: email-triage
description: Tu tries les emails recus par le service client d'une PME francaise.
language: francais

output_schema:
  type: object
  additionalProperties: false
  required: [categorie, urgence, resume]
  properties:
    categorie:
      type: string
      enum: [reclamation, facturation, technique, commercial, autre]
    urgence:
      type: string
      enum: [basse, moyenne, haute]
    resume:
      type: string

seeds: seeds.jsonl        # 10 à 50 exemples entrée → sortie attendue
test_set: test_set.jsonl  # écrit à la main, jamais vu à l'entraînement

teacher:
  provider: anthropic
  model: claude-opus-5-5

student:
  model: Qwen/Qwen2.5-1.5B-Instruct

budget:
  max_examples: 3000
  max_cost_usd: 5.0       # plafond dur : dépassé, le run s'arrête proprement
  max_duration_min: 120
```

L'exemple complet et commenté : [examples/email-triage/task.yaml](examples/email-triage/task.yaml).

### Mode 100 % local (RGPD strict)

Aucune donnée ne quitte la machine, ni pendant l'entraînement ni après. Le professeur devient un modèle open-weight servi par vLLM :

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000
```

```yaml
teacher:
  provider: openai_compatible
  base_url: http://localhost:8000/v1
  model: Qwen/Qwen2.5-32B-Instruct
  api_key_env: DISTILL_TEACHER_API_KEY
  temperature: 0.9
```

`distill estimate` affiche alors explicitement qu'aucun coût API n'est engagé. C'est aussi la réponse au risque juridique : certaines conditions d'utilisation interdisent d'entraîner un modèle sur les sorties du fournisseur — vérifiez les vôtres, ou passez en local.

## Évaluation : le protocole

La crédibilité du projet tient à un benchmark honnête. Les règles sont appliquées par le code, pas seulement écrites ici :

1. **Le test set est isolé avant toute génération.** Il n'entre jamais dans l'entraînement.
2. **Double garde contre la fuite.** Dédup exacte par empreinte, puis dédup sémantique par embeddings contre le test set — un exemple généré trop proche d'un exemple de test est jeté, pas utilisé.
3. **Mêmes entrées, même prompt d'inférence pour tous les modèles.** Le professeur ne bénéficie d'aucun prompt enrichi ([prompts.py](distill/prompts.py) est la source unique).
4. **Une sortie illisible est une erreur**, pas une ligne à ignorer. Un modèle qui répond n'importe quoi une fois sur dix n'obtient pas 100 % sur les neuf autres.
5. **Une baseline est systématiquement mesurée** : le même petit modèle sans fine-tuning. C'est ce qui prouve l'apport de la distillation plutôt que celui du modèle de base.
6. **Les échecs sont publiés.** Le rapport HTML contient les erreurs réelles de l'élève, non sélectionnées.

### Comparer les trois tailles d'élève

Le protocole demande de montrer le compromis taille / qualité. Chaque taille est un run complet, avec le même `task.yaml` et le même test set :

```bash
for m in Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct; do
  distill run --task examples/email-triage/task.yaml --student "$m" --skip-generation
done
```

`--skip-generation` à partir du deuxième run réutilise le corpus déjà produit : les trois élèves apprennent sur exactement les mêmes données, et la comparaison porte bien sur la taille du modèle. Les trois rapports sont dans `runs/email-triage/`.

### Tableau de résultats

Les chiffres sont produits par votre run, sur votre machine. Le rapport les écrit dans `runs/<tâche>/<run>/report.html` et `metrics.json`.

| Modèle | Exactitude | F1 macro | JSON valide | Latence moy. | Coût / 1 000 req. |
|---|---|---|---|---|---|
| LLM professeur | à mesurer | à mesurer | à mesurer | à mesurer | à mesurer |
| Qwen 1.5B brut | à mesurer | à mesurer | à mesurer | à mesurer | à mesurer |
| **Distill 1.5B** | à mesurer | à mesurer | à mesurer | à mesurer | à mesurer |

Objectifs de la v1 : **≥ 90 %** du score du professeur, **≥ 10×** moins cher pour 1 000 requêtes, **≥ 5×** plus rapide, le tout en **moins de 2 h sur un GPU**.

> Le coût local n'est pas zéro : il est dérivé du temps GPU réellement occupé, au prix horaire que vous déclarez dans `eval.local_gpu_cost_per_hour`. Comparer un coût API à « gratuit » serait malhonnête.

## Matériel

| Élève | VRAM nécessaire (QLoRA 4 bits) | Où |
|---|---|---|
| 0.5B | ~4 Go | Colab / Kaggle gratuit |
| 1.5B | ~8 Go | Colab / Kaggle gratuit, RTX 3060 |
| 3B | ~14 Go | RTX 4090, L4, A10 |

Sans GPU, tout le reste fonctionne : génération, critique, évaluation d'un modèle Ollama, rapport. Seul `--skip-training` est alors nécessaire.

## Structure du dépôt

```
distill/
├── distill/
│   ├── cli.py                  # distill run / eval / export / serve / report
│   ├── graph.py                # graphe LangGraph + état persisté
│   ├── config.py               # task.yaml validé par Pydantic
│   ├── schema.py               # JSON schema → modèle Pydantic
│   ├── prompts.py              # le prompt d'inférence, défini une seule fois
│   ├── metrics.py              # F1, exactitude, percentiles (sans sklearn)
│   ├── costs.py                # tarifs, suivi du budget, coût par 1 000 requêtes
│   ├── dedup.py                # embeddings + Qdrant, avec repli sans dépendance
│   ├── agents/                 # generator · critic · evaluator · improver
│   ├── training/               # lora_trainer (Unsloth ou TRL) · merge
│   ├── export/                 # gguf · ollama
│   ├── students/               # runners HF, Ollama, professeur
│   ├── serving/api.py          # FastAPI /predict
│   └── report/                 # rapport HTML autonome
├── examples/email-triage/      # task.yaml + 24 seeds + 201 exemples de test
├── docker/                     # image de service
└── tests/
```

Un run produit un dossier autonome :

```
runs/email-triage/20261003-141205/
├── task.resolved.yaml     # la config exacte utilisée
├── data/                  # generated · train · test · rejected (.jsonl)
├── train/                 # adaptateur LoRA, checkpoints, loss.jsonl
├── eval/                  # metrics.json, errors.jsonl
├── export/                # GGUF + Modelfile
├── logs/events.jsonl      # journal structuré de toutes les étapes
└── report.html
```

## Coûts

Le budget est plafonné dans `task.yaml` et appliqué pendant le run, pas après :

- `distill estimate` affiche une estimation **avant** de dépenser ;
- chaque appel au professeur est mis en cache sur disque — relancer un run ne repaie pas la génération ;
- au dépassement du plafond, le run s'arrête proprement, l'état et les données déjà produites sont conservés.

## Limites connues

Il vaut mieux les lire avant de lancer un run que de les découvrir après.

- **Périmètre v1 : tâches textuelles à sortie structurée.** Classification, extraction de champs, routage. Pas de génération de texte long et libre, pas de multimodal, pas de multi-GPU.
- **Les champs libres ne sont pas notés par égalité de chaînes.** Un résumé est comparé à titre indicatif ; le score porte sur les champs à valeurs fermées. Évaluer un résumé correctement demande un juge, pas un `==`.
- **Le test set fourni dans `examples/` a été rédigé pour ce dépôt et doit être relu avant de publier des chiffres.** Le protocole parle d'un test set validé à la main : relisez-le, corrigez-le, c'est votre garantie d'honnêteté, pas la nôtre.
- **La latence du professeur est mesurée sous concurrence** (4 appels en parallèle par défaut), celle de l'élève requête par requête. Le rapport précise la concurrence utilisée.
- **Unsloth ne tourne pas partout.** Sans lui, le pipeline bascule sur Transformers + PEFT + TRL, plus lent mais équivalent.
- **Le suivi est un tableau de bord de terminal, pas une interface web.** `distill watch` lit les fichiers que le run écrit de toute façon ; il n'y a ni serveur, ni base, ni historique multi-runs.

## Suivre un run en cours

Un run dure jusqu'à deux heures. Dans un second terminal :

```bash
distill watch --task examples/email-triage/task.yaml
```

Étapes en cours et terminées, nombre d'exemples retenus, ratio courant face au professeur, et la courbe de loss en sparkline. Tout vient de `logs/events.jsonl`, `train/loss.jsonl` et `state.json`, donc le suivi fonctionne aussi après coup (`--once`) et sur un run lancé par quelqu'un d'autre.

## Développement

```bash
git clone https://github.com/abdelilah-ourti/distill && cd distill
pip install -e ".[dev,serve]"
pytest                  # tests unitaires, sans GPU ni réseau
ruff check distill
mypy distill
```

Les tests ne font aucun appel réseau : le professeur est remplacé par un faux déterministe ([tests/conftest.py](tests/conftest.py)).

## Aller plus loin

- [docs/LANCEMENT.md](docs/LANCEMENT.md) — checklist de publication, textes des posts, carte de modèle Hugging Face. Les chiffres y sont volontairement laissés en blanc : ne publiez que ce que vous avez mesuré.

## Licence et crédits

Apache 2.0 — voir [LICENSE](LICENSE).

Les modèles élèves (Qwen 2.5, Llama 3.2) et les modèles professeurs restent soumis à leurs licences respectives.

Créé par **Abdelilah OURTI**, AI Engineer — [neurautech.com](https://neurautech.com).
