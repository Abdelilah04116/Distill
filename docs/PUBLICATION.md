# Publier `distill-slm` sur PyPI

**La version 0.1.0 est publiée** depuis le 4 octobre 2026 : [pypi.org/project/distill-slm](https://pypi.org/project/distill-slm/). Ce document sert désormais aux versions suivantes.

## Piège rencontré au premier envoi

Si `twine` échoue sur `OSError: Could not find a suitable TLS CA certificate bundle`, c'est l'installeur PostgreSQL qui a défini `CURL_CA_BUNDLE` au niveau Machine vers un fichier absent. `requests` la lit avant son propre magasin, donc toute requête HTTPS Python casse. Correctif ponctuel :

```powershell
Remove-Item Env:CURL_CA_BUNDLE
```

Correctif durable, sans droits administrateur (une variable Utilisateur prend le pas sur la Machine) :

```powershell
$dest = "$env:USERPROFILE\.certs\ca-bundle.crt"
New-Item -ItemType Directory -Force (Split-Path $dest) | Out-Null
Copy-Item (python -c "import certifi;print(certifi.where())") $dest
[Environment]::SetEnvironmentVariable('CURL_CA_BUNDLE', $dest, 'User')
```

## Avant de publier

Une version sur PyPI ne se remplace pas. On peut supprimer un fichier, jamais réutiliser le numéro. Donc :

- [ ] Un run complet a réellement eu lieu. Les chemins entraînement, fusion LoRA et export GGUF n'ont jamais été exécutés : publier avant de les avoir vus tourner, c'est livrer du code non éprouvé à des inconnus.
- [ ] `pytest`, `ruff check`, `ruff format --check` et `mypy` passent.
- [ ] Le dépôt est public sur `github.com/Abdelilah04116/distill` — le README de PyPI pointe dessus pour ses liens et son image.
- [ ] `project.version` dans `pyproject.toml` est le numéro que vous voulez figer.
- [ ] Un essai sur **TestPyPI** a réussi (voir plus bas).

## La méthode recommandée : trusted publishing

Pas de jeton à stocker, pas de secret à faire fuiter. PyPI vérifie l'identité du workflow GitHub par OIDC. C'est à configurer une seule fois.

### 1. Déclarer l'éditeur sur PyPI

Sur [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/), ajouter un *pending publisher* :

| Champ | Valeur |
|---|---|
| PyPI Project Name | `distill-slm` |
| Owner | `Abdelilah04116` |
| Repository name | `distill` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Faire la même chose sur [test.pypi.org](https://test.pypi.org/manage/account/publishing/) avec l'environment `testpypi`.

### 2. Créer les environnements GitHub

Dans *Settings → Environments* du dépôt, créer `pypi` et `testpypi`. Sur `pypi`, ajoutez-vous comme *required reviewer* : la publication attendra votre validation manuelle, ce qui évite un tag poussé par erreur.

### 3. Répéter sur TestPyPI

Depuis l'onglet *Actions* → *Publication PyPI* → *Run workflow*, cible `testpypi`. Puis vérifier depuis un environnement neuf :

```bash
python -m venv /tmp/essai && source /tmp/essai/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ distill-slm
distill version
distill validate --task examples/email-triage/task.yaml   # depuis un clone du dépôt
```

`--extra-index-url` est indispensable : TestPyPI n'héberge pas les dépendances (typer, pydantic, anthropic…).

### 4. Publier pour de vrai

```bash
git tag v0.1.0
git push origin v0.1.0
```

Le workflow rejoue les tests, vérifie que le tag correspond à `project.version`, construit, puis publie. Si vous avez mis un reviewer sur l'environnement `pypi`, il attend votre clic.

## La méthode manuelle

Si vous préférez ne pas passer par GitHub Actions :

```bash
pip install build twine
rm -rf dist
python -m build
twine check --strict dist/*

twine upload --repository testpypi dist/*   # répétition
twine upload dist/*                          # pour de vrai
```

Il faut alors un jeton d'API créé sur [pypi.org/manage/account/token](https://pypi.org/manage/account/token/), à mettre dans `~/.pypirc` :

```ini
[pypi]
  username = __token__
  password = pypi-AgEIcHlwaS5vcmc...
```

Ce fichier contient un secret en clair : `chmod 600 ~/.pypirc`, et jamais dans le dépôt.

## Après la première publication

- Repasser le `pending publisher` en éditeur normal sur PyPI (c'est automatique à la première publication réussie).
- Rétablir dans le README l'installation par `pip install distill-slm`, aujourd'hui remplacée par l'installation depuis le dépôt.
- Cocher la ligne correspondante dans [LANCEMENT.md](LANCEMENT.md) avant d'annoncer le paquet sur LinkedIn.

## Publier une version suivante

1. Monter `project.version` dans `pyproject.toml`.
2. Commiter, taguer `vX.Y.Z`, pousser le tag.

Le workflow refuse de publier si le tag et la version divergent — c'est volontaire, c'est l'erreur la plus courante.

## Points de vigilance propres à ce paquet

**Le README sert de page PyPI.** Ses liens et son image sont en URL absolues vers GitHub ; les liens relatifs ne fonctionneraient pas. Si vous renommez le dépôt ou changez de compte, il faut les reprendre.

**Les extras pèsent lourd.** `pip install distill-slm` n'installe que le noyau. `[train]` tire torch et ses quelques gigaoctets — c'est voulu, et c'est pour cela que l'extra existe.

**La licence du modèle n'est pas celle du code.** Apache 2.0 couvre Distill. Les modèles élèves et les sorties du professeur gardent leurs propres conditions ; le fichier [NOTICE](NOTICE) le rappelle, gardez-le dans la distribution.
