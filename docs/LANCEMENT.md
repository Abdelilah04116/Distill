# Plan de lancement

Checklist et textes prêts à publier. Les crochets `[...]` sont les seuls endroits à remplir : ce sont les chiffres de votre run, pris dans `runs/<tâche>/<run>/report.html`. **Ne publiez aucun chiffre que vous n'avez pas mesuré vous-même** — c'est le seul point sur lequel un post technique se fait démonter en commentaires.

## Avant de publier

- [ ] Un run complet est passé, rapport HTML à l'appui.
- [ ] Le test set a été relu à la main, ligne par ligne. C'est la garantie du benchmark.
- [ ] Les trois tailles d'élève (0.5B, 1.5B, 3B) ont tourné sur le même corpus.
- [ ] La baseline (modèle brut, sans fine-tuning) figure dans le tableau.
- [ ] Le dépôt est public, `pip install distill-slm` fonctionne depuis un environnement neuf.
- [ ] Le modèle GGUF est publié sur Hugging Face avec sa carte de modèle et ses limites.
- [ ] La vidéo de 60 à 90 s est enregistrée : une commande, le tableau, `ollama run`.

## Les quatre publications

### 1. Teaser — pendant le sprint 2

> J'essaie de faire battre un gros LLM par un modèle de 1B sur une vraie tâche : trier des emails clients en français.
>
> Pas en général. Sur une tâche précise, avec un test set écrit à la main et jamais vu à l'entraînement.
>
> Voilà les courbes d'entraînement de ce matin. Résultat chiffré dans 10 jours, succès ou échec.
>
> Sur quelle tâche répétitive aimeriez-vous arrêter de payer un LLM géant ?

*Visuel : capture de `distill watch` ou de la courbe de loss. Pas de lien.*

### 2. Post principal — jour du lancement

> Un modèle de [1,5]B qui atteint [XX] % du score d'un LLM géant sur une tâche précise, pour [XX]× moins cher et [X]× plus vite.
>
> La tâche : trier un email client en catégorie + urgence + résumé, en JSON strict.
> Le protocole : même test set de [201] exemples écrits à la main, mêmes entrées, même prompt, aucune fuite.
>
> | Modèle | F1 macro | JSON valide | Latence | Coût / 1 000 req. |
> |---|---|---|---|---|
> | Professeur | [..] | [..] | [..] | [..] |
> | Qwen 1.5B brut | [..] | [..] | [..] | [..] |
> | Distill 1.5B | [..] | [..] | [..] | [..] |
>
> Le gros modèle ne sert qu'une fois, comme professeur. Ensuite tout tourne en local : zéro donnée envoyée au cloud, ce qui règle aussi la question RGPD.
>
> Une commande, de bout en bout : `distill run --task email-triage.yaml`
>
> Sur quelle tâche voudriez-vous tester ça ?

*Visuel : la vidéo de 60 s, en natif. Lien GitHub en premier commentaire.*

### 3. Post technique — J+3

Carrousel PDF, une idée par page :

1. Le problème : payer un LLM géant pour classer des tickets.
2. L'architecture : 6 agents, 1 boucle, un graphe LangGraph.
3. Pourquoi un agent Critique — ce que donne un corpus synthétique sans dédup ni juge.
4. Pourquoi LoRA plutôt qu'un fine-tuning complet, à budget VRAM fixé.
5. Comment on évite la fuite du test set : isolation avant génération + dédup croisée par embeddings.
6. Pourquoi une sortie illisible compte comme une erreur, et ce que ça change au score.
7. Les chiffres.
8. Le lien.

### 4. Post honnête — J+7

> Là où mon petit modèle échoue.
>
> [N] erreurs sur [201] exemples de test. Elles ne sont pas réparties au hasard :
>
> - [catégorie A] confondue avec [catégorie B] : [n] fois. [Pourquoi.]
> - [autre motif] : [n] fois.
>
> Ce que j'en retiens : [leçon].
> Ce que je ne sais toujours pas faire : [limite].
>
> Les erreurs complètes sont dans le rapport HTML du dépôt. Je préfère les publier que les cacher : c'est ce qui rend les autres chiffres croyables.

*Les erreurs réelles sont dans `runs/<tâche>/<run>/eval/errors.jsonl`.*

## Bonnes pratiques

- Lien GitHub en premier commentaire, jamais dans le post.
- Une image ou une vidéo native par post, pas de lien externe qui sort le lecteur.
- Finir par une question ouverte.
- Répondre à tous les commentaires dans la première heure.
- Recycler chaque post en article sur neurautech.com.

## Carte du modèle Hugging Face

À publier avec le GGUF, en reprenant du rapport : la tâche et son schéma de sortie, le modèle de base et la méthode (LoRA, rang, nombre d'exemples), le tableau de résultats, le prompt système exact (celui du Modelfile), les limites connues et les cas d'échec, la licence du modèle de base, et la mention que les données d'entraînement sont synthétiques.
