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

## Les publications

### 0. Annonce du dépôt — publiable aujourd'hui

À publier avant d'avoir le moindre chiffre. Il parle de la méthode et de l'architecture, pas des résultats, donc il n'engage rien qui ne soit pas vérifiable dans le code. Visuel : `docs/architecture-linkedin.png`. Lien GitHub en premier commentaire.

> Classer un email en cinq catégories ne demande pas un modèle à plusieurs centaines de milliards de paramètres.
>
> On continue pourtant à payer pour ça, à chaque requête.
>
> J'ai passé les dernières semaines à construire l'alternative, et je viens de l'ouvrir en open source.
>
> Le principe : utiliser le gros LLM UNE seule fois, comme professeur, pour former un élève de 1,5 milliard de paramètres qui fait la même tâche en local.
>
> Ce n'est pas le fine-tuning qui est difficile. LoRA sur un GPU à 0,60 $ de l'heure, c'est un problème résolu.
>
> Le difficile, c'est les données.
>
> Demandez 3 000 emails à un gros modèle : vous obtiendrez 3 000 variantes du même email poli, bien ponctué, écrit par la même personne imaginaire. Le petit modèle apprend ça par cœur, puis s'effondre sur du vrai courrier.
>
> D'où un système multi-agents, et pas un script :
>
> 1. Générateur — tire une persona, un ton, une longueur, un niveau de fautes à chaque lot
> 2. Critique — déduplique sémantiquement, fait noter chaque exemple de 1 à 5, rejette sous 4
> 3. Entraîneur — LoRA / QLoRA sur un seul GPU, reprise après interruption
> 4. Évaluateur — élève contre professeur contre modèle brut, même test set, même prompt
> 5. Améliorateur — lit les erreurs réelles, génère des exemples ciblés, trois tours maximum
> 6. Exporteur — GGUF, Modelfile Ollama, API FastAPI, rapport HTML
>
> Orchestré par LangGraph avec état persisté : un run coupé reprend où il s'est arrêté.
>
> Trois décisions qui déterminent si les chiffres voudront dire quelque chose :
>
> → Le test set est écrit à la main et isolé AVANT toute génération. Double garde contre la fuite : empreinte exacte, puis similarité d'embeddings.
> → Une sortie illisible compte comme une erreur, pas comme une ligne à ignorer.
> → Le coût local n'est pas zéro : il est dérivé du temps GPU réellement occupé.
>
> Objectif affiché : 90 % du score du professeur, 10× moins cher, 5× plus rapide. Je publie les mesures la semaine prochaine, après le premier run complet — avec les cas d'échec, pas seulement les succès. C'est ce qui rend un benchmark croyable.
>
> Le code est sous Apache 2.0 et tourne en une commande.
>
> Sur quelle tâche répétitive payez-vous encore un modèle généraliste ?

**Premier commentaire** :

> Le dépôt : github.com/abdelilah-ourti/distill
>
> Il y a un notebook Colab si vous voulez l'essayer sans GPU : une clé API, et le pipeline tourne de bout en bout sur le T4 gratuit.

**Avant de poster** : si vous avez un cas client réel à raconter en ouverture, remplacez les deux premières lignes par celui-là. Une anecdote vécue vaut mieux qu'une généralité — mais ne racontez que ce qui s'est vraiment passé.

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
