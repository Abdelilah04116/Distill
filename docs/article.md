---
titre: "J'ai remplacé un LLM géant par un modèle de 1,5 milliard de paramètres. Voici les chiffres."
auteur: Abdelilah OURTI
role: AI Engineer
site: neurautech.com
date: À COMPLÉTER
statut: brouillon — les valeurs entre crochets attendent les chiffres de votre run
---

> **Avant publication** : chaque `[...]` est un chiffre à reprendre dans `runs/<tâche>/<run>/report.html`. N'en inventez aucun. Un article technique qui annonce des résultats non reproductibles se fait démonter en commentaires, et à juste titre.

---

Une entreprise que j'accompagne appelait un LLM haut de gamme 40 000 fois par mois. Pas pour écrire, pas pour raisonner : pour classer des emails en cinq catégories et leur attribuer un niveau d'urgence. Une tâche qu'un stagiaire apprend en une matinée, facturée au prix du modèle le plus cher du marché.

Le calcul est vite fait, et il est inconfortable : on paie un moteur de fusée pour aller chercher le pain.

J'ai voulu vérifier une intuition simple. Si la tâche est étroite et que la sortie est structurée, un modèle de 1,5 milliard de paramètres devrait pouvoir l'apprendre — à condition d'avoir les bonnes données. Et les bonnes données, le gros modèle peut les produire lui-même, une seule fois.

C'est le principe de la distillation. Ce qui suit, c'est ce que j'ai appris en l'outillant de bout en bout, et les chiffres que ça donne.

## Le problème n'est pas le modèle, c'est les données

La partie fine-tuning est largement résolue. LoRA, QLoRA, Unsloth : quelques centaines de lignes et un GPU à 0,60 $ de l'heure suffisent. Ce n'est plus là que ça coince.

Ça coince sur le jeu de données. Demandez à un gros modèle « génère-moi 3 000 emails de réclamation » et vous obtiendrez 3 000 variantes du même email poli, bien ponctué, de longueur moyenne, écrit par la même personne imaginaire. Un petit modèle entraîné là-dessus obtient d'excellents scores sur des données qui lui ressemblent, et s'effondre sur du vrai courrier.

Trois mécanismes corrigent ça, et ils doivent être dans le code, pas dans les bonnes intentions.

**Forcer la variation à chaque appel.** À chaque lot généré, le pipeline tire une persona, un ton, une longueur, un niveau de fautes de frappe et, une fois sur cinq, une consigne de cas limite : « mélange deux sujets, un seul doit l'emporter », « reste ambigu entre deux catégories », « moins de dix mots ». Ce n'est pas de la décoration : c'est la différence entre un corpus qui couvre l'espace du problème et un corpus qui couvre un point.

**Un agent critique, entre la génération et l'entraînement.** Il déduplique sémantiquement — deux emails peuvent être différents mot pour mot et identiques pour le modèle —, fait noter chaque exemple de 1 à 5 par un juge LLM, et rejette sous 4. Sur mon run, [N] % des exemples générés ont été écartés. C'est de l'argent dépensé puis jeté, et c'est le meilleur investissement du pipeline.

**Vérifier l'équilibre des classes et le dire.** Si la catégorie « autre » représente 3 % du corpus, le modèle apprendra à ne jamais la prédire, et l'exactitude globale restera flatteuse. Le pipeline le signale avant l'entraînement, pas après.

## La partie qui décide de tout : le protocole d'évaluation

Je pourrais vous annoncer n'importe quel chiffre. Ce qui compte, c'est ce qui le rend vérifiable.

**Le test set est écrit à la main et isolé avant toute génération.** [201] exemples, relus un par un. Il n'entre jamais dans l'entraînement, et le pipeline s'en assure deux fois : par empreinte exacte, puis par similarité d'embeddings. Un exemple généré trop proche d'un exemple de test est jeté. Sans cette seconde passe, la fuite se produit silencieusement — le générateur a vu les seeds, il reformule, et la reformulation tombe parfois pile sur un exemple de test.

**Le professeur et l'élève reçoivent exactement le même prompt.** Celui-ci est défini à un seul endroit du code et réutilisé à l'entraînement, à l'évaluation, dans le Modelfile Ollama et dans l'API. Donner au gros modèle un prompt plus riche qu'à l'élève produit un benchmark flatteur et faux.

**Une sortie illisible est une erreur.** Pas une ligne à ignorer. Un modèle qui répond n'importe quoi une fois sur dix n'obtient pas 100 % sur les neuf autres. C'est le biais le plus fréquent des benchmarks maison, et le plus facile à commettre de bonne foi.

**Une baseline est systématiquement mesurée** : le même petit modèle, sans fine-tuning. Sans elle, on ne sait pas si on mesure l'apport de la distillation ou simplement la qualité du modèle de base.

**Et le coût local n'est pas zéro.** Il est dérivé du temps GPU réellement occupé, multiplié par le prix horaire de la machine. Comparer un coût d'API à « gratuit parce que c'est chez moi » n'est pas un argument, c'est une omission.

## Les résultats

| Modèle | Exactitude | F1 macro | JSON valide | Latence moy. | Coût / 1 000 req. |
|---|---|---|---|---|---|
| LLM professeur | [..] | [..] | [..] | [..] ms | [..] $ |
| Qwen 2.5 1.5B brut | [..] | [..] | [..] | [..] ms | [..] $ |
| **Distill 1.5B** | [..] | [..] | [..] | [..] ms | [..] $ |

L'élève atteint **[XX] %** du score du professeur, répond **[X]×** plus vite et coûte **[XX]×** moins cher pour 1 000 requêtes. Le run complet a duré [XX] minutes sur [GPU] et consommé [X,XX] $ d'API.

La ligne qui m'intéresse le plus n'est pas la première. C'est la deuxième : le modèle brut, sans fine-tuning, obtient [..]. Tout l'écart entre cette ligne et la troisième, c'est ce que la distillation a apporté — et c'est ce qu'un benchmark sans baseline aurait attribué au modèle de base.

## Là où le petit modèle échoue

[N] erreurs sur [201] exemples. Elles ne sont pas réparties au hasard, et c'est une bonne nouvelle : des erreurs structurées se corrigent.

- **[catégorie A] confondue avec [catégorie B]** — [n] cas. [Explication : où est l'ambiguïté réelle dans le texte.]
- **[second motif]** — [n] cas. [Explication.]

Quand l'élève reste sous la cible, le pipeline ne régénère pas 3 000 exemples au hasard : il lit les erreurs, en déduit des consignes précises — « produis des cas où la catégorie vaut X mais qu'on confond facilement avec Y » — et fait générer un lot ciblé. Trois itérations au maximum, puis rapport final, même si la cible n'est pas atteinte. Un pipeline qui boucle indéfiniment est un pipeline cassé.

Sur mon run, [l'itération 1 a fait passer le score de X à Y / la cible a été atteinte dès la première passe].

## Ce que la distillation ne résout pas

Il faut le dire clairement, parce que c'est là que les projets se cassent la figure en production.

**Le périmètre est étroit, par construction.** Classification, extraction de champs, routage. Un modèle de 1,5B ne remplacera pas un LLM généraliste sur une tâche ouverte, et prétendre le contraire dessert l'argument.

**Les champs libres ne sont pas mesurés.** Dans ma tâche, le modèle produit aussi un résumé d'une ligne. Je ne le compte pas dans le score : comparer deux résumés par égalité de chaînes mettrait tout le monde à zéro, professeur compris. L'évaluer correctement demande un juge, donc un autre LLM, donc un autre protocole.

**Les données synthétiques restent synthétiques.** Elles couvrent bien l'espace qu'on leur a décrit. Les formulations que personne n'a imaginées restent hors couverture, et c'est précisément ce que le test set écrit à la main sert à révéler.

**Les conditions d'utilisation du professeur comptent.** Certains fournisseurs interdisent d'entraîner un modèle sur leurs sorties. Vérifiez les vôtres. C'est aussi pour ça que le pipeline accepte un professeur open-weight servi localement par vLLM : aucune donnée ne sort de la machine, et la question juridique disparaît en même temps que la question RGPD.

## Ce que j'en retire

La distillation n'est pas une technique de recherche, c'est une décision d'architecture. La question n'est pas « est-ce que ça marche », c'est « cette tâche est-elle assez étroite pour que ça marche ». Pour classer des tickets, extraire des champs de factures, router des intentions : oui, et l'économie est d'un ordre de grandeur.

Ce qui m'a le plus occupé n'a pas été l'entraînement. C'est tout ce qui l'entoure : rendre les données variées, empêcher la fuite du test set, faire en sorte que les chiffres annoncés soient les mêmes que ceux qu'on obtient en relançant. L'intelligence du système est dans l'agent critique et dans le protocole d'évaluation, pas dans la boucle de gradient.

Le code est ouvert, sous Apache 2.0, et il tourne en une commande :

```bash
pip install distill-slm
distill run --task examples/email-triage/task.yaml
ollama run distill-email-triage
```

[github.com/abdelilah-ourti/distill](https://github.com/abdelilah-ourti/distill)

Sur quelle tâche répétitive payez-vous encore un modèle généraliste ?

---

*Abdelilah OURTI est AI Engineer. Il écrit sur [neurautech.com](https://neurautech.com).*
