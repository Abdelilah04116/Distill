# Démo : GIF et vidéo

Deux livrables, deux outils. Le GIF du README est automatisable ; la vidéo de 60 à 90 secondes demande un enregistrement réel, parce que son intérêt est justement de montrer que ça tourne.

## Le GIF du README

```bash
go install github.com/charmbracelet/vhs@latest
vhs demo/demo.tape          # produit demo/distill.gif
```

[`demo.tape`](demo.tape) ne joue que des commandes qui fonctionnent hors ligne, plus la relecture d'un run déjà terminé. Pour que la séquence du milieu montre un vrai run, lancez-le d'abord une fois : grâce au cache des réponses du professeur, le second passage est quasi instantané et ne coûte rien.

```bash
distill run --task examples/email-triage/task.yaml   # une fois, pour de vrai
vhs demo/demo.tape                                   # le run rejoué sort du cache
```

Visez moins de 8 Mo, sinon GitHub ne charge pas l'animation sur mobile. Si le fichier est trop gros : `Set Width 980`, `Set FontSize 13`, et coupez la séquence 1.

## La vidéo de 60 à 90 secondes

Trois moments, rien d'autre. Chaque seconde passée ailleurs dilue la démonstration.

| Temps | Plan | Ce qu'on voit | Ce qu'on dit |
|---|---|---|---|
| 0:00–0:10 | Terminal, `task.yaml` à l'écran | Le fichier de tâche, 24 seeds | « Une description de tâche. Vingt-quatre exemples. » |
| 0:10–0:20 | `distill estimate` puis `distill run` | L'estimation de coût, le run qui démarre | « Une commande. Le gros modèle ne servira qu'une fois. » |
| 0:20–0:40 | Accéléré ×20 sur le run, ou `distill watch` | Étapes qui défilent, courbe de loss qui descend | « Génération, critique, entraînement LoRA, évaluation. » |
| 0:40–0:60 | Le tableau comparatif, plein écran | Les trois lignes, les chiffres | « Le petit modèle atteint X % du score, pour Y fois moins cher. » |
| 0:60–0:75 | `ollama run distill-email-triage` | Un email tapé, le JSON qui sort | « Et il tourne en local. Zéro donnée envoyée. » |
| 0:75–0:90 | Le rapport HTML, scroll sur la section des échecs | Les erreurs réelles | « Y compris là où il se trompe. » |

Points qui font la différence :

- **Ne coupez pas sur le tableau.** C'est le seul plan que les gens mettront en pause. Laissez-le trois secondes de plus que ce qui semble confortable.
- **Montrez la section des échecs.** Un post qui ne montre que les succès se fait démonter en commentaires ; un post qui montre ses erreurs n'a rien à défendre.
- **Pas de musique, pas de transitions.** Un terminal qui tourne se suffit.
- **Sous-titres en dur.** La plupart des lectures LinkedIn se font sans le son.

Enregistrement : [asciinema](https://asciinema.org) pour le terminal (`asciinema rec demo.cast`), OBS pour le plein écran avec le rapport HTML. Le `--font-size 16` minimum, sinon rien n'est lisible sur téléphone.

## Capture des courbes d'entraînement

Pour le post teaser, pendant le sprint 2 :

```bash
distill watch --task examples/email-triage/task.yaml
```

La sparkline de loss et le tableau des étapes suffisent, et ils ont l'avantage d'être vrais. Capture plein écran du terminal, rien à retoucher.
