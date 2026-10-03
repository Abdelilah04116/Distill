# Bande VHS : produit le GIF de demonstration du README.
#
#   go install github.com/charmbracelet/vhs@latest
#   vhs demo/demo.tape
#
# Elle ne depend d'aucune cle API : les commandes jouees sont celles qui
# tournent hors ligne, plus la relecture d'un run deja termine. Pour filmer un
# vrai run, voir demo/README.md.

Output demo/distill.gif

Set Shell bash
Set FontSize 15
Set Width 1180
Set Height 680
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms
Set PlaybackSpeed 1.0

Hide
Type "clear" Enter
Show

# --- 1. La tache tient dans un fichier ---------------------------------------
Type "# Une description de tache, 24 exemples. C'est tout ce qu'on fournit."
Enter
Sleep 1500

Type "head -18 examples/email-triage/task.yaml"
Enter
Sleep 4000

Type "clear"
Enter

# --- 2. On verifie avant de depenser -----------------------------------------
Type "# On valide le schema et les exemples avant de depenser un centime."
Enter
Sleep 1200

Type "distill validate --task examples/email-triage/task.yaml"
Enter
Sleep 3000

Type "distill estimate --task examples/email-triage/task.yaml"
Enter
Sleep 3000

Type "clear"
Enter

# --- 3. Une seule commande ----------------------------------------------------
Type "# Puis une seule commande pour tout le pipeline."
Enter
Sleep 1200

Type "distill run --task examples/email-triage/task.yaml"
Enter
Sleep 8000

Type "clear"
Enter

# --- 4. Le tableau comparatif -------------------------------------------------
Type "# A la fin : le tableau. Meme test set, memes entrees, meme prompt."
Enter
Sleep 1500
Sleep 6000

Type "clear"
Enter

# --- 5. Le modele tourne en local --------------------------------------------
Type "# Et le modele repond en local, instantanement."
Enter
Sleep 1200

Type "ollama run distill-email-triage \"Bonjour, je n'ai toujours pas recu ma commande du 3.\""
Enter
Sleep 6000

Sleep 2000
