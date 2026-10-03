"""Modelfile et enregistrement du modele dans Ollama.

Le Modelfile fige le prompt systeme utilise a l'entrainement. C'est ce qui fait
que `ollama run distill-email-triage` donne le meme resultat que l'evaluation :
l'utilisateur tape son email, rien d'autre, et obtient le JSON.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..config import TaskConfig
from ..errors import ExportError
from ..logging_utils import get_logger
from ..prompts import system_prompt

MODELFILE_TEMPLATE = """# Genere par Distill - https://github.com/abdelilah-ourti/distill
# Tache : {task}
FROM {gguf}

SYSTEM \"\"\"{system}\"\"\"

# Decodage deterministe : la tache a une seule bonne reponse.
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_predict {num_predict}
PARAMETER num_ctx {num_ctx}
"""


def write_modelfile(
    cfg: TaskConfig,
    gguf_path: Path | str,
    destination: Path | str,
    *,
    num_predict: int = 256,
) -> Path:
    """Ecrit le Modelfile a cote du GGUF, avec un chemin relatif.

    Relatif volontairement : le dossier d'export reste deplacable et
    partageable tel quel, y compris sur Hugging Face.
    """
    gguf = Path(gguf_path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        reference = gguf.relative_to(target.parent)
    except ValueError:
        reference = gguf

    content = MODELFILE_TEMPLATE.format(
        task=cfg.name,
        gguf=f"./{reference.as_posix()}",
        system=system_prompt(cfg).replace('"""', "'''"),
        num_predict=num_predict,
        num_ctx=cfg.student.max_seq_length,
    )
    target.write_text(content, encoding="utf-8")
    get_logger().info(f"Modelfile ecrit : {target}")
    return target


def ollama_available() -> bool:
    return shutil.which("ollama") is not None


def create_model(model_name: str, modelfile: Path | str) -> bool:
    """Enregistre le modele dans Ollama. Retourne False si Ollama n'est pas la.

    Un echec ici ne doit pas faire echouer le run : les artefacts GGUF et le
    Modelfile restent utilisables, l'utilisateur lancera la commande lui-meme.
    """
    log = get_logger()
    path = Path(modelfile)
    if not ollama_available():
        log.warning(
            "[yellow]Ollama introuvable[/] : installe-le depuis https://ollama.com puis lance\n"
            f"  ollama create {model_name} -f {path}"
        )
        return False

    log.info(f"ollama create {model_name}")
    process = subprocess.run(
        ["ollama", "create", model_name, "-f", str(path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(path.parent),
    )
    if process.returncode != 0:
        tail = (process.stderr or process.stdout or "").strip().splitlines()[-10:]
        raise ExportError("ollama create a echoue :\n" + "\n".join(tail))
    log.info(f"[bold green]modele pret[/] : ollama run {model_name}")
    return True
