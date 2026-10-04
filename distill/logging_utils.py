"""Observabilite : console lisible pour l'humain, JSONL pour la machine.

Chaque evenement part aux deux endroits. La console sert la demo, le JSONL
sert le rapport HTML et le debug d'un run qui a tourne pendant deux heures.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

#: sortie destinee a l'utilisateur (tableaux, resultats) -> stdout
console = Console()
#: journal technique -> stderr, pour pouvoir rediriger l'un sans l'autre
log_console = Console(stderr=True)
_lock = threading.Lock()
_event_file: Path | None = None


def setup_logging(events_path: Path | None = None, *, verbose: bool = False) -> logging.Logger:
    """Branche la console Rich et, si demande, le journal JSONL du run."""
    global _event_file
    _event_file = events_path
    if events_path is not None:
        events_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("distill")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler = RichHandler(
        console=log_console,
        show_path=False,
        show_time=True,
        omit_repeated_times=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("distill")


def log_event(stage: str, event: str, **fields: Any) -> None:
    """Ecrit une ligne structuree dans events.jsonl (sans toucher la console)."""
    if _event_file is None:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": stage,
        "event": event,
        **fields,
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _lock, _event_file.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_events(events_path: Path) -> list[dict[str, Any]]:
    """Relit le journal d'un run, pour le rapport ou un post-mortem."""
    if not events_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


@contextmanager
def step(stage: str, title: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Encadre une etape : log d'entree, chronometre, log de sortie meme en cas d'echec.

    Le dict cede permet a l'etape d'ajouter des champs au log de fin
    (nombre d'exemples produits, score obtenu, ...).
    """
    logger = get_logger()
    logger.info(f"[bold cyan]{stage}[/] {title}")
    log_event(stage, "start", title=title, **fields)
    extra: dict[str, Any] = {}
    started = time.monotonic()
    try:
        yield extra
    except Exception as exc:
        elapsed = time.monotonic() - started
        logger.error(f"[bold red]{stage}[/] echec apres {elapsed:.1f}s : {exc}")
        log_event(stage, "error", title=title, error=str(exc), elapsed_s=round(elapsed, 2))
        raise
    elapsed = time.monotonic() - started
    detail = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info(f"[bold green]{stage}[/] termine en {elapsed:.1f}s {detail}")
    log_event(stage, "end", title=title, elapsed_s=round(elapsed, 2), **extra)


def die(message: str, code: int = 1) -> None:
    """Sortie d'erreur propre pour la CLI.

    Le message est echappe : il contient souvent des crochets - un conseil
    d'installation comme 'distill-slm[train]', un chemin, une liste - que Rich
    prendrait pour du balisage et avalerait silencieusement.
    """
    console.print(f"[bold red]Erreur[/] {escape(message)}")
    sys.exit(code)
