"""Mini tableau de bord de suivi, dans le terminal.

Un run dure jusqu'a deux heures. Pendant ce temps, l'utilisateur veut savoir
trois choses : ou en est le pipeline, combien ca a deja coute, et si la loss
descend. Ce module lit les fichiers que le run ecrit de toute facon
(events.jsonl, loss.jsonl, state.json) et les rafraichit a l'ecran.

Volontairement sans serveur web : un tableau de bord qu'il faut deployer pour
surveiller un run local n'aurait pas sa place ici.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .logging_utils import console, read_events
from .paths import RunPaths

REFRESH_SECONDS = 2.0
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 48) -> str:
    """Courbe en une ligne. Suffit a voir si la loss descend ou stagne."""
    if len(values) < 2:
        return ""
    step = max(1, len(values) // width)
    sampled = values[::step][-width:]
    low, high = min(sampled), max(sampled)
    span = (high - low) or 1.0
    return "".join(
        SPARK[min(len(SPARK) - 1, int((v - low) / span * (len(SPARK) - 1)))] for v in sampled
    )


def _read_losses(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _steps_table(events: list[dict[str, Any]]) -> Table:
    table = Table(title="Etapes", header_style="dim", expand=True)
    table.add_column("Etape")
    table.add_column("Etat")
    table.add_column("Duree", justify="right")
    table.add_column("Detail", overflow="fold")

    started: dict[str, dict[str, Any]] = {}
    finished: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") == "start":
            started[event["stage"]] = event
        elif event.get("event") in {"end", "error"}:
            finished[event["stage"]] = event

    for stage, start in started.items():
        done = finished.get(stage)
        if done is None:
            state, style, duration = "en cours", "yellow", "-"
            detail = str(start.get("title", ""))
        elif done.get("event") == "error":
            state, style = "echec", "red"
            duration = f"{done.get('elapsed_s', 0):.0f}s"
            detail = str(done.get("error", ""))[:80]
        else:
            state, style = "termine", "green"
            duration = f"{done.get('elapsed_s', 0):.0f}s"
            detail = " ".join(
                f"{k}={v}"
                for k, v in done.items()
                if k not in {"ts", "stage", "event", "title", "elapsed_s"}
            )[:80]
        table.add_row(stage, f"[{style}]{state}[/]", duration, detail)
    return table


def _loss_panel(losses: list[dict[str, Any]]) -> Panel:
    values = [float(row["loss"]) for row in losses if "loss" in row]
    if len(values) < 2:
        return Panel("en attente des premiers pas d'entrainement", title="Loss", border_style="dim")
    trend = "en baisse" if values[-1] < values[0] else "stable ou en hausse"
    body = (
        f"[cyan]{sparkline(values)}[/]\n"
        f"pas {losses[-1].get('step', '?')} · depart {values[0]:.4f} · "
        f"actuel {values[-1]:.4f} · {trend}"
    )
    return Panel(body, title="Loss d'entrainement", border_style="cyan")


def _state_panel(state: dict[str, Any]) -> Panel:
    lines = [
        f"iteration        : {state.get('iteration', 0)}",
        f"exemples generes : {state.get('generated_total', 0)}",
        f"entrainement     : {state.get('train_size', 0)} exemples",
        f"test             : {state.get('test_size', 0)} exemples",
    ]
    if "quality_ratio" in state:
        met = state.get("target_met")
        lines.append(
            f"ratio / prof     : {state['quality_ratio']:.1%} "
            f"({'[green]cible atteinte[/]' if met else '[yellow]sous la cible[/]'})"
        )
    if state.get("stop_reason"):
        lines.append(f"arret            : {state['stop_reason']}")
    return Panel("\n".join(lines), title="Etat du run", border_style="blue")


def render(paths: RunPaths) -> Group:
    """Une image du run a l'instant t."""
    events = read_events(paths.events)
    losses = _read_losses(paths.loss_curve)
    state = {}
    if paths.state_file.is_file():
        try:
            state = json.loads(paths.state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {}
    return Group(_state_panel(state), _steps_table(events), _loss_panel(losses))


def watch(paths: RunPaths, *, refresh: float = REFRESH_SECONDS, once: bool = False) -> None:
    """Suit un run en direct jusqu'a sa fin ou jusqu'a Ctrl-C."""
    if once:
        console.print(render(paths))
        return
    console.print(f"suivi du run [bold]{paths.run_id}[/] — Ctrl-C pour quitter")
    with Live(render(paths), console=console, refresh_per_second=4) as live:
        try:
            while True:
                time.sleep(refresh)
                live.update(render(paths))
                events = read_events(paths.events)
                if any(e.get("stage") == "run" and e.get("event") == "end" for e in events):
                    live.update(render(paths))
                    break
        except KeyboardInterrupt:
            pass
    console.print(
        f"rapport : [cyan]{paths.report}[/]" if paths.report.is_file() else "run en cours"
    )
