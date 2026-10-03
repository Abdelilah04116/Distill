"""Conversion GGUF via llama.cpp.

Deux appels successifs : conversion du modele fusionne en GGUF 16 bits, puis
quantification vers les formats demandes (Q4_K_M pour la vitesse, Q8_0 pour la
qualite). On passe par les scripts de llama.cpp plutot que de reimplementer
quoi que ce soit - le format evolue vite, l'outil de reference suit.

llama.cpp est cherche dans cet ordre : export.llama_cpp_dir du task.yaml, la
variable LLAMA_CPP_DIR, puis les binaires du PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..errors import ExportError
from ..logging_utils import get_logger

CONVERT_SCRIPTS = ("convert_hf_to_gguf.py", "convert-hf-to-gguf.py")
QUANTIZE_BINARIES = ("llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe")


@dataclass
class GGUFArtifacts:
    f16: Path
    quantized: dict[str, Path]

    def best(self) -> Path:
        """Le fichier a servir par defaut : le plus petit disponible."""
        for quant in ("Q4_K_M", "Q5_K_M", "Q8_0"):
            if quant in self.quantized:
                return self.quantized[quant]
        return next(iter(self.quantized.values()), self.f16)


def find_llama_cpp(explicit: Path | None = None) -> Path | None:
    """Localise le depot llama.cpp, ou None s'il n'est pas la."""
    candidates = [
        explicit,
        Path(os.environ["LLAMA_CPP_DIR"]) if os.environ.get("LLAMA_CPP_DIR") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.is_dir():
            return candidate
    return None


def _convert_script(llama_cpp: Path | None) -> Path | None:
    if llama_cpp is None:
        return None
    for name in CONVERT_SCRIPTS:
        script = llama_cpp / name
        if script.is_file():
            return script
    return None


def _quantize_binary(llama_cpp: Path | None) -> str | None:
    search_dirs = []
    if llama_cpp is not None:
        search_dirs += [llama_cpp, llama_cpp / "build" / "bin", llama_cpp / "bin"]
    for directory in search_dirs:
        for name in QUANTIZE_BINARIES:
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    for name in QUANTIZE_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _run(command: Sequence[str | Path], step: str) -> None:
    log = get_logger()
    log.info(f"  $ {' '.join(str(c) for c in command[:3])} ...")
    process = subprocess.run(
        [str(c) for c in command],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        tail = (process.stderr or process.stdout or "").strip().splitlines()[-15:]
        raise ExportError(f"{step} a echoue (code {process.returncode}) :\n" + "\n".join(tail))


def convert_to_gguf(
    merged_dir: Path | str,
    output_dir: Path | str,
    model_name: str,
    *,
    quantizations: Sequence[str] = ("Q4_K_M", "Q8_0"),
    llama_cpp_dir: Path | None = None,
) -> GGUFArtifacts:
    """Produit le GGUF 16 bits puis ses versions quantifiees."""
    log = get_logger()
    merged = Path(merged_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not merged.is_dir():
        raise ExportError(f"modele fusionne introuvable : {merged}")

    llama_cpp = find_llama_cpp(llama_cpp_dir)
    script = _convert_script(llama_cpp)
    if script is None:
        raise ExportError(
            "llama.cpp introuvable. Clone-le puis indique son chemin :\n"
            "  git clone https://github.com/ggerganov/llama.cpp\n"
            "  export LLAMA_CPP_DIR=$PWD/llama.cpp\n"
            "ou renseigne export.llama_cpp_dir dans task.yaml."
        )

    f16 = out / f"{model_name}-f16.gguf"
    log.info(f"conversion GGUF f16 : {f16.name}")
    _run(
        [sys.executable, script, str(merged), "--outfile", str(f16), "--outtype", "f16"],
        "la conversion GGUF",
    )

    produced: dict[str, Path] = {}
    quantize = _quantize_binary(llama_cpp)
    if quantize is None:
        log.warning(
            "[yellow]llama-quantize introuvable[/] : seul le GGUF f16 est produit. "
            "Compile llama.cpp (cmake -B build && cmake --build build) pour obtenir Q4_K_M."
        )
        return GGUFArtifacts(f16=f16, quantized={})

    for quant in quantizations:
        target = out / f"{model_name}-{quant}.gguf"
        log.info(f"quantification {quant} : {target.name}")
        _run([quantize, str(f16), str(target), quant], f"la quantification {quant}")
        produced[quant] = target
        log.info(f"  {target.name} : {target.stat().st_size / 1e9:.2f} Go")

    return GGUFArtifacts(f16=f16, quantized=produced)
