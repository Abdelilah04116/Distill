"""Fusion de l'adaptateur LoRA dans les poids de base.

Etape obligatoire avant la conversion GGUF : llama.cpp ne sait pas charger un
adaptateur PEFT a la volee, il lui faut un modele complet. La fusion se fait
en 16 bits sur CPU si besoin, ce qui est lent mais ne demande pas de VRAM
supplementaire au moment ou le GPU vient de servir a l'entrainement.
"""

from __future__ import annotations

from pathlib import Path

from ..config import TaskConfig
from ..errors import ExportError
from ..logging_utils import get_logger


def merge_lora(cfg: TaskConfig, adapter_dir: Path | str, output_dir: Path | str) -> Path:
    """Fusionne l'adaptateur et ecrit un modele Hugging Face complet."""
    log = get_logger()
    adapter = Path(adapter_dir)
    out = Path(output_dir)
    if not adapter.is_dir():
        raise ExportError(f"adaptateur introuvable : {adapter}")
    out.mkdir(parents=True, exist_ok=True)

    log.info(f"fusion de l'adaptateur {adapter.name} dans {cfg.student.model}")
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ExportError(
            "la fusion LoRA demande transformers et peft : pip install 'distill-slm[train]'"
        ) from exc

    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            str(adapter),
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        merged = model.merge_and_unload()
        merged.save_pretrained(str(out), safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter))
        tokenizer.save_pretrained(str(out))
    except Exception as exc:  # remonte en erreur d'export lisible
        raise ExportError(f"echec de la fusion LoRA : {exc}") from exc

    log.info(f"modele fusionne ecrit dans {out}")
    return out
