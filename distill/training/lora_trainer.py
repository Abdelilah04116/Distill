"""F4 - Entrainement LoRA / QLoRA sur un seul GPU.

Deux backends pour la meme recette : Unsloth quand il est installe (plus rapide
et moins gourmand en VRAM), sinon Transformers + PEFT + TRL. Le resultat est le
meme adaptateur LoRA, donc l'export et l'evaluation ne savent pas lequel a servi.

Les hyperparametres par defaut sont choisis pour marcher sans reglage sur une
tache etroite a sortie structuree : r=16, lr=2e-4, 2 epoques. Ils ne sont pas
optimaux partout, ils sont raisonnables partout.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import TaskConfig
from ..data import Example
from ..errors import TrainingError
from ..logging_utils import get_logger, log_event
from ..paths import RunPaths
from ..prompts import chat_messages, target_text

# Projections couvertes par LoRA : vaut pour Qwen 2.5 comme pour Llama 3.2.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainingResult:
    adapter_dir: Path
    backend: str
    steps: int = 0
    final_loss: float = 0.0
    duration_s: float = 0.0
    n_examples: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_dir": str(self.adapter_dir),
            "backend": self.backend,
            "steps": self.steps,
            "final_loss": round(self.final_loss, 4),
            "duration_s": round(self.duration_s, 1),
            "n_examples": self.n_examples,
        }


def build_dataset(cfg: TaskConfig, examples: Sequence[Example], tokenizer: Any) -> Any:
    """Serialise les exemples dans le gabarit de chat du modele eleve.

    On passe par apply_chat_template plutot que par un format maison : le
    modele a ete pre-entraine avec ces balises, s'en ecarter coute des points
    pour rien.
    """
    from datasets import Dataset

    rows = []
    for ex in examples:
        conversation = [
            *chat_messages(cfg, ex.input),
            {"role": "assistant", "content": target_text(ex.output)},
        ]
        rows.append({"text": tokenizer.apply_chat_template(conversation, tokenize=False)})
    return Dataset.from_list(rows)


def _load_unsloth(cfg: TaskConfig) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.student.model,
        max_seq_length=cfg.student.max_seq_length,
        load_in_4bit=cfg.student.load_in_4bit,
        dtype=None if cfg.student.dtype == "auto" else cfg.student.dtype,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, tokenizer


def _load_transformers(cfg: TaskConfig) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs: dict[str, Any] = {"device_map": "auto", "torch_dtype": "auto"}
    if cfg.student.dtype != "auto":
        load_kwargs["torch_dtype"] = getattr(torch, cfg.student.dtype)
    if cfg.student.load_in_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.student.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.student.model, **load_kwargs)
    if cfg.student.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=cfg.training.lora_r,
        lora_alpha=cfg.training.lora_alpha,
        lora_dropout=cfg.training.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model, tokenizer


def _loss_callback(loss_file: Path) -> Any:
    """Ecrit chaque log d'entrainement en JSONL, pour le suivi en temps reel."""
    from transformers import TrainerCallback

    class LossRecorder(TrainerCallback):  # type: ignore[misc]
        def __init__(self) -> None:
            self.history: list[dict[str, Any]] = []
            loss_file.parent.mkdir(parents=True, exist_ok=True)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not logs or "loss" not in logs:
                return
            record = {"step": int(state.global_step), **{k: v for k, v in logs.items()}}
            self.history.append(record)
            with loss_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            get_logger().info(
                f"  step {record['step']:>4} | loss {logs['loss']:.4f}"
                + (f" | lr {logs['learning_rate']:.2e}" if "learning_rate" in logs else "")
            )

    return LossRecorder()


def train_lora(
    cfg: TaskConfig,
    examples: Sequence[Example],
    paths: RunPaths,
    *,
    output_dir: Path | None = None,
) -> TrainingResult:
    """Entraine l'adaptateur LoRA et le sauvegarde. Reprend un checkpoint s'il existe."""
    log = get_logger()
    if not examples:
        raise TrainingError("aucun exemple d'entrainement : le Critique a tout rejete")

    adapter_dir = Path(output_dir) if output_dir else paths.adapter_dir
    checkpoints_dir = adapter_dir.parent / "checkpoints"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    backend = "unsloth"
    try:
        if not cfg.training.use_unsloth:
            raise ImportError("desactive dans task.yaml")
        model, tokenizer = _load_unsloth(cfg)
    except ImportError as exc:
        log.info(f"Unsloth indisponible ({exc}), repli sur Transformers + PEFT")
        backend = "trl"
        try:
            model, tokenizer = _load_transformers(cfg)
        except ImportError as inner:
            raise TrainingError(
                "dependances d'entrainement absentes : pip install 'distill-slm[train]'"
            ) from inner

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise TrainingError("trl est requis : pip install 'distill-slm[train]'") from exc

    dataset = build_dataset(cfg, examples, tokenizer)
    recorder = _loss_callback(paths.loss_curve)

    sft_config = SFTConfig(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum,
        num_train_epochs=cfg.training.epochs,
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        max_steps=cfg.training.max_steps or -1,
        logging_steps=10,
        save_steps=cfg.training.save_steps,
        save_total_limit=2,
        seed=cfg.seed,
        data_seed=cfg.seed,
        report_to=[],
        max_seq_length=cfg.student.max_seq_length,
        dataset_text_field="text",
        optim="adamw_8bit" if cfg.student.load_in_4bit else "adamw_torch",
        lr_scheduler_type="linear",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=sft_config,
        callbacks=[recorder],
    )

    # Reprise apres interruption : on ne recommence pas deux heures de calcul
    # parce que la session SSH est tombee.
    resume = bool(cfg.training.resume_from_checkpoint and any(checkpoints_dir.glob("checkpoint-*")))
    if resume:
        log.info(f"reprise depuis le dernier checkpoint de {checkpoints_dir}")

    log.info(f"entrainement {backend} : {len(examples)} exemples, {cfg.training.epochs} epoques")
    started = time.monotonic()
    stats = trainer.train(resume_from_checkpoint=resume or None)
    duration = time.monotonic() - started

    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    result = TrainingResult(
        adapter_dir=adapter_dir,
        backend=backend,
        steps=int(getattr(stats, "global_step", 0) or 0),
        final_loss=float(getattr(stats, "training_loss", 0.0) or 0.0),
        duration_s=duration,
        n_examples=len(examples),
        history=recorder.history,
    )
    log.info(f"adaptateur sauvegarde dans {adapter_dir} (loss finale {result.final_loss:.4f})")
    log_event("entrainement", "done", **result.as_dict())
    return result
