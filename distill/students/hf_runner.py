"""Inference de l'eleve via Transformers, avec ou sans adaptateur LoRA.

C'est le chemin utilise pendant la boucle d'evaluation, avant tout export : il
permet de mesurer l'eleve des la fin de l'entrainement, sans attendre la
quantification GGUF. Le meme objet sert a la baseline (modele brut, sans
adaptateur), ce qui garantit que la seule difference mesuree est le fine-tuning.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import TaskConfig
from ..prompts import chat_messages
from .base import Prediction, Predictor


class HFPredictor(Predictor):
    """Eleve local charge en memoire GPU (ou CPU, lentement)."""

    def __init__(
        self,
        cfg: TaskConfig,
        *,
        base_model: str | None = None,
        adapter_dir: Path | str | None = None,
        kind: str = "student",
        name: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depend de l'installation
            raise RuntimeError(
                "transformers et torch sont requis pour evaluer l'eleve localement "
                "(pip install 'distill-slm[train]')"
            ) from exc

        self.cfg = cfg
        self.kind = kind
        self.billing = "gpu"
        self.max_new_tokens = max_new_tokens
        self._torch = torch
        model_id = base_model or cfg.student.model
        self.name = name or (f"{model_id} + LoRA" if adapter_dir else f"{model_id} (brut)")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"device_map": "auto"}
        if cfg.student.dtype != "auto":
            load_kwargs["torch_dtype"] = getattr(torch, cfg.student.dtype)
        else:
            load_kwargs["torch_dtype"] = "auto"
        if cfg.student.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:  # noqa: BLE001 - bitsandbytes absent (Windows, CPU)
                pass

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if adapter_dir is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter_dir))
        self.model.eval()

    def _sync(self) -> None:
        if self._torch.cuda.is_available():
            self._torch.cuda.synchronize()

    def predict(self, text: str) -> Prediction:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            chat_messages(self.cfg, text),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        self._sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                # Decodage glouton : une evaluation doit etre reproductible.
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        self._sync()
        latency = time.perf_counter() - started

        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return Prediction(
            raw=raw.strip(),
            latency_s=latency,
            input_tokens=int(inputs["input_ids"].shape[1]),
            output_tokens=int(new_tokens.shape[0]),
        )

    def close(self) -> None:
        del self.model
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
