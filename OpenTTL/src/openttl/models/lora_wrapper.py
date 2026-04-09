
from __future__ import annotations

from typing import Any

from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel


def peft_lora_config_from_cfg(peft_cfg: Any) -> LoraConfig:
    return LoraConfig(
        r=int(peft_cfg.r),
        lora_alpha=int(peft_cfg.lora_alpha),
        lora_dropout=float(peft_cfg.lora_dropout),
        bias=str(peft_cfg.bias),
        target_modules=list(peft_cfg.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def inject_lora(model: PreTrainedModel, peft_cfg: Any) -> PreTrainedModel:
    if not bool(getattr(peft_cfg, "enabled", True)):
        return model
    lcfg = peft_lora_config_from_cfg(peft_cfg)
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    return model


def save_adapter(model: PreTrainedModel, path: str) -> None:
    model.save_pretrained(path)


def merge_lora_inplace(model: PreTrainedModel) -> PreTrainedModel:
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model
