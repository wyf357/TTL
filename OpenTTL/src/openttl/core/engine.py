
from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import OmegaConf
from transformers import DataCollatorForLanguageModeling, TrainingArguments

from openttl.core.trainer import TTATrainer
from openttl.data.adapt_eval import build_train_dataset
from openttl.data.stream import batched_stream, iter_hf_dataset
from openttl.models.loader import load_causal_lm, load_causal_lm_eval, load_tokenizer
from openttl.models.lora_wrapper import inject_lora, save_adapter
from openttl.strategies import build_strategy
from openttl.utils.logging import LOG, log_trainable_summary, setup_logging


def _training_arguments(train_cfg: Any) -> TrainingArguments:
    d = OmegaConf.to_container(train_cfg, resolve=True)
    if not isinstance(d, dict):
        raise TypeError("train config must map to dict")
    d.pop("online_max_steps", None)
    return TrainingArguments(**d)


def _maybe_load_teacher(cfg: Any, device: torch.device) -> Optional[torch.nn.Module]:
    path = OmegaConf.select(cfg, "strategy.teacher_model_path")
    use_t = bool(OmegaConf.select(cfg, "strategy.use_teacher_for_selection") or False)
    if not use_t or not path:
        return None
    tcfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    tcfg.pretrained_model_name_or_path = path
    teacher = load_causal_lm_eval(tcfg)
    teacher.to(device)
    return teacher


def run_offline(cfg: Any) -> None:
    setup_logging()
    tokenizer = load_tokenizer(cfg.model)
    model = load_causal_lm(cfg.model)
    model = inject_lora(model, cfg.model.peft)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_trainable_summary(model)

    teacher = _maybe_load_teacher(cfg, device)
    strategy = build_strategy(cfg)
    train_ds = build_train_dataset(tokenizer, cfg.data)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    args = _training_arguments(cfg.train)

    common = dict(
        strategy=strategy,
        teacher_model=teacher,
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    )
    try:
        trainer = TTATrainer(**common, processing_class=tokenizer)
    except TypeError:
        trainer = TTATrainer(**common, tokenizer=tokenizer)
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    LOG.info("Saved checkpoint to %s", cfg.train.output_dir)


def run_online(cfg: Any) -> None:
    setup_logging()
    from accelerate import Accelerator

    acc = Accelerator(
        gradient_accumulation_steps=int(getattr(cfg.train, "gradient_accumulation_steps", 1)),
        mixed_precision="bf16" if bool(getattr(cfg.train, "bf16", False)) else None,
    )
    device = acc.device
    tokenizer = load_tokenizer(cfg.model)
    model = load_causal_lm(cfg.model)
    model = inject_lora(model, cfg.model.peft)
    model.to(device)
    log_trainable_summary(model)

    teacher = _maybe_load_teacher(cfg, device)
    strategy = build_strategy(cfg)
    strategy.setup(model, teacher)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg.train.learning_rate),
        weight_decay=float(getattr(cfg.train, "weight_decay", 0.0)),
    )
    model, opt = acc.prepare(model, opt)

    train_ds = build_train_dataset(tokenizer, cfg.data)
    pad_id = tokenizer.pad_token_id or 0
    bs = int(cfg.train.per_device_train_batch_size)
    stream = batched_stream(
        iter_hf_dataset(train_ds, shuffle=True, seed=int(getattr(cfg.train, "seed", 0))),
        batch_size=bs,
        pad_token_id=int(pad_id),
    )

    max_steps = int(OmegaConf.select(cfg, "train.online_max_steps") or 100)
    model.train()
    step = 0
    for batch in stream:
        if step >= max_steps:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with acc.accumulate(model):
            loss = strategy.compute_loss(model, batch)
            acc.backward(loss)
            opt.step()
            opt.zero_grad()
        if acc.is_main_process:
            LOG.info("online step %s loss=%s", step, float(loss.detach()))
        step += 1

    out = str(cfg.train.output_dir)
    acc.wait_for_everyone()
    if acc.is_main_process:
        save_adapter(acc.unwrap_model(model), out)
        LOG.info("Online TTA finished, adapter saved to %s", out)
