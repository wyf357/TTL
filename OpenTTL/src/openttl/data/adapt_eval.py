
from __future__ import annotations

from typing import Any, Dict, List

from datasets import Dataset, load_dataset


def _dummy_texts(n: int) -> Dataset:
    lines = [
        "The quick brown fox jumps over the lazy dog. " * 2,
        "Machine learning adapts models at test time. " * 2,
        "OpenTTL uses PEFT LoRA for efficient updates. " * 2,
    ]
    texts = [lines[i % len(lines)] for i in range(n)]
    return Dataset.from_dict({"text": texts})


def load_raw_dataset(data_cfg: Any) -> Dataset:
    src = str(getattr(data_cfg, "source", "huggingface"))
    if src == "dummy":
        n = int(getattr(data_cfg, "max_samples", 32) or 32)
        return _dummy_texts(n)
    if src == "mmlu":
        from openttl.data.mmlu import load_mmlu_unlabeled_raw

        return load_mmlu_unlabeled_raw(data_cfg)
    path = getattr(data_cfg, "hf_path", None)
    if not path:
        raise ValueError("data.hf_path required when source != dummy")
    name = getattr(data_cfg, "hf_name", None)
    split = str(getattr(data_cfg, "split", "train"))
    ds = load_dataset(path, name, split=split)
    max_s = getattr(data_cfg, "max_samples", None)
    if max_s is not None:
        ds = ds.select(range(min(int(max_s), len(ds))))
    return ds


def tokenize_for_clm(ds: Dataset, tokenizer: Any, data_cfg: Any) -> Dataset:
    col = str(getattr(data_cfg, "text_column", "text"))
    max_len = int(getattr(data_cfg, "max_length", 512))
    use_chat = bool(getattr(data_cfg, "use_chat_template", False))

    def _tok(batch: Dict[str, List]) -> Dict[str, List]:
        texts = list(batch[col])
        if use_chat and hasattr(tokenizer, "apply_chat_template"):
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for t in texts
            ]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        enc["labels"] = [list(x) for x in enc["input_ids"]]
        return enc

    return ds.map(_tok, batched=True, remove_columns=[c for c in ds.column_names])


def build_train_dataset(tokenizer: Any, data_cfg: Any) -> Dataset:
    raw = load_raw_dataset(data_cfg)
    col = str(getattr(data_cfg, "text_column", "text"))
    if col not in raw.column_names:
        raise ValueError(f"Column {col} not in {raw.column_names}")
    tok_ds = tokenize_for_clm(raw, tokenizer, data_cfg)
    keep = [c for c in ("input_ids", "attention_mask", "labels") if c in tok_ds.column_names]
    tok_ds.set_format(type="torch", columns=keep)
    return tok_ds
