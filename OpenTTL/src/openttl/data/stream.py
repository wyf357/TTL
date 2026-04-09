
from __future__ import annotations

from typing import Any, Dict, Iterator, List

import torch


def iter_hf_dataset(ds: Any, shuffle: bool = False, seed: int = 0) -> Iterator[Dict[str, Any]]:
    import random

    n = len(ds)
    order = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
    for i in order:
        yield {k: v for k, v in ds[int(i)].items()}


def collate_pad(batch: List[Dict[str, Any]], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    input_ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    attn = [torch.tensor(x["attention_mask"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    inp = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    am = torch.nn.utils.rnn.pad_sequence(attn, batch_first=True, padding_value=0)
    lab = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": inp, "attention_mask": am, "labels": lab}


def batched_stream(
    row_iter: Iterator[Dict[str, Any]],
    batch_size: int,
    pad_token_id: int = 0,
) -> Iterator[Dict[str, torch.Tensor]]:
    buf: List[Dict[str, Any]] = []
    for row in row_iter:
        buf.append(row)
        if len(buf) >= batch_size:
            yield collate_pad(buf, pad_token_id=pad_token_id)
            buf = []
    if buf:
        yield collate_pad(buf, pad_token_id=pad_token_id)
