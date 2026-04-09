
from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger("openttl")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def log_trainable_summary(model: Any) -> None:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG.info("Parameters trainable: %s / %s (%.4f%%)", train, total, 100.0 * train / max(total, 1))
