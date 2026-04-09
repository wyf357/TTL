
from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from openttl.strategies.base import Strategy
from openttl.strategies.tent import TentStrategy
from openttl.strategies.eata import EATAStrategy
from openttl.strategies.tlm import TLMStrategy
from openttl.strategies.come import COMEStrategy

STRATEGY_REGISTRY = {
    "tent": TentStrategy,
    "tlm": TLMStrategy,
    "eata": EATAStrategy,
    "come": COMEStrategy,
}


def build_strategy(cfg: Any) -> Strategy:
    name = OmegaConf.select(cfg, "strategy.name")
    if name is None:
        raise ValueError("strategy.name missing in config")
    name = str(name).lower()
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](cfg.strategy)

