"""评测桥接（EmbodiedBench 等），与训练主流程解耦。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_embodiedbench_merged_config",
    "get_embodiedbench_evaluator_class",
    "run_embodiedbench_eval",
    "run_embodiedbench_from_omegaconf",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from openttl.eval import embodiedbench_bridge as _eb

        return getattr(_eb, name)
    raise AttributeError(name)
