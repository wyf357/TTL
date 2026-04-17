"""EmbodiedBench 官方评测桥接入口（需已安装 ``embodiedbench`` 包与上游仿真环境）。"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="eval_embodiedbench")
def main(cfg: DictConfig) -> None:
    from openttl.eval.embodiedbench_bridge import run_embodiedbench_from_omegaconf

    run_embodiedbench_from_omegaconf(cfg)


if __name__ == "__main__":
    main()
