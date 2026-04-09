
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))


@hydra.main(version_base=None, config_path=str(_ROOT / "configs"), config_name="config")
def main(cfg: DictConfig) -> None:
    from openttl.core.engine import run_online

    run_online(cfg)


if __name__ == "__main__":
    main()
