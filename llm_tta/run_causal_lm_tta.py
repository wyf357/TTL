#!/usr/bin/env python3
"""
大模型因果语言模型 TTA 运行入口（独立文件，不修改 OpenTTL 源码）。

复用 openttl.core.engine 中的 run_offline / run_online，与主配置
OpenTTL/configs/config.yaml 及 Hydra 命令行覆写完全兼容。

用法示例（在 TTL 或任意工作目录下执行）：

  # 离线/训练式 TTA（默认）
  python /root/TTL/llm_tta/run_causal_lm_tta.py model.pretrained_model_name_or_path=/path/to/model

  # 流式/在线 TTA 若干步
  python /root/TTL/llm_tta/run_causal_lm_tta.py tta_mode=online model=qwen35_2b \\
      model.pretrained_model_name_or_path=/path/to/model train.online_max_steps=50

  # 切策略与数据
  python /root/TTL/llm_tta/run_causal_lm_tta.py tta_mode=offline strategy=tent data=mmlu_tta
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# OpenTTL 安装路径：…/TTL/OpenTTL/src
_TTL = Path(__file__).resolve().parent.parent
_OPENTTL = _TTL / "OpenTTL"
if not _OPENTTL.is_dir():
    raise RuntimeError(f"未找到 OpenTTL 目录: {_OPENTTL}")
_OPENTTL_SRC = _OPENTTL / "src"
if str(_OPENTTL_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENTTL_SRC))

_CONFIGS = _OPENTTL / "configs"


@hydra.main(version_base=None, config_path=str(_CONFIGS), config_name="config")
def main(cfg: DictConfig) -> None:
    from openttl.core.engine import run_offline, run_online

    mode = str(OmegaConf.select(cfg, "tta_mode") or "offline").lower().strip()
    if mode in ("online", "stream", "streaming"):
        run_online(cfg)
    elif mode in ("offline", "train", "training", ""):
        run_offline(cfg)
    else:
        raise ValueError(
            f"不支持的 tta_mode={mode!r}，请使用 offline 或 online（或 stream / streaming）"
        )


if __name__ == "__main__":
    main()
