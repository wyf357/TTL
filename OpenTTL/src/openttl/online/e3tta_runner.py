from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional, Union

import torch
from omegaconf import OmegaConf
from torch import nn

from openttl.adapters.base import ModelAdapter
from openttl.inference.base import InferenceEngine
from openttl.strategies import build_strategy
from openttl.strategies.base import Strategy
from openttl.strategies.e3tta import E3TTAStrategy

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    LOG.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[E3-TTA] %(message)s')
    handler.setFormatter(formatter)
    LOG.addHandler(handler)


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class E3TTARunner:
    """E3-TTA专用Runner：只更新门控参数，不导出LoRA。
    
    E3-TTA特点：
    1. 只更新门控参数（每层1个向量，总计<100K参数）
    2. 模型主干完全冻结
    3. 不需要同步权重到推理引擎（门控只在训练时生效）
    4. 损失函数只包含熵稳定项和正则项
    """

    def __init__(
        self,
        cfg: Any,
        *,
        model: nn.Module,
        adapter: ModelAdapter,
        inference: InferenceEngine,
        device: torch.device,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.inference = inference
        self.device = device
        
        # 构建或传入策略
        if strategy is None:
            strategy = build_strategy(cfg)
        if not isinstance(strategy, E3TTAStrategy):
            raise ValueError(f"E3TTARunner requires E3TTAStrategy, got {type(strategy)}")
        self.strategy = strategy

        # E3-TTA配置
        oc = OmegaConf.select(cfg, "online") or OmegaConf.create({})
        self._sync_every = int(OmegaConf.select(oc, "sync_every_n_updates") or 1)
        
        # 注意：E3-TTA不需要memory saver模式，因为门控参数很少
        self._enable_memory_saver = False

        # 设置策略（会冻结模型主干，初始化门控）
        self.strategy.setup(self.model, teacher_model=None)
        
        # 使用策略自己的优化器（只优化门控参数）
        self._optimizer = self.strategy.gate_optimizer
        if self._optimizer is None:
            raise ValueError("E3TTAStrategy gate_optimizer not initialized")
        
        self._update_idx = 0
        
        LOG.info("E3-TTA Runner initialized: %d trainable parameters", 
                 sum(p.numel() for p in self.strategy.get_trainable_params()))

    @property
    def tokenizer(self) -> Any:
        return self.adapter.tokenizer()

    @property
    def processor(self) -> Any:
        return getattr(self.adapter, "_processor", None)

    def update(self, batch: Dict[str, Any]) -> float:
        """单步E3-TTA更新；返回标量loss。
        
        与LoRA-based TTA不同，E3-TTA：
        1. 不需要导出权重
        2. 不需要同步到推理引擎
        3. 门控参数在策略内部管理
        """
        import traceback
        
        self.model.train()
        for param in self.model.parameters():
            if param.requires_grad:
                pass
        self._optimizer.zero_grad(set_to_none=True)

        loss_val: float = float("nan")
        train_ok = False
        
        try:
            LOG.debug("[E3-TTA DEBUG] Step 1: Moving batch to device %s", self.device)
            batch = _batch_to_device(batch, self.device)
            LOG.debug("[E3-TTA DEBUG] Step 1 OK: Batch keys = %s", list(batch.keys()))
            
            LOG.debug("[E3-TTA DEBUG] Step 2: Calling compute_loss...")
            result = self.strategy.compute_loss(self.model, batch)
            LOG.debug("[E3-TTA DEBUG] Step 2 OK: compute_loss returned type = %s", type(result))
            
            if isinstance(result, tuple):
                loss, outputs = result
                LOG.debug("[E3-TTA DEBUG] Step 2a: Unpacked tuple, loss shape = %s", loss.shape if hasattr(loss, 'shape') else type(loss))
            else:
                loss = result
                LOG.debug("[E3-TTA DEBUG] Step 2b: Single loss tensor, shape = %s", loss.shape if hasattr(loss, 'shape') else type(loss))
            
            LOG.debug("[E3-TTA DEBUG] Step 3: Calling loss.backward()...")
            loss.backward()
            LOG.debug("[E3-TTA DEBUG] Step 3 OK: backward completed")
            
            LOG.debug("[E3-TTA DEBUG] Step 4: Calling optimizer.step()...")
            self._optimizer.step()
            LOG.debug("[E3-TTA DEBUG] Step 4 OK: optimizer step completed")
            
            LOG.debug("[E3-TTA DEBUG] Step 5: Calling on_batch_end...")
            self.strategy.on_batch_end(None, batch, loss)
            LOG.debug("[E3-TTA DEBUG] Step 5 OK: on_batch_end completed")
            
            loss_val = float(loss.detach().cpu())
            train_ok = True
            
            LOG.debug("[E3-TTA DEBUG] Step 6: Cleanup, loss = %.4f", loss_val)
            del loss, batch
            
        except Exception as e:
            error_msg = f"E3-TTA update step failed: {str(e)}"
            LOG.warning(error_msg, exc_info=True)
            LOG.warning("[E3-TTA DEBUG] Full traceback:\n%s", traceback.format_exc())
            raise RuntimeError(error_msg) from e
            
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if train_ok:
            self._update_idx += 1
            if self._update_idx % self._sync_every == 0:
                LOG.debug("E3-TTA update %d completed, loss=%.4f", self._update_idx, loss_val)
        else:
            raise RuntimeError("E3-TTA update step failed.")

        return loss_val

    def reset_scene(self):
        """检测到新场景时重置场景缓存。"""
        self.strategy.reset_scene_cache()
        LOG.info("E3-TTA scene cache reset")

    def get_current_entropies(self) -> Optional[torch.Tensor]:
        """获取当前各层的熵值（用于监控）。"""
        return self.strategy.get_current_entropies()

    def enabled(self) -> bool:
        return bool(OmegaConf.select(self.cfg, "online.enabled") or False)

    def cleanup(self):
        """清理资源。"""
        self.strategy.cleanup()
