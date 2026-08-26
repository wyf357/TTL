"""MANTA Runner: Online TTA runner for MANTA strategy.

MANTA manages its own gamma parameters (2K scalars) and does NOT require LoRA.
The runner simply calls strategy.compute_loss() which internally optimizes gamma
and applies modulation via forward hooks.

Reference: MANTA.md, E3TTARunner pattern
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from openttl.adapters.base import ModelAdapter
from openttl.inference.base import InferenceEngine
from openttl.strategies.base import Strategy

LOG = logging.getLogger(__name__)


def _batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move batch tensors to device."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


class MANTARunner:
    """Online TTA runner for MANTA strategy.
    
    Unlike OnlineTTARunner which manages LoRA adapters, MANTARunner:
    1. Uses the same model for both training and inference
    2. Gamma parameters are optimized in-place via hooks
    3. No LoRA export or SGLang sync needed
    4. Hooks automatically apply gamma modulation during forward pass
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
        """Initialize MANTA runner.
        
        Args:
            cfg: Hydra config
            model: The model (will have MANTA hooks registered)
            adapter: Model adapter
            inference: Inference engine
            device: Device to run on
            strategy: MANTA strategy (will be created if None)
        """
        from openttl.strategies import build_strategy
        from openttl.strategies.manta import MANTAStrategy
        
        self.cfg = cfg
        self.model = model
        self.adapter = adapter
        self.inference = inference
        self.device = device
        
        # Build or use provided strategy
        if strategy is None:
            strategy = build_strategy(cfg)
        
        if not isinstance(strategy, MANTAStrategy):
            raise ValueError(
                f"MANTARunner requires MANTAStrategy, got {type(strategy).__name__}"
            )
        
        self.strategy = strategy
        
        # Setup strategy (registers hooks, creates gamma params)
        self.strategy.setup(self.model, teacher_model=None)
        
        LOG.info("MANTARunner initialized with hooks registered")
    
    @property
    def tokenizer(self) -> Any:
        return self.adapter.tokenizer()
    
    @property
    def processor(self) -> Any:
        return getattr(self.adapter, "_processor", None)
    
    def update(self, batch: Dict[str, Any]) -> float:
        """Single MANTA TTA update step.
        
        This calls strategy.compute_loss() which:
        1. Sets visual/text masks for hooks
        2. Runs forward pass (hooks apply gamma modulation)
        3. Computes IADE loss from collected hidden states
        4. Optimizes gamma parameters
        5. Returns loss
        
        Args:
            batch: Input batch with input_ids, attention_mask, pixel_values, etc.
            
        Returns:
            Scalar loss value
        """
        # Move batch to device
        batch = _batch_to_device(batch, self.device)
        
        # Call strategy compute_loss
        # This internally:
        # - Sets masks for hooks
        # - Runs forward pass with gamma modulation
        # - Computes IADE loss
        # - Optimizes gamma
        loss = self.strategy.compute_loss(self.model, batch)
        
        # Extract scalar loss value
        loss_val = float(loss.detach().cpu())
        
        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        LOG.debug(f"MANTA update: loss={loss_val:.6f}")
        
        return loss_val
    
    def enabled(self) -> bool:
        """Check if online TTA is enabled."""
        from omegaconf import OmegaConf
        return bool(OmegaConf.select(self.cfg, "online.enabled") or False)
