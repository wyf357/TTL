"""MANTA-Uni: Modality-Aware Normalization Tuning with Unified Entropy.

A minimal-parameter test-time adaptation algorithm that introduces 2K scalars
(for last K layers) to modulate RMSNorm, optimizing an Implicit Cross-Modal
Association Distribution Entropy (IADE) loss.

Reference: MANTA.md
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy
from openttl.strategies.tta_shared import tta_model_forward

STRATEGY_LOG = logging.getLogger(__name__)
if not STRATEGY_LOG.handlers:
    STRATEGY_LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[MANTA Strategy] %(message)s')
    handler.setFormatter(formatter)
    STRATEGY_LOG.addHandler(handler)


class MANTAStrategy(Strategy):
    """MANTA-Uni: Modality-aware RMSNorm modulation + IADE loss minimization.
    
    Core features:
    1. Only 2K learnable scalars (gamma_v, gamma_t) for last K layers
    2. Modality-aware RMSNorm modulation (visual vs text tokens)
    3. Implicit Cross-Modal Association Distribution Entropy (IADE) loss
    4. Architecture-agnostic: works with any early-fusion multimodal model
    
    Reference: MANTA.md
    """
    
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        
        # Configuration parameters
        self.num_adapt_layers = int(getattr(cfg, "num_adapt_layers", 4))
        self.lambda_global = float(getattr(cfg, "lambda_global", 1.0))
        self.lr = float(getattr(cfg, "lr", 1e-2))
        self.eps = float(getattr(cfg, "epsilon", 1e-9))
        self.adapt_steps = int(getattr(cfg, "adapt_steps", 1))
        
        # Components initialized in setup()
        self.gamma: Optional[nn.Parameter] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.hidden_size: Optional[int] = None
        self.num_layers: Optional[int] = None
        self.tau: Optional[float] = None
        self.model_dtype: Optional[torch.dtype] = None
        self.model_device: Optional[torch.device] = None
        
        # Hook handles for RMSNorm modulation
        self._hook_handles: list = []
        
        # Current batch info for hooks
        self._current_visual_mask: Optional[torch.Tensor] = None
        self._current_text_mask: Optional[torch.Tensor] = None
        self._current_layer_idx: int = 0
        self._stored_hidden_outputs: list = []  # For IADE loss computation
        self._stored_modulated_outputs: list = []  # Modulated outputs from RMSNorm hooks (for direct gradient flow)
        
    def setup(
        self,
        model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
    ) -> None:
        """Initialize MANTA parameters and register RMSNorm hooks."""
        super().setup(model, teacher_model)
        
        # Get model dimensions
        self.hidden_size = getattr(model.config, "hidden_size", None)
        if self.hidden_size is None:
            self.hidden_size = getattr(model.config, "d_model", 2048)
            
        self.num_layers = getattr(model.config, "num_hidden_layers", None)
        if self.num_layers is None:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                self.num_layers = len(model.model.layers)
            elif hasattr(model, 'layers'):
                self.num_layers = len(model.layers)
            else:
                self.num_layers = 24  # Default fallback
        
        # Temperature for softmax scaling
        self.tau = self.hidden_size ** 0.5
        
        # Get model dtype and device
        self.model_dtype = next(model.parameters()).dtype
        self.model_device = next(model.parameters()).device
        
        STRATEGY_LOG.info(
            f"MANTA Setup: hidden_size={self.hidden_size}, num_layers={self.num_layers}, "
            f"adapt_layers={self.num_adapt_layers}, tau={self.tau:.2f}"
        )
        
        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Create learnable gamma parameters: [K, 2] for (visual, text)
        K = min(self.num_adapt_layers, self.num_layers)
        self.gamma = nn.Parameter(
            torch.ones(K, 2, device=self.model_device, dtype=self.model_dtype)
        )
        self.gamma.requires_grad = True
        
        # Create optimizer for gamma only
        self.optimizer = torch.optim.Adam([self.gamma], lr=self.lr)
        
        STRATEGY_LOG.info(
            f"MANTA initialized: {K} layers, {2*K} parameters, "
            f"lr={self.lr}, lambda_global={self.lambda_global}"
        )
        
        # Register forward hooks on RMSNorm modules
        self._register_rmsnorm_hooks(model)
    
    def _get_layers(self, model: nn.Module):
        """Get decoder layers from model."""
        # Qwen3_5ForConditionalGeneration: model.model.language_model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'language_model'):
            if hasattr(model.model.language_model, 'layers'):
                return model.model.language_model.layers
        
        # Qwen3_5ForCausalLM: model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        
        # Qwen3_5Model (unwrapped): model.layers
        if hasattr(model, 'layers'):
            return model.layers
        
        raise ValueError(f"Cannot find model layers. Model type: {type(model)}")
    
    def _compute_num_image_tokens(self, inputs: Dict[str, Any]) -> int:
        """Compute total number of image tokens from image_grid_thw.
        
        For Qwen3.5-VL: num_tokens = sum(grid.prod() // merge_size**2)
        """
        image_grid_thw = inputs.get("image_grid_thw")
        if image_grid_thw is None:
            STRATEGY_LOG.warning("image_grid_thw not found in inputs")
            return 0
        
        if image_grid_thw.numel() == 0:
            STRATEGY_LOG.warning("image_grid_thw is empty")
            return 0
        
        # Get merge_size from model config or use default
        # For Qwen3.5-VL, merge_size is typically 2
        merge_size = 2
        if hasattr(self, '_merge_size'):
            merge_size = self._merge_size
        
        merge_length = merge_size ** 2
        
        # image_grid_thw shape: (num_images, 3) [t, h, w]
        if image_grid_thw.dim() == 2 and image_grid_thw.shape[0] > 0:
            num_tokens = 0
            num_images = image_grid_thw.shape[0]
            
            for i in range(num_images):
                grid = image_grid_thw[i]  # [t, h, w]
                # grid.prod() = t * h * w
                tokens_for_image = int(grid.prod().item() // merge_length)
                num_tokens += tokens_for_image
                
                STRATEGY_LOG.debug(
                    f"Image {i}: grid={grid.tolist()}, prod={grid.prod().item()}, "
                    f"tokens={tokens_for_image}"
                )
            
            STRATEGY_LOG.info(
                f"Computed num_image_tokens={num_tokens} from {num_images} images "
                f"(merge_size={merge_size})"
            )
            return num_tokens
        
        STRATEGY_LOG.warning(
            f"image_grid_thw has unexpected shape: {image_grid_thw.shape}"
        )
        return 0
    
    def _register_rmsnorm_hooks(self, model: nn.Module):
        """Register forward hooks on input_layernorm modules for last K layers.
        
        The hooks apply modality-aware gamma scaling AFTER the original RMSNorm,
        avoiding double normalization and ensuring gamma is used during inference.
        """
        # Clear existing hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        
        layers = self._get_layers(model)
        L = self.num_layers
        K = self.gamma.shape[0]
        
        STRATEGY_LOG.info(f"Registering MANTA hooks on last {K} layers (indices {L-K} to {L-1})")
        
        # Register hooks only on last K layers
        for layer_idx in range(L - K, L):
            layer = layers[layer_idx]
            rmsnorm = layer.input_layernorm
            
            # Calculate k index for gamma lookup
            k = layer_idx - (L - K)
            
            def make_hook(rms_layer_idx, gamma_k):
                """Create closure to capture layer_idx and k."""
                def rmsnorm_hook(module, input, output):
                    """Apply modality-aware gamma scaling after original RMSNorm."""
                    if self._current_visual_mask is None or self._current_text_mask is None:
                        return output
                    
                    # output shape: (B, N, d)
                    # Get gamma values for this layer
                    gamma_v = self.gamma[gamma_k, 0]
                    gamma_t = self.gamma[gamma_k, 1]
                    
                    # Create scale tensor: (B, N, 1)
                    B, N = output.shape[:2]
                    scale = torch.ones(B, N, 1, device=output.device, dtype=output.dtype)
                    
                    # Apply visual and text gamma using masks
                    v_mask = self._current_visual_mask.unsqueeze(-1)
                    t_mask = self._current_text_mask.unsqueeze(-1)
                    scale = scale * (v_mask * gamma_v + t_mask * gamma_t)
                    
                    # Apply scaling: output * scale
                    modulated_output = output * scale
                    
                    # CRITICAL: Store modulated output for IADE loss (maintains gradient flow to gamma)
                    self._stored_modulated_outputs.append((gamma_k, modulated_output))
                    
                    return modulated_output
                
                return rmsnorm_hook
            
            handle = rmsnorm.register_forward_hook(make_hook(layer_idx, k))
            self._hook_handles.append(handle)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
    
    def _compute_iade_loss(
        self,
        h_v: torch.Tensor,
        h_t: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Compute IADE (Implicit Cross-Modal Association Distribution Entropy) loss.
        
        Args:
            h_v: Visual token hidden states (N_v, d)
            h_t: Text token hidden states (N_t, d)
            layer_idx: Layer index for logging
            
        Returns:
            IADE loss scalar
        """
        # Validate inputs
        if h_v.numel() == 0 or h_t.numel() == 0:
            STRATEGY_LOG.warning(
                f"Layer {layer_idx}: Empty visual or text tokens! "
                f"h_v.shape={h_v.shape}, h_t.shape={h_t.shape}"
            )
            return torch.tensor(0.0, device=h_v.device, dtype=h_v.dtype)
        
        # Step 1: Build implicit cross-modal association matrix
        # Normalize inputs for numerical stability
        h_v_norm = F.normalize(h_v, p=2, dim=-1)
        h_t_norm = F.normalize(h_t, p=2, dim=-1)
        
        S = torch.matmul(h_t_norm, h_v_norm.t())  # (N_t, N_v) - now in [-1, 1] range
        
        # Check for NaN in S
        if torch.isnan(S).any():
            STRATEGY_LOG.error(f"Layer {layer_idx}: NaN detected in similarity matrix S")
            return torch.tensor(0.0, device=h_v.device, dtype=h_v.dtype)
        
        # Step 2: Row-level softmax (text -> visual implicit attention)
        P = F.softmax(S, dim=-1)  # (N_t, N_v)
        
        # Check for NaN in P
        if torch.isnan(P).any():
            STRATEGY_LOG.error(f"Layer {layer_idx}: NaN detected in softmax matrix P")
            return torch.tensor(0.0, device=h_v.device, dtype=h_v.dtype)
        
        # Step 3: Local Peaking - mean entropy per row
        # Clamp P to avoid log(0)
        P_clamped = torch.clamp(P, min=self.eps, max=1.0)
        local_entropy = -(P_clamped * torch.log(P_clamped)).sum(dim=-1).mean()
        
        # Step 4: Global Diversity - entropy of row mean
        P_bar = P.mean(dim=0)  # (N_v,)
        P_bar_clamped = torch.clamp(P_bar, min=self.eps, max=1.0)
        global_entropy = -(P_bar_clamped * torch.log(P_bar_clamped)).sum()
        
        # Combined loss
        loss = local_entropy - self.lambda_global * global_entropy
        
        # Final NaN check
        if torch.isnan(loss):
            STRATEGY_LOG.error(
                f"Layer {layer_idx}: Final loss is NaN! "
                f"local_entropy={local_entropy.item():.4f}, "
                f"global_entropy={global_entropy.item():.4f}"
            )
            return torch.tensor(0.0, device=h_v.device, dtype=h_v.dtype)
        
        return loss
    
    def _register_output_hooks(self, model: nn.Module, layer_indices: list):
        """Register hooks on decoder layer outputs to collect hidden states for IADE loss."""
        layers = self._get_layers(model)
        handles = []
        
        for layer_idx in layer_indices:
            def make_output_hook(idx):
                def output_hook(module, input, output):
                    """Collect hidden state output."""
                    if isinstance(output, tuple):
                        hidden = output[0]
                    else:
                        hidden = output
                    self._stored_hidden_outputs.append((idx, hidden))
                    return output
                return output_hook
            
            handle = layers[layer_idx].register_forward_hook(make_output_hook(layer_idx))
            handles.append(handle)
        
        return handles
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Compute MANTA loss using hooks for proper gradient flow."""
        if self.gamma is None or self.optimizer is None:
            raise RuntimeError("MANTA strategy not initialized. Call setup() first.")
        
        # Get visual/text token separation
        num_image_tokens = self._compute_num_image_tokens(inputs)
        
        # DEBUG: Log if num_image_tokens is 0
        if num_image_tokens == 0:
            STRATEGY_LOG.warning(
                "No image tokens found! This will result in zero IADE loss. "
                "Check if image_grid_thw is in inputs."
            )
            # Run standard forward to get outputs
            out = tta_model_forward(model, inputs)
            dummy_loss = out.logits.float().sum() * 0.0
            return (dummy_loss, out) if return_outputs else dummy_loss
        
        # Get sequence length from input_ids
        input_ids = inputs["input_ids"]
        B, seq_len = input_ids.shape
        
        # Create visual and text masks
        # Token order in Qwen3.5: [visual_tokens, text_tokens]
        visual_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=input_ids.device)
        text_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=input_ids.device)
        
        if num_image_tokens <= seq_len:
            visual_mask[:, :num_image_tokens] = True
            text_mask[:, num_image_tokens:] = True
            STRATEGY_LOG.debug(
                f"Mask created: num_image_tokens={num_image_tokens}, seq_len={seq_len}, "
                f"visual_tokens_per_sample={visual_mask.sum(dim=1).tolist()}, "
                f"text_tokens_per_sample={text_mask.sum(dim=1).tolist()}"
            )
        else:
            STRATEGY_LOG.warning(
                f"num_image_tokens ({num_image_tokens}) > seq_len ({seq_len}), "
                f"adjusting masks"
            )
            visual_mask[:, :seq_len] = True
        
        # Verify masks are not empty
        if visual_mask.sum() == 0:
            STRATEGY_LOG.error("visual_mask is all False! IADE loss will be zero.")
        if text_mask.sum() == 0:
            STRATEGY_LOG.error("text_mask is all False! IADE loss will be zero.")
        
        # Set masks for hooks to use
        self._current_visual_mask = visual_mask
        self._current_text_mask = text_mask
        
        L = self.num_layers
        K = self.gamma.shape[0]
        adapt_layer_indices = list(range(L - K, L))
        
        # Register output hooks to collect hidden states for IADE loss
        output_hook_handles = self._register_output_hooks(model, adapt_layer_indices)
        
        # Perform MANTA adaptation steps
        total_loss = None
        
        for step in range(self.adapt_steps):
            self.optimizer.zero_grad()
            
            # Clear stored outputs
            self._stored_hidden_outputs = []
            self._stored_modulated_outputs = []  # Clear modulated outputs
            
            # Forward pass through model (hooks will apply gamma modulation automatically)
            # RMSNorm hooks store modulated outputs for IADE loss
            out = tta_model_forward(model, inputs)
            
            # Compute IADE loss from modulated outputs (direct gradient flow to gamma)
            layer_losses = []
            for layer_k, modulated_hidden in self._stored_modulated_outputs:
                # Apply masks to get visual and text tokens
                h_v = modulated_hidden[visual_mask]
                h_t = modulated_hidden[text_mask]
                
                # Compute IADE loss for this layer
                layer_loss = self._compute_iade_loss(h_v, h_t, layer_idx=layer_k)
                layer_losses.append(layer_loss)
            
            if layer_losses:
                step_loss = sum(layer_losses) / len(layer_losses)
            else:
                STRATEGY_LOG.warning("No modulated outputs collected, using dummy loss")
                step_loss = out.logits.float().sum() * 0.0
            
            # Diagnostic logging
            if step == 0:
                STRATEGY_LOG.debug(
                    f"Collected {len(self._stored_modulated_outputs)} modulated outputs, "
                    f"step_loss={step_loss.item():.6f}"
                )
            
            # Backward and optimize
            step_loss.backward()
            
            # Check for NaN gradients
            has_nan_grad = False
            if self.gamma.grad is not None:
                if torch.isnan(self.gamma.grad).any():
                    has_nan_grad = True
                    STRATEGY_LOG.error("NaN detected in gamma gradients! Skipping optimizer step.")
            
            if not has_nan_grad:
                self.optimizer.step()
            else:
                STRATEGY_LOG.error("Optimizer step skipped due to NaN gradients")
            
            if total_loss is None:
                total_loss = step_loss.detach()
            else:
                total_loss = total_loss + step_loss.detach()
        
        # Remove output hooks
        for handle in output_hook_handles:
            handle.remove()
        
        # Clear masks
        self._current_visual_mask = None
        self._current_text_mask = None
        self._stored_modulated_outputs = []  # Clear modulated outputs
        
        # Average loss across steps
        avg_loss = total_loss / self.adapt_steps
        
        STRATEGY_LOG.info(
            f"MANTA loss: {avg_loss.item():.6f}, gamma[0]={self.gamma[0].tolist()}"
        )
        
        return (avg_loss, out) if return_outputs else avg_loss
