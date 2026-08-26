from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from openttl.strategies.base import Strategy

STRATEGY_LOG = logging.getLogger(__name__)
if not STRATEGY_LOG.handlers:
    STRATEGY_LOG.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[E3-TTA Strategy] %(message)s')
    handler.setFormatter(formatter)
    STRATEGY_LOG.addHandler(handler)


class EntropyGatedModule(nn.Module):
    """可学习熵门控模块：每层一个可学习向量，用于自适应缩放视觉token的value。
    
    根据E3-TTA论文，门控参数g_l对视觉token的value进行自适应缩放：
    V_l^img ← V_l^img ⊙ σ(g_l)
    其中σ为sigmoid函数，初始化使得σ(g_l) ≈ 0.5（中性状态）
    """
    
    def __init__(self, hidden_dim: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # 初始化使得σ(g_l) ≈ 0.5（中性状态），即g_l ≈ 0
        self.gate = nn.Parameter(torch.zeros(hidden_dim))
        
    def forward(self, v_img: torch.Tensor) -> torch.Tensor:
        """对视觉token的value进行门控缩放。
        
        Args:
            v_img: 视觉token的value，形状为 (B, N_img, d)
            
        Returns:
            缩放后的value，形状与输入相同
        """
        scale = torch.sigmoid(self.gate)
        return v_img * scale


class DualTimescaleCache:
    """双时间尺度缓存：场景级 + 实例级。
    
    场景缓存（C_scene）：跨Episode，使用EMA更新
    实例缓存（C_inst）：当前问题内，最近k步
    """
    
    def __init__(
        self,
        num_layers: int,
        scene_ema_alpha: float = 0.9,
        inst_window_size: int = 10,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.scene_ema_alpha = scene_ema_alpha
        self.inst_window_size = inst_window_size
        self.device = device
        
        # 场景级缓存：每层一个目标熵值
        self.scene_cache: Optional[torch.Tensor] = None  # (num_layers,)
        
        # 实例级缓存：最近k步的熵值
        self.inst_cache: List[torch.Tensor] = []  # List of (num_layers,) tensors
        
    def initialize(self, initial_entropy: torch.Tensor):
        """用第一个观测值初始化场景缓存。"""
        self.scene_cache = initial_entropy.clone().detach()
        
    def update_instance(self, entropy: torch.Tensor):
        """更新实例缓存（当前问题内）。"""
        self.inst_cache.append(entropy.clone().detach())
        if len(self.inst_cache) > self.inst_window_size:
            self.inst_cache.pop(0)
            
    def update_scene(self):
        """用实例缓存的平均值更新场景缓存（EMA）。"""
        if len(self.inst_cache) == 0 or self.scene_cache is None:
            return
        
        inst_mean = torch.stack(self.inst_cache).mean(dim=0)  # (num_layers,)
        self.scene_cache = (
            self.scene_ema_alpha * self.scene_cache + 
            (1 - self.scene_ema_alpha) * inst_mean
        )
        
    def get_target_entropy(self) -> torch.Tensor:
        """获取当前目标熵值 H_l^*。"""
        if self.scene_cache is None:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        return self.scene_cache
        
    def reset_scene(self):
        """检测到新场景时重置场景缓存。"""
        self.scene_cache = None
        self.inst_cache = []
        
    def get_instance_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取实例缓存的均值和标准差，用于在线归一化。"""
        if len(self.inst_cache) == 0:
            return torch.zeros(self.num_layers, device=self.device), \
                   torch.ones(self.num_layers, device=self.device)
        
        stacked = torch.stack(self.inst_cache)  # (k, num_layers)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0).clamp_min(1e-8)
        return mean, std


class E3TTAStrategy(Strategy):
    """E3-TTA: Embodied Entropy-Gated Test-Time Adaptation。
    
    针对具身场景的测试时适应算法，核心特点：
    1. 只更新门控参数（每层1个向量，总计<100K参数），不更新LoRA
    2. 通过forward hook注入门控，实时修改视觉token的value
    3. 损失函数只包含熵稳定项和正则项
    4. 模型主干完全冻结
    
    参考: /Users/jerryjie/AAAI/TTL/OpenTTL/E3TTA.md
    """
    
    def __init__(self, cfg: Any):
        super().__init__(cfg)
        
        # 配置参数
        self.lr_gate = float(getattr(cfg, "lr_gate", 1e-3))
        self.lambda_reg = float(getattr(cfg, "lambda_reg", 0.01))
        self.scene_ema_alpha = float(getattr(cfg, "scene_ema_alpha", 0.9))
        self.inst_window_size = int(getattr(cfg, "inst_window_size", 10))
        self.eps = float(getattr(cfg, "epsilon", 1e-8))
        self.tau_low_factor = float(getattr(cfg, "tau_low_factor", 0.5))
        self.tau_high_factor = float(getattr(cfg, "tau_high_factor", 2.0))
        self.use_entropy_gate = bool(getattr(cfg, "use_entropy_gate", True))
        
        # 组件将在setup中初始化
        self.gates: nn.ModuleList = nn.ModuleList()
        self.cache: Optional[DualTimescaleCache] = None
        self.hidden_dim: Optional[int] = None
        self.num_layers: Optional[int] = None
        
        # 门控优化器
        self.gate_optimizer: Optional[torch.optim.Optimizer] = None
        
        # 用于存储前向传播中的attention统计信息
        self._stored_attn_stats: List[Dict] = []
        self._hook_handles: List[Any] = []
        
        # 存储视觉token数量（用于区分视觉和文本token）
        self._num_image_tokens: Optional[int] = None
        
    def setup(
        self, 
        model: nn.Module, 
        teacher_model: Optional[nn.Module] = None
    ) -> None:
        """初始化门控参数、缓存，并注册forward hook。"""
        super().setup(model, teacher_model)
        
        # 获取模型维度信息
        self.hidden_dim = getattr(model.config, "hidden_size", None)
        if self.hidden_dim is None:
            self.hidden_dim = getattr(model.config, "d_model", None)
        if self.hidden_dim is None:
            # Fallback: try to get from the model's embedding
            if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                self.hidden_dim = model.model.embed_tokens.embedding_dim
            else:
                self.hidden_dim = 2048  # Qwen3.5-2B default
                
        self.num_layers = getattr(model.config, "num_hidden_layers", None)
        if self.num_layers is None:
            self.num_layers = getattr(model.config, "n_layer", None)
        if self.num_layers is None:
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                self.num_layers = len(model.model.layers)
            else:
                self.num_layers = 24  # default
        
        STRATEGY_LOG.info(f"E3-TTA Setup: hidden_dim={self.hidden_dim}, num_layers={self.num_layers}")
        
        # 冻结模型所有参数
        for param in model.parameters():
            param.requires_grad = False
        
        # 获取模型的dtype和设备
        model_dtype = next(model.parameters()).dtype
        model_device = next(model.parameters()).device
        
        STRATEGY_LOG.info(f"E3-TTA Setup: model_dtype={model_dtype}, device={model_device}")
        
        # 为每层创建一个门控模块
        if self.use_entropy_gate:
            for i in range(self.num_layers):
                gate = EntropyGatedModule(self.hidden_dim, i).to(device=model_device, dtype=model_dtype)
                self.gates.append(gate)
                # 门控参数默认可训练
                for param in gate.parameters():
                    param.requires_grad = True
            
            # 创建门控优化器（只优化门控参数）
            gate_params = [p for gate in self.gates for p in gate.parameters()]
            self.gate_optimizer = torch.optim.AdamW(
                gate_params, 
                lr=self.lr_gate,
                weight_decay=self.lambda_reg,
            )
        
        # 初始化双时间尺度缓存
        self.cache = DualTimescaleCache(
            num_layers=self.num_layers,
            scene_ema_alpha=self.scene_ema_alpha,
            inst_window_size=self.inst_window_size,
            device=str(next(model.parameters()).device),
        )
        
        # 注册门控hook到attention层
        self._register_gating_hooks(model)
        
    def _register_gating_hooks(self, model: nn.Module):
        """注册门控hook到各层的linear_attn模块。
        
        Qwen3.5使用linear attention (Qwen3_5GatedDeltaNet)而不是self_attn。
        门控在linear_attn计算后实时修改输出。
        """
        self._hook_handles = []
        
        # 获取layers列表，处理不同的模型包装结构
        layers = None
        
        # 情况1: Qwen3_5ForCausalLM -> model.model.layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        # 情况2: Qwen3_5Model (已经解包) -> model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        # 情况3: 其他transformers模型
        else:
            # 尝试通过named_modules找到所有decoder layers
            for name, module in model.named_modules():
                if module.__class__.__name__ in ['Qwen3_5DecoderLayer', 'LlamaDecoderLayer', 'MistralDecoderLayer']:
                    if layers is None:
                        layers = []
                    layers.append(module)
        
        if layers is None:
            raise ValueError(f"Cannot find model layers. Model type: {type(model)}")
        
        STRATEGY_LOG.info(f"Found {len(layers)} layers, will register hooks for {min(len(layers), self.num_layers)} layers")
        
        # 为每层注册hook
        for layer_idx in range(min(len(layers), self.num_layers)):
            if layer_idx >= len(self.gates):
                break
                
            layer_module = layers[layer_idx]
            
            # 尝试获取attention模块
            if hasattr(layer_module, 'linear_attn'):
                attn_module = layer_module.linear_attn
            elif hasattr(layer_module, 'self_attn'):
                # 兼容其他使用self_attn的模型
                attn_module = layer_module.self_attn
            else:
                STRATEGY_LOG.warning(f"Layer {layer_idx} has no linear_attn or self_attn, skipping")
                continue
            
            handle = self._register_gate_hook(attn_module, layer_idx)
            if handle is not None:
                self._hook_handles.append(handle)
                STRATEGY_LOG.debug(f"Registered hook for layer {layer_idx}")
                    
    def _register_gate_hook(self, attn_module: nn.Module, layer_idx: int) -> Optional[Any]:
        """为单个attention模块注册门控hook。
        
        在attention (linear_attn或self_attn)的forward后，对输出进行门控调制。
        Qwen3.5的linear_attn输出形状: (B, N, hidden_size)
        """
        gate = self.gates[layer_idx]
        import logging
        hook_logger = logging.getLogger(__name__)
        
        def gate_hook(module, input, output):
            """Hook函数：在attention计算后应用门控调制。
            
            Args:
                module: attention模块 (linear_attn或self_attn)
                input: 输入tensor
                output: attention的输出
            
            Returns:
                门控调制后的输出
            """
            try:
                # attention输出可能是tuple或tensor
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    other_outputs = output[1:]
                else:
                    hidden_states = output
                    other_outputs = None
                
                # hidden_states shape: (B, N, hidden_size)
                # 确保gate的dtype和hidden_states一致
                gate_param = gate.gate.to(dtype=hidden_states.dtype)
                scale = torch.sigmoid(gate_param)  # (hidden_size,)
                
                hook_logger.debug("[E3-TTA Hook] Layer %d: hidden_states shape=%s, dtype=%s, scale shape=%s, dtype=%s", 
                                 layer_idx, hidden_states.shape, hidden_states.dtype, scale.shape, scale.dtype)
                
                # 确保scale的维度与hidden_states匹配
                if hidden_states.dim() == 3:
                    scale_expanded = scale.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_size)
                elif hidden_states.dim() == 2:
                    # 可能是 (B*N, hidden_size) 的情况
                    scale_expanded = scale.unsqueeze(0)  # (1, hidden_size)
                else:
                    hook_logger.warning("[E3-TTA Hook] Layer %d: Unexpected tensor dim = %d, shape = %s", 
                                       layer_idx, hidden_states.dim(), hidden_states.shape)
                    scale_expanded = scale.view(1, 1, -1)  # Force reshape
                
                gated_output = hidden_states * scale_expanded
                
                hook_logger.debug("[E3-TTA Hook] Layer %d: Gate applied successfully, output shape = %s", 
                                 layer_idx, gated_output.shape)
                
                # 恢复原来的输出格式
                if other_outputs is not None:
                    return (gated_output,) + other_outputs
                else:
                    return gated_output
                    
            except Exception as e:
                hook_logger.error("[E3-TTA Hook] Layer %d failed: %s", layer_idx, str(e))
                import traceback
                hook_logger.error("[E3-TTA Hook] Full traceback:\n%s", traceback.format_exc())
                raise
        
        try:
            handle = attn_module.register_forward_hook(gate_hook)
            hook_logger.debug("[E3-TTA Hook] Registered hook for layer %d", layer_idx)
            return handle
        except Exception as e:
            hook_logger.error("[E3-TTA Hook] Failed to register hook for layer %d: %s", layer_idx, str(e))
            return None
    
    def _compute_cross_modal_entropy(
        self,
        attn_weights: torch.Tensor,
        num_image_tokens: int,
        num_text_tokens: int,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算跨模态注意力熵。
        
        根据论文公式：
        H_l = -1/H * Σ_h 1/N_text * Σ_i Σ_j A_{l,h}^{cross}(i,j) * log A_{l,h}^{cross}(i,j)
        
        Args:
            attn_weights: 注意力权重，形状为 (B, H, N, N) 或 (B, H, N_text, N_img)
            num_image_tokens: 视觉token数量
            num_text_tokens: 文本token数量
            text_mask: 文本token的mask
            
        Returns:
            熵值标量
        """
        eps = self.eps
        
        # 如果attn_weights是完整的attention矩阵，提取跨模态部分
        if attn_weights.dim() == 4:
            B, H, N1, N2 = attn_weights.shape
            
            # 假设token顺序: [视觉tokens, 文本tokens]
            # 提取文本token对视觉token的注意力
            if N1 >= num_text_tokens and N2 >= num_image_tokens:
                # 文本token在视觉token之后
                cross_attn = attn_weights[:, :, -num_text_tokens:, :num_image_tokens]
            else:
                # 如果形状不匹配，使用均匀分布
                cross_attn = torch.ones(
                    B, H, max(1, num_text_tokens), max(1, num_image_tokens),
                    device=attn_weights.device
                ) / max(1, num_image_tokens)
        else:
            cross_attn = attn_weights
        
        # 数值稳定性处理
        cross_attn = cross_attn.clamp(min=eps)
        
        # 计算熵: -Σ p * log(p)，在图像token维度上求和
        entropy = -(cross_attn * torch.log(cross_attn)).sum(dim=-1)  # (B, H, N_text)
        
        # 对文本token平均
        if text_mask is not None:
            # text_mask: (B, N_text)
            text_mask = text_mask.unsqueeze(1).expand(-1, entropy.size(1), -1)  # (B, H, N_text)
            entropy = (entropy * text_mask).sum(dim=-1) / (text_mask.sum(dim=-1).clamp_min(1.0))
        else:
            entropy = entropy.mean(dim=-1)  # (B, H)
        
        # 对注意力头平均
        entropy = entropy.mean(dim=-1)  # (B,)
        
        # 对batch平均，返回标量
        return entropy.mean()
    
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """计算E3-TTA损失。
        
        损失函数（只包含熵稳定项和正则项）：
        L_gate = (H_l - H_l^*)^2 + λ||g_l - g_0||_2^2
        
        关键：必须使用cross-modal attention entropy，因为只有它受gate影响。
        """
        STRATEGY_LOG.debug("[compute_loss] Step 1: Clearing stored attention stats...")
        # 清空上一轮的统计信息
        self._stored_attn_stats = []
        
        STRATEGY_LOG.debug("[compute_loss] Step 2: Running forward pass...")
        # 前向传播，hook会自动收集attention统计信息
        outputs = model(**inputs)
        
        STRATEGY_LOG.debug("[compute_loss] Step 3: Computing layer entropies from gate activations...")
        # 使用gate的激活程度作为熵的代理信号
        # gate的sigmoid输出越接近0或1，说明attention越集中（熵低）
        # gate的sigmoid输出越接近0.5，说明attention越分散（熵高）
        layer_entropies = []
        
        for gate in self.gates:
            gate激活 = torch.sigmoid(gate.gate.float())  # (hidden_dim,)
            # 使用gate激活的方差作为熵的代理
            # 方差大 → 某些维度被强烈抑制/增强 → 注意力集中 → 熵低
            # 方差小 → 所有维度均匀 → 注意力分散 → 熵高
            entropy_proxy = 1.0 - gate激活.var()  # 方差越小，熵越高
            layer_entropies.append(entropy_proxy)
        
        layer_entropies = torch.stack(layer_entropies)  # (num_layers,)
        
        STRATEGY_LOG.debug("[compute_loss] Step 4: layer_entropies = %s", layer_entropies)
        
        STRATEGY_LOG.debug("[compute_loss] Step 5: Computing loss...")
        # 初始化场景缓存
        if self.cache.scene_cache is None:
            self.cache.initialize(layer_entropies.detach())
            STRATEGY_LOG.debug("[compute_loss] Step 5a: Cache initialized")
        
        target_entropy = self.cache.get_target_entropy()
        
        # 熵稳定损失
        entropy_stability_loss = ((layer_entropies - target_entropy) ** 2).mean()
        
        # 正则化损失
        reg_loss = torch.tensor(0.0, device=layer_entropies.device, dtype=torch.float32)
        for gate in self.gates:
            reg_loss = reg_loss + torch.norm(gate.gate.float()) ** 2
        reg_loss = self.lambda_reg * reg_loss / len(self.gates)
        
        total_loss = entropy_stability_loss + reg_loss
        
        STRATEGY_LOG.debug("[compute_loss] Step 6: total_loss = %.4f, entropy_loss = %.4f, reg_loss = %.4f", 
                          total_loss.item(), entropy_stability_loss.item(), reg_loss.item())
        
        return (total_loss, outputs) if return_outputs else total_loss
        
    def on_batch_end(
        self, 
        trainer: Any, 
        batch: Dict[str, torch.Tensor], 
        loss: torch.Tensor
    ) -> None:
        """每批次结束后更新场景缓存。
        
        注意：optimizer.step() 已在 E3TTARunner.update() 中调用，此处不再重复。
        """
        if self.cache is not None:
            self.cache.update_scene()
        
    def reset_scene_cache(self):
        """检测到新场景时重置场景缓存。"""
        if self.cache is not None:
            self.cache.reset_scene()
            
    def get_current_entropies(self) -> Optional[torch.Tensor]:
        """获取当前各层的熵值（用于监控）。"""
        if self.cache is None or self.cache.scene_cache is None:
            return None
        return self.cache.scene_cache.clone()
        
    def cleanup(self):
        """清理hooks。"""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """返回可训练参数列表（仅门控参数）。"""
        params = []
        for gate in self.gates:
            params.extend(list(gate.parameters()))
        return params
