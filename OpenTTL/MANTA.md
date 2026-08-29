## 二、MANTA-Uni：统一隐空间跨模态熵

### 2.1 核心洞察（Observation）

> **Observation（统一隐空间关联性）**：在任意 Early-Fusion 多模态大模型中，无论第 $l$ 层的序列混合算子是标准 MHA、GQA、DeltaNet 还是 RWKV，其输出都将视觉 token 与文本 token 映射到**同一个 $d$ 维向量空间**。因此，**文本 token 与视觉 token 的表示内积**天然构成了一个与架构无关的“隐式跨模态关联分布”。

我们无需窥探层内算子，只需在**层后空间**操作。

### 2.2 可学习参数（极简）

仅对选定层（如最后 $K=4$ 层，可任意调整）引入 **2 个标量**：

$$
\Theta_{\text{adapt}} = \left\{ \gamma_v^{(l)},\ \gamma_t^{(l)} \right\}_{l=L-K+1}^{L}
$$

总参数量 **$2K$**（例如 $K=4$ 时仅 **8 个参数**）。若追求极致，可令 $K=1$（仅最后输出层，**2 个参数**）。

### 2.3 模态感知 RMSNorm 调制（与之前一致）

Qwen3.5 / Gemma4 均使用 RMSNorm。对第 $l$ 层输入隐状态 $H^{(l-1)}$：

$$
\tilde{H}^{(l-1)}[j] = \frac{H^{(l-1)}[j]}{\text{RMS}(H^{(l-1)}[j])} \odot (\gamma_v^{(l)} \cdot \gamma_0^{(l)}),\quad j \in \mathcal{V}
$$

$$
\tilde{H}^{(l-1)}[i] = \frac{H^{(l-1)}[i]}{\text{RMS}(H^{(l-1)}[i])} \odot (\gamma_t^{(l)} \cdot \gamma_0^{(l)}),\quad i \in \mathcal{T}
$$

所有权重矩阵、投影层、视觉编码器**完全冻结**。RMSNorm 的预训练权重 $\gamma_0^{(l)}$ 也冻结，仅乘性调制 $\gamma_v^{(l)}, \gamma_t^{(l)}$ 可学习。

### 2.4 统一自监督损失：隐式关联分布熵（IADE）

设第 $l$ 层**输出**为 $H^{(l)}$（已完成前向计算，无论内部是何种算子），分离视觉/文本：

$$
H_v^{(l)} = \{h_{v,j}^{(l)}\}_{j=1}^{N_v},\quad H_t^{(l)} = \{h_{t,i}^{(l)}\}_{i=1}^{N_t}
$$

**Step 1：构建隐式跨模态关联矩阵**

$$
S_{ij}^{(l)} = \frac{h_{t,i}^{(l)\top} h_{v,j}^{(l)}}{\tau},\quad \tau = \sqrt{d}
$$

**Step 2：行级 Softmax（文本→视觉的隐式注意力）**

$$
P_{ij}^{(l)} = \frac{\exp(S_{ij}^{(l)})}{\sum_{j'=1}^{N_v} \exp(S_{ij'}^{(l)})}
$$

**Step 3：统一熵损失（所有层共享同一公式）**

$$
\mathcal{L}_{\text{MANTA-Uni}}^{(l)} = \underbrace{\sum_{i=1}^{N_t} H\left(P_{i,\cdot}^{(l)}\right)}_{\text{Local Peaking } \mathcal{L}_{\text{local}}} \ -\ \lambda \underbrace{H\left( \frac{1}{N_t}\sum_{i=1}^{N_t} P_{i,\cdot}^{(l)} \right)}_{\text{Global Diversity } \mathcal{L}_{\text{global}}}
$$

**Step 4：层聚合**

$$
\mathcal{L}_{\text{MANTA-Uni}} = \sum_{l=L-K+1}^{L} \mathcal{L}_{\text{MANTA-Uni}}^{(l)}
$$

---

## 三、为什么这个公式是“通用且优雅”的

| 特性 | 解释 |
|------|------|
| **与算子无关** | 不依赖 QK^T、不依赖 DeltaNet 门控、不依赖 Mamba 状态。只要模型输出 $d$ 维向量，就能计算内积 |
| **与归一化无关** | 无论是 RMSNorm（Qwen/Gemma）还是 LayerNorm（LLaVA/InternVL），调制公式只需改为乘性缩放即可 |
| **与模态编码器无关** | 视觉是 ViT、CNN 还是 Patchify 都不关心，因为进入 LLM 后都是 $d$ 维 token |
| **与任务无关** | ERQA、VQA、Grounding 都适用，因为损失只约束“问题-视觉”对齐结构，不依赖答案生成长度 |

---

## 四、Qwen3.5 上的极简实现代码

不再需要判断层类型。核心逻辑只有 **20 行**：

```python
import torch
import torch.nn.functional as F

class MANTAUni:
    def __init__(self, model, num_adapt_layers=4, lambda_global=1.0):
        """
        model: 原生多模态 Early-Fusion 模型 (Qwen3.5-VL / Gemma-4V)
        num_adapt_layers: 只更新最后 K 层，默认 4
        """
        self.model = model
        self.L = len(model.model.layers)
        self.K = num_adapt_layers
        self.lambda_global = lambda_global
        self.tau = model.config.hidden_size ** 0.5
        
        # 仅引入 2K 个标量参数
        self.gamma = torch.nn.Parameter(torch.ones(self.K, 2))  # [K, 2], (visual, text)
        
        # 冻结全部
        for p in self.model.parameters():
            p.requires_grad = False
        self.gamma.requires_grad = True
        
        self.opt = torch.optim.Adam([self.gamma], lr=1e-2)

    def modality_rmsnorm(self, hidden, layer_idx, visual_mask, text_mask):
        """
        对第 layer_idx 层应用模态感知 RMSNorm 调制。
        适用于 Qwen3.5 (RMSNorm) 和 Gemma4 (RMSNorm)。
        若为 LayerNorm 模型，将 RMS 替换为 mean/var 即可。
        """
        # 获取该层预训练 RMSNorm 权重 (d,)
        rmsnorm = self.model.model.layers[layer_idx].input_layernorm
        gamma_0 = rmsnorm.weight  # (d,)
        
        # RMS 计算
        rms = hidden.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        normed = hidden / rms  # (N, d)
        
        # 构建模态缩放 (N, 1)
        scale = torch.ones(hidden.size(0), 1, device=hidden.device)
        # 只更新最后 K 层，gamma 索引映射
        k = layer_idx - (self.L - self.K)
        if 0 <= k < self.K:
            scale[visual_mask] = self.gamma[k, 0]
            scale[text_mask] = self.gamma[k, 1]
        
        return normed * (scale * gamma_0)

    def compute_layer_loss(self, h_t, h_v):
        """
        统一的 IADE 损失，适用于任何层。
        h_t: (N_t, d), h_v: (N_v, d)
        """
        # 隐式关联矩阵 (N_t, N_v)
        S = torch.matmul(h_t, h_v.t()) / self.tau
        
        # 行级 softmax
        P = F.softmax(S, dim=-1)  # (N_t, N_v)
        
        # Local Peaking: 每行熵之和
        local = -(P * torch.log(P + 1e-9)).sum(dim=-1).mean()
        
        # Global Diversity: 行均值的熵
        P_bar = P.mean(dim=0)  # (N_v,)
        global_ent = -(P_bar * torch.log(P_bar + 1e-9)).sum()
        
        return local - self.lambda_global * global_ent

    def forward_and_loss(self, unified_embeds, visual_mask, text_mask):
        hidden = unified_embeds
        total_loss = 0.0
        
        for l in range(self.L):
            # 1. 模态感知归一化（仅最后 K 层生效，前面 gamma=1）
            hidden = self.modality_rmsnorm(hidden, l, visual_mask, text_mask)
            
            # 2. 通过模型第 l 层（完全冻结，无需关心是 Attention 还是 DeltaNet）
            hidden = self.model.model.layers[l](hidden)[0] if isinstance(
                self.model.model.layers[l](hidden), tuple
            ) else self.model.model.layers[l](hidden)
            
            # 3. 仅对最后 K 层计算统一损失
            if l >= self.L - self.K:
                h_v = hidden[visual_mask]
                h_t = hidden[text_mask]
                total_loss += self.compute_layer_loss(h_t, h_v)
        
        return total_loss, hidden

    def adapt(self, unified_embeds, visual_mask, text_mask, steps=5):
        for _ in range(steps):
            self.opt.zero_grad()
            loss, _ = self.forward_and_loss(unified_embeds, visual_mask, text_mask)
            loss.backward()
            self.opt.step()
        # 返回优化后的 gamma 已嵌入模型，可直接 generate
```

---
