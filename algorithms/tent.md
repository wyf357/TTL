# Skill: TENT-LLM (Test-Time Entropy Minimization with LoRA)

## Description
This skill implements the **TENT** algorithm adapted for **Large Language Models (LLMs)**. It enables fully test-time adaptation by minimizing the **average entropy of the generated sequence** using only unlabeled test data. 

**Key Adaptations from Original TENT:**
1.  **Parameter Modulation**: Replaces Batch Normalization affine parameters with **LoRA (Low-Rank Adaptation)** matrices. This respects the Transformer architecture (which lacks global BN statistics) while maintaining TENT's principle of efficient, low-dimensional parameter updates to prevent catastrophic forgetting.
2.  **Entropy Objective**: Computes the **mean Shannon entropy** across all tokens in the generated sequence (or input context), rather than a single classification output. This leverages the autoregressive nature of LLMs to enforce confidence throughout the generation process.

## Core Principles
1.  **Autoregressive Entropy Minimization**: The model adapts by reducing uncertainty in its next-token predictions. High entropy indicates hesitation or confusion; minimizing it forces the model to commit to high-probability tokens consistent with the target domain's distribution.
2.  **Low-Rank Feature Modulation**: Only LoRA parameters ($A, B$) are updated. The pre-trained weights remain frozen. This ensures stability and computational efficiency during inference.
3.  **Source-Free & Unsupervised**: Requires no labeled data and no access to the original training set. Adaptation is driven solely by the model's own predictions on the test input.

## Mathematical Formulation

### 1. Sequence Entropy Objective
For a given input sequence $x$, let the LLM generate a sequence of tokens $y = \{y_1, y_2, ..., y_T\}$. At each step $t$, the model outputs a probability distribution $P_t$ over the vocabulary $V$.

The **Average Sequence Entropy** $\mathcal{L}_{ent}$ is defined as:

$$
\mathcal{L}_{ent}(x, y; \Theta_{LoRA}) = -\frac{1}{T} \sum_{t=1}^{T} \sum_{v \in V} P_t(v) \log(P_t(v) + \epsilon)
$$

Where:
*   $T$ is the length of the generated sequence (or the number of tokens considered).
*   $P_t(v)$ is the softmax probability of token $v$ at step $t$.
*   $\Theta_{LoRA}$ represents the trainable LoRA parameters.
*   $\epsilon$ is a small constant for numerical stability (e.g., $10^{-8}$).

*Note: In some implementations, this entropy can be computed on the **input prompt** tokens (self-supervised masked prediction) or the **generated response** tokens. For general adaptation, minimizing entropy on the generated response is standard for generative tasks.*

### 2. LoRA Parameterization
The weight update for a linear layer $W$ is decomposed as:
$$
W' = W + \Delta W = W + B A
$$
Where:
*   $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{k \times r}$ are low-rank matrices ($r \ll d, k$).
*   **Frozen**: Pre-trained weights $W$.
*   **Trainable**: $\Theta_{LoRA} = \{A_l, B_l\}$ for selected layers $l$ (typically Attention $q\_proj, v\_proj$).

### 3. Optimization Step
The LoRA parameters are updated via gradient descent to minimize the sequence entropy:

$$
\Theta_{LoRA} \leftarrow \Theta_{LoRA} - \eta \nabla_{\Theta_{LoRA}} \mathcal{L}_{ent}
$$

## Algorithm Workflow

1.  **Initialization**:
    *   Load pre-trained LLM.
    *   Inject LoRA adapters into target modules (e.g., Attention layers).
    *   Initialize LoRA params ($A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$).
    *   Freeze all backbone weights.

2.  **Test-Time Adaptation Loop** (for each unlabeled test sample $x$):
    a.  **Generation/Forward Pass**:
        *   Enable gradients for LoRA parameters.
        *   Generate sequence $y$ (or process input $x$) to obtain logits for $T$ tokens.
        *   Compute softmax probabilities $P_t$ for each token position.
    
    b.  **Loss Computation**:
        *   Calculate Mean Sequence Entropy $\mathcal{L}_{ent}$ using the formula above.

    c.  **Backward Pass**:
        *   Compute gradients $\nabla_{\Theta_{LoRA}} \mathcal{L}_{ent}$.

    d.  **Parameter Update**:
        *   Update LoRA parameters using an optimizer (e.g., AdamW).

    e.  **Inference**:
        *   Use the updated model for subsequent predictions.

## Pseudo-code

```python
Algorithm: TENT-LLM (LoRA-based Sequence Entropy Minimization)

Input: 
    - Pre-trained LLM f_theta with frozen weights theta
    - Unlabeled test data stream D_test = {x_j}
    - LoRA rank r, learning rate eta, sequence length T
    - Optimizer Opt (e.g., AdamW)

Output: 
    - Adapted LLM with updated LoRA parameters Theta_LoRA

1: // Initialization
2: Inject LoRA adapters into LLM attention layers
3: Define trainable parameters Theta_LoRA = {A_l, B_l}
4: Freeze all other parameters in theta
5: Initialize Opt with Theta_LoRA and learning rate eta

6: // Test-Time Adaptation Loop
7: For each sample x in D_test do:

8:     // Forward Pass with Gradients Enabled for LoRA
9:     Set LoRA modules to train() mode
10:    Set LLM backbone to eval() mode
    
11:    // Generate sequence or get logits for T tokens
12:    // Y_hat contains logits for each step t=1...T
13:    Logits = f_theta(x, Theta_LoRA) 
    
14:    // Compute Probabilities for each token step
15:    Probs = Softmax(Logits, dim=-1)  // Shape: [B, T, Vocab]
    
16:    // Compute Mean Sequence Entropy
17:    // H_t = - sum(p * log(p)) for each token t
18:    Token_Entropies = -sum(Probs * log(Probs + 1e-8), dim=-1)
19:    L_ent = mean(Token_Entropies)    // Average over T steps
    
20:    // Backward Pass
21:    Opt.zero_grad()
22:    L_ent.backward()
    
23:    // Update LoRA Parameters
24:    Opt.step()
    
25: End For

26: Return LLM with updated Theta_LoRA
```

## Implementation Details & Best Practices

*   **Target Modules**: Apply LoRA to `q_proj` and `v_proj` in Self-Attention layers. These layers are most sensitive to semantic shifts and domain-specific terminology.
*   **Sequence Length ($T$)**: 
    *   For **Generative Tasks**: Use the generated response tokens. Limit $T$ to a reasonable length (e.g., 50-100 tokens) to balance adaptation signal and computation.
    *   For **Understanding Tasks**: You can compute entropy on the **input prompt** tokens (treating it as a denoising objective). This is often more stable as the "ground truth" input is fixed.
*   **Learning Rate**: Use a very small learning rate (e.g., $1e-5$ to $1e-6$). Since entropy minimization can lead to confident but wrong predictions (collapse), small steps ensure gradual adaptation.
*   **Preventing Collapse**: 
    *   Pure entropy minimization risks the model predicting the same high-frequency token repeatedly. 
    *   **Mitigation**: Ensure the LoRA rank $r$ is small (e.g., 8 or 16) to constrain the update space. 
    *   **Optional**: Add a small diversity regularizer if collapse is observed, though this deviates slightly from pure TENT.
*   **Batch Size**: If memory allows, use a small batch size ($B > 1$) to stabilize the gradient estimate of the entropy. If $B=1$, consider gradient accumulation.
*   **Optimizer**: AdamW is recommended over SGD for LoRA updates due to its adaptive learning rate properties, which help in navigating the sparse gradient landscape of entropy loss.

## References
*   Wang, D., et al. "Tent: Fully test-time adaptation by entropy minimization." ICLR 2021. (Core Algorithm)
*   Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. (Parameter Efficiency)
*   Hu, J., et al. "Test-Time Learning for Large Language Models." ICML 2025. (Context for LLM Adaptation)