# Skill: Test-Time Learning for LLMs (TLM)

## Description
This skill implements the **Test-Time Learning (TLM)** algorithm, also known as **TLM**, designed to dynamically adapt Large Language Models (LLMs) to target domains using only unlabeled test data. It addresses distribution shifts (e.g., domain-specific terminology, linguistic variations) by minimizing input perplexity, employing a sample-efficient learning strategy, and utilizing Low-Rank Adaptation (LoRA) to prevent catastrophic forgetting.

## Core Principles
1.  **Input Perplexity Minimization**: Instead of output entropy minimization (common in TTA), TLM minimizes the perplexity of the *input* sequence $x$. Theoretical and empirical evidence shows that minimizing input perplexity $P(x; \Theta)$ correlates with reduced output perplexity $P(y|x; \Theta)$, improving generation quality.
2.  **Sample-Efficient Learning**: Not all test samples are equally useful. High-perplexity samples contain more information for adaptation. TLM actively selects and weights these samples.
3.  **LoRA-based Adaptation**: To mitigate catastrophic forgetting and reduce computational cost, only LoRA parameters are updated during test time, keeping the pre-trained weights frozen.

## Mathematical Formulation

### 1. Perplexity Definition
For a token sequence $x = \{x_1, x_2, ..., x_T\}$, the perplexity under model parameters $\Theta$ is:
$$
P(x; \Theta) = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log p(x_t | x_{1:t-1}; \Theta) \right)
$$

### 2. Optimization Objective
The goal is to minimize the weighted input perplexity over the test batch. Let $\tilde{\Theta} = \Theta + \Delta\Theta$ be the adapted parameters, where $\Theta$ are frozen pre-trained weights and $\Delta\Theta$ are LoRA parameters.

The loss function for a single sample $x$ is:
$$
\mathcal{L}(x; \tilde{\Theta}) = S(x) \cdot P(x; \tilde{\Theta})
$$

Where $S(x)$ is the sample selection score (weight).

### 3. Sample Selection Score $S(x)$
To prioritize informative (high-perplexity) samples and ignore well-modeled (low-perplexity) ones, we define:
$$
S(x) = \lambda \cdot \exp\left( \log P(x; \tilde{\Theta}) - \log P_0 \right) \cdot \mathbb{I}\{P(x; \tilde{\Theta}) > P_0\}
$$
Simplifying the exponential term:
$$
S(x) = \lambda \cdot \frac{P(x; \tilde{\Theta})}{P_0} \cdot \mathbb{I}\{P(x; \tilde{\Theta}) > P_0\}
$$

*   $\lambda$: Scaling factor (hyperparameter, e.g., 0.1).
*   $P_0$: Perplexity threshold (hyperparameter, e.g., $e^3$).
*   $\mathbb{I}\{\cdot\}$: Indicator function. If $P(x) \le P_0$, the sample is skipped ($S(x)=0$).

### 4. Parameter Update (LoRA)
LoRA decomposes the weight update $\Delta W$ for a specific layer as:
$$
\Delta W = B A
$$
Where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{k \times r}$ are trainable low-rank matrices ($r \ll \min(d, k)$).
*   Initialization: $A$ with Gaussian initialization, $B$ with zeros.
*   Only $A$ and $B$ are updated via gradient descent on $\mathcal{L}(x; \tilde{\Theta})$.

## Algorithm Workflow

1.  **Initialization**: Load pre-trained LLM $\Theta$. Attach LoRA adapters $\Delta\Theta$ (matrices $A, B$) to target layers (e.g., attention $W_q, W_v$). Initialize $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$.
2.  **Test-Time Loop**: For each batch of unlabeled test data $\mathcal{X}_{batch} = \{x_i\}_{i=1}^B$:
    a.  **Forward Pass**: Compute logits and perplexity $P(x_i; \tilde{\Theta})$ for each $x_i$ in the batch.
    b.  **Sample Selection**: Calculate weights $S(x_i)$ using the threshold $P_0$.
    c.  **Loss Computation**: Compute weighted loss $\mathcal{L}_{batch} = \frac{1}{B} \sum_{i=1}^B S(x_i) P(x_i; \tilde{\Theta})$.
    d.  **Backward Pass**: Compute gradients $\nabla_{\Delta\Theta} \mathcal{L}_{batch}$.
    e.  **Update**: Update LoRA parameters $\Delta\Theta$ using an optimizer (e.g., AdamW).
3.  **Inference**: Use the updated model $\tilde{\Theta}$ to generate responses for the current batch or subsequent batches.

## Pseudo-code

```python
Algorithm: Test-Time Learning for LLMs (TLM)

Input: 
    - Pre-trained LLM parameters Θ (frozen)
    - Unlabeled test dataset D_test = {x_j}_{j=1}^M
    - LoRA rank r, learning rate η, batch size B
    - Hyperparameters λ, P_0
    - Optimizer Opt (e.g., AdamW)

Output: 
    - Adapted LLM parameters Θ̃ = Θ + ΔΘ

1: Initialize LoRA parameters ΔΘ:
   For each target layer:
      A ~ N(0, σ²)
      B = 0
   Θ̃ ← Θ + ΔΘ

2: For each batch X = {x_b}_{b=1}^B in D_test do:

3:   // Step 1: Calculate Perplexity and Weights
4:   Initialize batch_loss ← 0
5:   For each sample x in X do:
6:      // Compute input perplexity P(x; Θ̃)
7:      log_probs ← LogSoftmax(LLM_Forward(x, Θ̃))
8:      nll ← -Mean(log_probs corresponding to input tokens)
9:      P_x ← Exp(nll)
      
10:     // Calculate Sample Selection Score S(x)
11:     If P_x > P_0 then:
12:        S_x ← λ * (P_x / P_0)
13:     Else:
14:        S_x ← 0
15:     End If
      
16:     // Accumulate weighted loss
17:     batch_loss ← batch_loss + S_x * P_x
18:   End For

19:   // Step 2: Optimization
20:   If batch_loss > 0 then:
21:      Loss ← batch_loss / B  // Normalize by batch size
22:      Opt.zero_grad()
23:      Loss.backward()        // Backpropagate only through LoRA params
24:      Opt.step()             // Update A and B
25:   End If

26: End For

27: Return Θ̃
```

## Implementation Details & Best Practices

*   **Target Layers**: Apply LoRA to Query ($W_q$) and Value ($W_v$) projection matrices in attention blocks.
*   **Hyperparameters**:
    *   $\lambda$: Typically set to `0.1`.
    *   $P_0$: Typically set to $e^3 \approx 20.08$. Adjust based on domain complexity.
    *   Learning Rate: Small learning rates are crucial (e.g., $1e-5$ to $5e-5$) to prevent overfitting to single test batches.
*   **Decoder-Only Models**: This method is optimized for autoregressive decoder-only models (e.g., LLaMA, Qwen).
*   **Efficiency**: Samples with $P(x) \le P_0$ skip the backward pass entirely, saving computation.
*   **Catastrophic Forgetting**: By freezing $\Theta$ and only updating low-rank $\Delta\Theta$, the model retains general knowledge while adapting to local distribution shifts.

## References
*   Hu, J., Zhang, Z., Chen, G., et al. "Test-Time Learning for Large Language Models." ICML 2025.
*   Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
