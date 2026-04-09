# Skill: COME-LLM (Conservative Test-Time Adaptation for LLMs)

## Description
This skill implements the **COME (Conservatively Minimizing Entropy)** algorithm adapted for **Large Language Models (LLMs)**. It addresses the "overconfidence" issue of standard Entropy Minimization (EM) during Test-Time Adaptation (TTA). 

Unlike traditional EM which blindly minimizes prediction entropy (leading to collapse on OOD data), COME-LLM explicitly models **uncertainty** using **Subjective Logic**. It encourages the model to be confident only when there is sufficient evidence (high logit norms) and allows it to express "I don't know" (high uncertainty mass) for ambiguous or OOD inputs. This is achieved by minimizing the **Opinion Entropy** while constraining the deviation of uncertainty from the pre-trained baseline via **Logit Norm Regularization**.

## Core Principles
1.  **Subjective Logic for Uncertainty**: Instead of treating Softmax probabilities as ground truth confidence, COME interprets logits as **evidence** for a Dirichlet distribution. This decomposes prediction into **Belief Mass** ($b$) and **Uncertainty Mass** ($u$).
2.  **Conservative Entropy Minimization**: The objective is to minimize the entropy of the *opinion* (belief + uncertainty), not just the class probability. This prevents the model from forcing high confidence on unreliable samples.
3.  **Logit Norm Constraint**: To prevent the uncertainty mass $u$ from collapsing to zero (overconfidence) or diverging wildly, COME constrains the $L_p$ norm of the logits. This acts as a regularizer that keeps the adapted model's "certainty level" consistent with its pre-trained state unless strong evidence suggests otherwise.
4.  **Parameter-Efficient Adaptation**: Uses **LoRA** to update only a small subset of parameters, ensuring efficient test-time learning without catastrophic forgetting.

## Mathematical Formulation

### 1. Subjective Opinion Modeling (Per-Token)
For a given input context, at generation step $t$, let $z_t \in \mathbb{R}^{|V|}$ be the logits output by the LLM for the vocabulary $V$. 

We transform logits into **evidence** $e_t$ and Dirichlet parameters $\alpha_t$:
$$
e_t = \exp(z_t) \quad (\text{element-wise})
$$
$$
\alpha_t = e_t + 1
$$

The **strength** of the Dirichlet distribution $S_t$ is:
$$
S_t = \sum_{v \in V} \alpha_{t,v} = \sum_{v \in V} (e_{t,v} + 1) = |V| + \sum_{v \in V} e_{t,v}
$$

The **Belief Mass** $b_{t,v}$ for each token $v$ and **Uncertainty Mass** $u_t$ are:
$$
b_{t,v} = \frac{e_{t,v}}{S_t}
$$
$$
u_t = \frac{|V|}{S_t}
$$
Note: $\sum_{v} b_{t,v} + u_t = 1$.

### 2. Opinion Entropy Objective
The entropy of the subjective opinion $M_t = [b_{t,1}, ..., b_{t,|V|}, u_t]$ is defined as:
$$
H(M_t) = - \left( \sum_{v \in V} b_{t,v} \log(b_{t,v} + \epsilon) + u_t \log(u_t + \epsilon) \right)
$$

The total loss for a generated sequence of length $T$ is the mean opinion entropy:
$$
\mathcal{L}_{COME} = \frac{1}{T} \sum_{t=1}^{T} H(M_t)
$$

### 3. Conservative Constraint via Logit Norm
To ensure the uncertainty estimation remains conservative and does not diverge significantly from the pre-trained model's behavior, we constrain the magnitude of the logits. 

We decompose the logits $z_t$ into direction and magnitude:
$$
\hat{z}_t = \frac{z_t}{\|z_t\|_p} \cdot \|z_t\|_p^{\text{detach}} \cdot \tau
$$
Where:
*   $\|z_t\|_p$ is the $L_p$ norm of the logits (typically $p=2$).
*   $\|z_t\|_p^{\text{detach}}$ is the norm with gradients stopped (acting as a constant reference from the current forward pass).
*   $\tau$ is a hyperparameter controlling the tightness of the constraint (default $\tau=1$).

In practice, this is implemented by normalizing the logits used for evidence calculation:
$$
z_t^{\text{come}} = \tau \cdot \|z_t\|_p^{\text{detach}} \cdot \frac{z_t}{\|z_t\|_p}
$$
Then, compute evidence $e_t = \exp(z_t^{\text{come}})$. This ensures that the optimization of $\mathcal{L}_{COME}$ adjusts the *direction* of the logits (relative probabilities) while keeping the *magnitude* (overall confidence/uncertainty level) stable relative to the current state.

### 4. Parameter Update (LoRA)
Only LoRA parameters $\Theta_{LoRA}$ are updated:
$$
\Theta_{LoRA} \leftarrow \Theta_{LoRA} - \eta \nabla_{\Theta_{LoRA}} \mathcal{L}_{COME}
$$

## Algorithm Workflow

1.  **Initialization**:
    *   Load Pre-trained LLM.
    *   Inject LoRA adapters into Attention layers ($W_q, W_v$).
    *   Freeze backbone weights.

2.  **Test-Time Adaptation Loop** (for each unlabeled test prompt $x$):
    a.  **Generation & Forward Pass**:
        *   Generate a sequence of tokens $y = \{y_1, ..., y_T\}$ (or use a fixed length rollout).
        *   At each step $t$, obtain logits $z_t$ from the model.
    
    b.  **Logit Constraint Application**:
        *   Compute $L_p$ norm of $z_t$.
        *   Apply the conservative transformation: $z_t^{\text{come}} = \tau \cdot \|z_t\|_p^{\text{detach}} \cdot \frac{z_t}{\|z_t\|_p}$.

    c.  **Opinion Calculation**:
        *   Compute evidence $e_t = \exp(z_t^{\text{come}})$.
        *   Compute Strength $S_t$, Belief $b_t$, and Uncertainty $u_t$.

    d.  **Loss Computation**:
        *   Calculate Opinion Entropy $H(M_t)$ for each step.
        *   Aggregate: $\mathcal{L}_{COME} = \text{Mean}(H(M_t))$.

    e.  **Backward & Update**:
        *   Compute gradients w.r.t LoRA parameters.
        *   Update LoRA weights.

3.  **Inference**:
    *   Use the updated LoRA parameters for subsequent generations.

## Pseudo-code

```python
Algorithm: COME-LLM (Conservative Test-Time Adaptation with LoRA)

Input: 
    - Pre-trained LLM f_theta with frozen weights theta
    - Unlabeled test prompts D_test = {x_j}
    - LoRA rank r, learning rate eta
    - Hyperparameters: p (norm type, default 2), tau (scale, default 1), T (gen length)
    - Optimizer Opt (e.g., AdamW)

Output: 
    - Adapted LLM with updated LoRA parameters Theta_LoRA

1: // Initialization
2: Inject LoRA adapters into LLM attention layers
3: Define trainable parameters Theta_LoRA = {A_l, B_l}
4: Freeze all other parameters in theta
5: Initialize Opt with Theta_LoRA and learning rate eta

6: // Test-Time Adaptation Loop
7: For each prompt x in D_test do:

8:     Initialize sequence y = []
9:     Initialize total_loss = 0
    
10:    // Generate T tokens and accumulate loss
11:    For t = 1 to T do:
        
12:        // Forward pass to get logits for next token
13:        Set LoRA modules to train() mode
14:        Logits_z = f_theta(x, y_prev, Theta_LoRA) // Shape: [1, Vocab]
        
15:        // --- COME Core: Conservative Logit Transformation ---
16:        // Calculate L_p norm of current logits
17:        Norm_z = Norm(Logits_z, p=p)
        
18:        // Detach norm to stop gradient flow through magnitude
19:        Norm_z_detached = Norm_z.detach()
        
20:        // Normalize logits to unit direction, then scale by detached norm * tau
21:        Logits_come = tau * Norm_z_detached * (Logits_z / (Norm_z + 1e-8))
        
22:        // --- Subjective Logic Opinion Calculation ---
23:        // Convert constrained logits to evidence
24:        Evidence_e = exp(Logits_come)
        
25:        // Calculate Dirichlet Strength S
26:        S = sum(Evidence_e) + Vocab_Size
        
27:        // Calculate Belief Mass b and Uncertainty Mass u
28:        Belief_b = Evidence_e / S
29:        Uncertainty_u = Vocab_Size / S
        
30:        // --- Opinion Entropy Loss ---
31:        // H(M) = - sum(b * log(b)) - u * log(u)
32:        Entropy_belief = -sum(Belief_b * log(Belief_b + 1e-8))
33:        Entropy_uncert = -Uncertainty_u * log(Uncertainty_u + 1e-8)
34:        H_opinion = Entropy_belief + Entropy_uncert
        
35:        total_loss = total_loss + H_opinion
        
36:        // Sample next token (for autoregressive generation context)
37:        Probs = Softmax(Logits_z) // Use original logits for sampling stability
38:        Next_Token = Sample(Probs)
39:        Append Next_Token to y
        
40:    End For

41:    // Average loss over sequence length
42:    Loss_COME = total_loss / T

43:    // Backward Pass and Update
44:    Opt.zero_grad()
45:    Loss_COME.backward()
46:    Opt.step()

47: End For

48: Return LLM with updated Theta_LoRA
```

## Implementation Details & Best Practices

*   **Vocabulary Size Handling**: The term $|V|$ (Vocab Size) in the uncertainty calculation $u_t = |V|/S_t$ is crucial. For large vocabularies (e.g., 32,000+), $S_t$ will be large, making $u_t$ small if evidence is high. Ensure numerical stability with `1e-8` epsilons.
*   **Norm Type ($p$)**: $p=2$ (Euclidean norm) is recommended as a balance between strictness and smoothness. $p=\infty$ (Max norm) is stricter but may hinder adaptation on reliable samples.
*   **Tau ($\tau$)**: 
    *   $\tau=1$: Standard conservative adaptation.
    *   $\tau < 1$: More conservative (forces lower confidence/higher uncertainty).
    *   $\tau > 1$: Less conservative (allows higher confidence).
    *   Start with $\tau=1$.
*   **Generation Length ($T$)**: 
    *   For **Adaptation**, you don't necessarily need to generate a full answer. A short rollout (e.g., $T=10-20$ tokens) is often sufficient to capture the domain's linguistic style and terminology for adaptation.
    *   Longer $T$ provides more signal but increases computation.
*   **Sampling vs. Greedy**: During the adaptation forward pass, you must generate tokens to create the context for the next step. Using **Top-K Sampling** or **Nucleus Sampling** is better than Greedy Decoding for exploration, but ensure the *loss calculation* uses the logits from the actual path taken.
*   **LoRA Target Modules**: Apply to `q_proj` and `v_proj`. These modules control the attention mechanism's focus and are highly sensitive to domain shifts.
*   **Preventing Collapse**: The core benefit of COME is preventing collapse. If you observe the model generating repetitive tokens, check if $\tau$ is too low or if the learning rate is too high. The logit norm constraint should naturally prevent the logits from exploding (which causes $u \to 0$ and $b \to 1$ artificially).

## References
*   Zhang, Q., Bian, Y., Kong, X., Zhao, P., & Zhang, C. (2025). "COME: Test-Time Adaption by Conservatively Minimizing Entropy." *ICLR 2025*.
*   Hu, E. J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
*   Jøsang, A. "Subjective Logic: A formalism for reasoning under uncertainty." *Springer 2018*.