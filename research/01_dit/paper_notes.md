# DiT Paper Notes (Peebles & Xie, 2023)

## Key Concepts
1. **Diffusion Transformer (DiT):** Replaces the standard U-Net backbone in latent diffusion models with a pure Vision Transformer architecture.
2. **Permutation Equivariance:** Self-attention without positional information is strictly **permutation equivariant** ($f(\pi(X)) = \pi(f(X))$); fixed 2D sinusoidal grid embeddings break this permutation symmetry and inject spatial coordinate geometry.
3. **Scaling Laws:** DiT follows standard Transformer scaling laws: higher compute (FLOPs) consistently leads to lower FID and higher sample quality.
4. **Conditioning Mechanisms Comparison:**
   - *In-context conditioning:* Concatenating tokens (least effective).
   - *Cross-attention:* Effective but adds parameter and memory overhead.
   - *Adaptive LayerNorm (adaLN-Zero):* Modulating scale ($\gamma$) and shift ($\beta$) parameters of normalized activations, with residual gating ($\alpha$) initialized to zero (highest efficiency and best performance in benchmarks).
5. **Activation Modulation vs. Static Weights:**
   The timestep conditioning embedding $\mathbf{c}_{\text{time}} \in \mathbb{R}^{B \times D}$ does **not** dynamically mutate or rewrite the learned Transformer block weights ($W_q, W_k, W_v, W_o, W_{\text{mlp}}$). Instead, $\mathbf{c}_{\text{time}}$ linearly generates per-sample activation shifts ($\beta_1, \beta_2$), activation scales ($\gamma_1, \gamma_2$), and residual gate multipliers ($\alpha_1, \alpha_2$) that modulate the normalized feature activations:
   $$\text{Modulate}(\text{LN}(\mathbf{x}), \gamma, \beta) = \text{LN}(\mathbf{x}) \odot (1 + \gamma) + \beta$$
   All Transformer layer weights and biases remain static parameters updated exclusively via gradient descent.
6. **Continuous Time Scaling:**
   Normalized flow time $t \in [0, 1]$ is scaled by a factor `time_scale = 1000.0` before computing sinusoidal frequency bands to ensure rich phase dynamics across the unit interval while keeping the external flow objective strictly in $[0, 1]$.
7. **Why We Use Multiple Heads:**
   Dividing the hidden representation space $D$ into $H$ parallel subspaces allows the model to concurrently attend to multiple types of relationships. Different heads can learn different patterns (e.g., local spatial structure, long-range dependencies, semantic similarity, geometric alignments, or textural continuity). These are illustrative possibilities of learned representations; no particular head is guaranteed to represent a specific human-interpretable concept.
8. **PyTorch SDPA Backends & Computational Complexity:**
   Conceptually, the full attention matrix is $\mathbf{A} \in \mathbb{R}^{B \times H \times N \times N}$.
   PyTorch `scaled_dot_product_attention` dynamically dispatches to available backends:
   - **FlashAttention Backend:** Can avoid materializing the full $N \times N$ attention matrix in GPU HBM by computing tiled attention in fast on-chip SRAM with online softmax.
   - **Memory-Efficient Backend:** Reduces peak attention matrix storage via chunking/tiling.
   - **Math / Reference Backend:** May explicitly construct larger intermediate $N \times N$ tensors.
   
   **Complexity Note:** Standard dense attention remains fundamentally quadratic in sequence length: $\mathcal{O}(N^2 \cdot d)$. Optimized kernels significantly improve memory traffic, cache locality, and runtime throughput, but do not alter the quadratic compute nature of full pairwise attention.
9. **MLP Intermediate Capacity & Parameter Share:**
   - The $4\times$ expansion ($384 \to 1536$) is an architectural design choice that provides additional intermediate feature capacity for non-linear transformations before projecting back to the model dimension. A $384 \to 384$ MLP is not inherently invalid or an information bottleneck, but the wider hidden layer provides significantly greater representational expressiveness.
   - In terms of parameters, the MLP contains approximately $2\times$ the parameters of Self-Attention ($1{,}181{,}568$ vs $591{,}360$ params, differing slightly from exact $2\times$ due to bias counts), accounting for approximately two-thirds ($\approx 66.6\%$) of the core attention+MLP capacity.
10. **adaLN-Zero Conditioning & Zero-Init Invariant:**
    - The conditioning vector $\mathbf{c} \in \mathbb{R}^{B \times D}$ projects linearly via $\text{SiLU} \to \text{Linear}(D \to 6D)$ to yield 6 modulation vectors: $(\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) \in \mathbb{R}^{B \times D}$.
    - Modulation formula on normalized activations:
      $$\text{Modulate}(\text{LN}(\mathbf{x}), \beta, \gamma) = \text{LN}(\mathbf{x}) \odot (1 + \gamma) + \beta$$
    - The gating parameters $\alpha_1, \alpha_2$ scale the residual outputs of the Attention and MLP branches.
    - **Strict Zero-Initialization:** The final modulation linear projection is strictly initialized with zero weights and zero biases. At step 0, all shifts, scales, and residual gates evaluate to exactly 0:
      $$\beta = 0, \; \gamma = 0 \implies \text{Modulate}(\text{LN}(\mathbf{x}), 0, 0) = \text{LN}(\mathbf{x})$$
      $$\alpha = 0 \implies \mathbf{x}_{\text{out}} = \mathbf{x} + 0 \cdot h = \mathbf{x}$$
      Thus, every DiT block begins training as an exact identity function, which helps stabilize early optimization and signal propagation across deep stacks.
11. **Final Layer Head & Output Zero-Initialization:**
    - Maps the contextual token representations $[B, N, D]$ through a conditioned LayerNorm ($\text{shift}_{\text{final}}, \text{scale}_{\text{final}}$ from $\mathbf{c}$) followed by a single velocity-prediction output projection $\text{Linear}(D \to P^2 \cdot C_{\text{out}})$.
    - **Why zero-initialize the FinalLayer?** Zero-initialization is a deliberate DiT baseline initialization strategy that starts the model with zero velocity predictions ($v_\theta \equiv \mathbf{0}$). This prevents arbitrary large random initial outputs before learning begins and gives the model a simple, stable starting point, helping early optimization stability. It is an optimization and stability choice, not a strict mathematical requirement of the Flow Matching objective.
12. **Frozen Baseline Flow Matching Objective:**
    - **Linear Interpolation Path:** $\mathbf{x}_t = (1 - t)\mathbf{x}_0 + t\mathbf{x}_1$ connecting standard Gaussian noise $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ ($t=0$) to clean data latent $\mathbf{x}_1 \sim p_{\text{data}}(\mathbf{x})$ ($t=1$). No $\sigma_{\text{min}}$ parameter is used in the frozen baseline.
    - **Pathwise Teacher Target vs. Inference:** $\mathbf{v}_{\text{target}} = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0 = \mathbf{x}_{\text{data}} - \mathbf{x}_{\text{noise}}$ is the pathwise teacher velocity for a paired sample $(\mathbf{x}_0, \mathbf{x}_1)$. At inference, $\mathbf{x}_1$ is completely unavailable; the network receives only $(\mathbf{x}_t, t, \mathbf{y}_{\text{text}})$ and predicts a conditional vector field $\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{y}_{\text{text}})$.
    - **Flow Matching Loss:** $\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \| \mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{y}_{\text{text}}) - (\mathbf{x}_1 - \mathbf{x}_0) \|_2^2 \right]$.
13. **Checkpoint Save & Resume Integrity:**
    - A true training checkpoint encapsulates the complete experimental state: model parameters ($\theta$), optimizer state ($m_t, v_t$), global step index, configuration dictionary, and full RNG states (Python, PyTorch CPU, PyTorch CUDA).
    - **Atomic Replacement Semantics:** Checkpoints are serialized to a temporary file in the destination directory and committed using `os.replace`. Atomic replacement prevents the primary checkpoint path from being exposed to a partially written serialization under normal process failure, provided the temporary file and destination are on the same filesystem. Atomic replacement does not, by itself, guarantee persistence through sudden hardware/power loss (unless explicit fsync is called).
    - **Future-Facing State & Trust:** Any stateful components introduced in future experiments (e.g. LR schedulers, AMP `GradScaler`, EMA weights, distributed samplers) must be incorporated into checkpoint serialization for exact continuation. Deserialization via unrestricted `torch.load` must only be performed on trusted, project-produced artifacts.
14. **Deterministic Euler ODE Sampler ($t=0 \to t=1$):**
    - Under the frozen Flow Matching convention ($t=0$ noise $\to t=1$ data), the learned velocity field $\mathbf{v}_\theta(\mathbf{x}_t, t)$ points in the direction of increasing time $t$.
    - The generative ODE $\frac{d\mathbf{x}}{dt} = \mathbf{v}_\theta(\mathbf{x}(t), t)$ is discretized with step size $\Delta t = \frac{1}{N}$ forward in time:
      $$\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t \cdot \mathbf{v}_\theta(\mathbf{x}_k, t_k, \mathbf{y}_{\text{text}})$$
    - The update uses **$+ \Delta t \cdot \mathbf{v}_\theta$** (moving from noise toward data). Using $-\Delta t \cdot \mathbf{v}$ would integrate in reverse, converting data back into noise.
