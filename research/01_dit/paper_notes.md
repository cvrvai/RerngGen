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
      Thus, every DiT block begins training as an exact identity function, stabilizing forward/backward signal propagation across deep stacks.
