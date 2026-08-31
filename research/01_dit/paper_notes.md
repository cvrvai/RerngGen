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
   The timestep conditioning embedding $\mathbf{c}_{\text{time}} \in \mathbb{R}^{B \times D}$ does **not** dynamically change or mutate the learned Transformer block weights ($W_q, W_k, W_v, W_o, W_{\text{mlp}}$). Instead, $\mathbf{c}$ linearly generates per-sample activation shifts ($\beta_1, \beta_2$), activation scales ($\gamma_1, \gamma_2$), and residual gate multipliers ($\alpha_1, \alpha_2$) that modulate the normalized feature activations:
   $$\text{Modulate}(\text{LN}(\mathbf{x}), \gamma, \beta) = \text{LN}(\mathbf{x}) \odot (1 + \gamma) + \beta$$
   All Transformer layer weights remain static parameters updated exclusively via gradient descent.
6. **Continuous Time Scaling:**
   Normalized flow time $t \in [0, 1]$ is scaled by a factor `time_scale = 1000.0` before computing sinusoidal frequency bands to ensure rich phase dynamics across the unit interval while keeping the external flow objective strictly in $[0, 1]$.
