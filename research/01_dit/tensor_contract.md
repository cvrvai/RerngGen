# Baseline DiT Tensor Contracts

## 1. Input Specifications
- **Latent Tensor:** $x \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{latent}} \times W_{\text{latent}}}$
  - For standard SD-style VAE: $C_{\text{in}} = 4$.
  - Spatial resolution: $H_{\text{latent}} = H_{\text{pixel}} / 8$, $W_{\text{latent}} = W_{\text{pixel}} / 8$.
  - Example: $256 \times 256$ RGB image $\to 32 \times 32$ latent.
- **Timestep:** $t \in \mathbb{R}^{B}$ (continuous flow time values in $[0, 1]$).
- **Pooled Text Conditioning Vector (Optional):** $\mathbf{y}_{\text{text}} \in \mathbb{R}^{B \times D}$ (already projected to Transformer model dimension $D$).
  - *Architectural Boundary:* Text encoder feature extraction and dimension projection live upstream of `DiT`. `DiT` accepts $\mathbf{y}_{\text{text}} \in [B, D]$ directly.
  - *Global Condition Contract:* $\mathbf{c} = \mathbf{t}_{\text{embed}} + \mathbf{y}_{\text{text}} \in \mathbb{R}^{B \times D}$. If $\mathbf{y}_{\text{text}}$ is absent (`None`), $\mathbf{c} = \mathbf{t}_{\text{embed}} + \mathbf{0} = \mathbf{t}_{\text{embed}}$.

## 2. Patch Transformation Contract
- **Patch Size:** $P \in \mathbb{Z}^+$ (typically $P=2$).
- **Sequence Length:** $N = (H_{\text{latent}} / P) \times (W_{\text{latent}} / P)$.
- **Patchified Tokens:** $x_{\text{tokens}} \in \mathbb{R}^{B \times N \times D}$ where $D$ is `hidden_size`.
- **Constraint:** $H_{\text{latent}} \pmod P == 0$ and $W_{\text{latent}} \pmod P == 0$.

## 3. Block Invariant
- **Block Input:** $x \in \mathbb{R}^{B \times N \times D}$, conditioning $c \in \mathbb{R}^{B \times D}$.
- **Block Output:** $x_{\text{out}} \in \mathbb{R}^{B \times N \times D}$.
- **Zero-Init Invariant:** At step 0, $\forall x, c: \text{DiTBlock}(x, c) == x$.

## 4. Final Output Head Contract
- **Projected Patches:** $x_{\text{proj}} \in \mathbb{R}^{B \times N \times (P^2 \cdot C_{\text{out}})}$.
- **Unpatchified Latent:** $v_{\text{pred}} \in \mathbb{R}^{B \times C_{\text{out}} \times H_{\text{latent}} \times W_{\text{latent}}}$.
- **Exact Shape Match:** $\text{shape}(v_{\text{pred}}) == \text{shape}(x_{\text{latent}})$.

## 5. Frozen Flow Matching Baseline Objective (Step 11 Contract)
- **Noise Distribution:** $x_0 = x_{\text{noise}} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- **Data Distribution:** $x_1 = x_{\text{data}} \sim p_{\text{data}}(x)$
- **Timestep Sampling:** $t \sim \text{Uniform}(0, 1)$
- **Linear Interpolation Path:**
  $$x_t = (1 - t) x_0 + t x_1$$
- **Target Velocity Field:**
  $$v_{\text{target}} = \frac{d x_t}{d t} = x_1 - x_0 = x_{\text{data}} - x_{\text{noise}}$$
- **Loss Function:**
  $$\mathcal{L}_{\text{FM}}(\theta) = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t, \mathbf{y}_{\text{text}}) - (x_1 - x_0) \|_2^2 \right]$$
- *Rule on Variants:* Any $\sigma_{\text{min}}$, cosine timestep weighting, log-normal sampling, or alternative interpolation path is strictly considered an experimental variant requiring a separate Architecture Decision Record (ADR) and must not modify the frozen v0 baseline.
