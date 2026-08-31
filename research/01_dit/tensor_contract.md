# Baseline DiT Tensor Contracts

## 1. Input Specifications
- **Latent Tensor:** $x \in \mathbb{R}^{B \times C_{\text{in}} \times H_{\text{latent}} \times W_{\text{latent}}}$
  - For standard SD-style VAE: $C_{\text{in}} = 4$.
  - Spatial resolution: $H_{\text{latent}} = H_{\text{pixel}} / 8$, $W_{\text{latent}} = W_{\text{pixel}} / 8$.
  - Example: $256 \times 256$ RGB image $\to 32 \times 32$ latent.
- **Timestep:** $t \in \mathbb{R}^{B}$ (continuous values in $[0, 1]$ for Flow Matching or $[0, 1000]$ for discrete diffusion).
- **Conditioning Vector (Optional/v0):** $y \in \mathbb{R}^{B \times D_{\text{text}}}$ (e.g., pooled CLIP/T5 text embedding) or class labels $y \in \mathbb{Z}^{B}$.

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
