# DiT Paper Notes (Peebles & Xie, 2023)

## Key Concepts
1. **Diffusion Transformer (DiT):** Replaces the standard U-Net backbone in latent diffusion models with a pure Vision Transformer architecture.
2. **Scaling Laws:** DiT follows standard Transformer scaling laws: higher compute (FLOPs) consistently leads to lower FID and higher sample quality.
3. **Conditioning Mechanisms Comparison:**
   - *In-context conditioning:* Concatenating tokens (least effective).
   - *Cross-attention:* Effective but adds parameter and memory overhead.
   - *Adaptive LayerNorm (adaLN-Zero):* Modulating scale and shift parameters of LayerNorm via an affine projection of conditioning embeddings, with residual gating initialized to zero (highest efficiency and best performance in class-conditioned benchmarks).
