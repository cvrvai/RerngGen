# Differences from Previous Architectures

## DiT vs Traditional Latent U-Net (e.g., Stable Diffusion 1.5/2.1)
1. **Homogeneous Backbone:** DiT replaces multi-scale downsampling/upsampling and cross-resolution skip connections with a constant token dimension $D$ across all layers.
2. **Patch Tokenization:** Latents are divided into non-overlapping patches $P \times P$, matching Vision Transformer (ViT) dynamics.
3. **Conditioning:** adaLN-Zero scales and shifts normalized states inside each block directly, eliminating complex spatial cross-attention layers for global conditioning.
4. **Compute Uniformity:** Every block performs identical operations on sequence length $N$, simplifying memory profiling, tensor parallelization, and kernel optimization.
