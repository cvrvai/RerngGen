# Lab 01: Baseline DiT (Diffusion Transformer)

## Mission
Implement, understand, test, and audit a minimal, pure Diffusion Transformer (DiT) architecture from scratch with zero boilerplate bloat.

## Architecture Lineage
`01_dit (Baseline)` $\to$ `02_mmdit` $\to$ `03_flux` $\to$ `04_sana` $\to$ `05_lumina` $\to$ `RerngGen Core`

## Core Mechanism
- **Input:** Latent representation $z \in \mathbb{R}^{B \times C \times H \times W}$ from a frozen VAE.
- **Tokenization:** Patchify via $P \times P$ spatial patches projected to dimension $D$.
- **Positional Encoding:** Fixed 2D sinusoidal grid embeddings.
- **Conditioning:** Continuous sinusoidal timestep MLP combined with pooled text embedding.
- **Modulation:** Adaptive Layer Normalization (`adaLN-Zero`) modulating LayerNorm parameters and residual scaling gates.
- **Attention:** Multi-Head Self-Attention (standard / PyTorch SDPA).
- **Output:** Final adaLN + Linear projection unpatchified back to latent space $\mathbb{R}^{B \times C \times H \times W}$.
- **Objective:** Conditional Flow Matching (CFM) vector field regression outside the model.
