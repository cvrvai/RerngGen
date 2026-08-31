# Baseline DiT Architecture Specification

## Overview
The baseline Diffusion Transformer operates directly in the latent space of a pretrained, frozen Variational Autoencoder (VAE).

```
                    Latent Tensor [B, C_latent, H_latent, W_latent]
                                       │
                                       ▼
                       Patch Embedding (Conv2d / Linear)
                                       │
                                       ▼
                            Tokens [B, N, D]
                                       │
                      + 2D Sinusoidal Position Embedding
                                       │
                                       ▼
Timestep t ──► Sinusoidal MLP ──► c_time ──┐
Text prompt ──► Pooled Projection ──► c_text ──┴─► c = c_time + c_text [B, D]
                                                    │
                 ┌──────────────────────────────────┤
                 ▼                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │                   DiT Block (x Depth)                  │
      │                                                        │
      │   c ──► Linear ──► (gamma_1, beta_1, alpha_1,          │
      │                     gamma_2, beta_2, alpha_2)          │
      │                                                        │
      │   h1 = SelfAttention( Modulate(LN(x), gamma_1, beta_1) )│
      │   x = x + alpha_1 * h1                                 │
      │                                                        │
      │   h2 = MLP( Modulate(LN(x), gamma_2, beta_2) )         │
      │   x = x + alpha_2 * h2                                 │
      └────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                Final Layer (Conditioned LN + Linear Head)
                                       │
                                       ▼
                       Unpatchify [B, C_latent, H_latent, W_latent]
                                       │
                                       ▼
                             Predicted Velocity v
```

## Key Invariants
1. **Zero-Initialization:** The residual scaling factors $\alpha_1, \alpha_2$ and the weights/biases of the final linear projection are initialized to 0. At initialization, each DiT block acts as an identity function.
2. **Outside Objective:** Flow-matching target generation, interpolation, and loss computation are strictly isolated outside the model.
3. **Configuration Driven:** All dimensions ($C, H, W, P, D, L, H$) are parameterizable via YAML.
