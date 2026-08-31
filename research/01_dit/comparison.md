# Architecture Comparison Matrix

| Property | Standard U-Net (SD 1.5) | Baseline DiT (Peebles & Xie) | MMDiT (SD3) | FLUX.1 |
| :--- | :--- | :--- | :--- | :--- |
| **Token Representation** | 2D Spatial Convolutions | 1D Sequence of Patches | Multimodal Tokens | Joint Stream $\to$ Single Stream |
| **Conditioning Mode** | Cross-Attention + ResNet add | adaLN-Zero | Joint Self-Attention | Joint + Double Stream Attention |
| **Position Encoding** | Implicit (Conv padding) | 2D Sinusoidal / Learned | 2D RoPE / Sinusoidal | 2D RoPE |
| **Residual Gating** | 1.0 (Fixed add) | Zero-init $\alpha \odot h$ | Zero-init gating | Modulated residual |
