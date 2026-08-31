"""Single Diffusion Transformer (DiT) Block.

Combines Adaptive Layer Normalization (adaLN-Zero), Multi-Head Self-Attention,
and Pointwise MLP with zero-initialized residual gating.
"""

import torch
import torch.nn as nn
from src.adaln import AdaLNZero, modulate
from src.attention import Attention
from src.mlp import Mlp


class DiTBlock(nn.Module):
    """Diffusion Transformer Block.

    Architecture:
        x [B, N, D], c [B, D]
        1. (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = adaLN(c)
        2. x = x + gate_msa * Attention( Modulate( LN1(x), shift_msa, scale_msa ) )
        3. x = x + gate_mlp * MLP( Modulate( LN2(x), shift_mlp, scale_mlp ) )

    Args:
        hidden_size (int): Transformer token embedding dimension D (default: 384).
        num_heads (int): Number of parallel self-attention heads H (default: 6).
        mlp_ratio (float): Hidden expansion ratio for the MLP (default: 4.0).
        qkv_bias (bool): Whether to include bias in QKV projection (default: True).
        eps (float): Epsilon for LayerNorm (default: 1e-6).
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # Layer Normalizations without learnable static affine parameters (adaLN provides dynamic affine)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

        # Core sub-blocks
        self.attn = Attention(hidden_size=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=nn.GELU(approximate="tanh"),
        )

        # adaLN-Zero modulation producing 6 vectors [B, D] with strict zero-init
        self.adaLN_modulation = AdaLNZero(hidden_size=hidden_size, num_modulations=6)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input token sequence [B, N, D].
            c (torch.Tensor): Conditioning vector [B, D].

        Returns:
            torch.Tensor: Output token sequence [B, N, D].
        """
        B, N, D = x.shape
        if D != self.hidden_size:
            raise ValueError(
                f"Expected input feature dimension {self.hidden_size}, but got {D} (x shape: {x.shape})."
            )
        if c.shape != (B, self.hidden_size):
            raise ValueError(
                f"Expected condition vector shape {(B, self.hidden_size)}, but got {c.shape}."
            )

        # 1. Compute 6 modulation vectors from conditioning vector c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)

        # 2. Attention Sub-Block with Modulated LayerNorm and Residual Gate
        # Modulate(LN1(x), shift_msa, scale_msa) -> Attention -> gate_msa * Out -> Residual Add
        norm_x1 = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.attn(norm_x1)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 3. MLP Sub-Block with Modulated LayerNorm and Residual Gate
        # Modulate(LN2(x), shift_mlp, scale_mlp) -> MLP -> gate_mlp * Out -> Residual Add
        norm_x2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(norm_x2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out

        return x
