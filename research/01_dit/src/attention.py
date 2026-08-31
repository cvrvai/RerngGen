"""Multi-Head Self-Attention module for Diffusion Transformers (DiT).

Implements standard multi-head self-attention using a single fused QKV linear
projection and PyTorch's scaled_dot_product_attention (SDPA).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Multi-Head Self-Attention.

    Args:
        hidden_size (int): Token embedding dimension D (e.g. 384).
        num_heads (int): Number of parallel attention heads H (e.g. 6).
        qkv_bias (bool): Whether to include bias in the fused QKV projection. Default: True.
        out_bias (bool): Whether to include bias in the output linear projection. Default: True.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_heads: int = 6,
        qkv_bias: bool = True,
        out_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
            )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused QKV projection maps [B, N, D] -> [B, N, 3*D] in a single GEMM operation
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)
        # Final output projection maps concatenated heads back to model space [B, N, D]
        self.proj = nn.Linear(hidden_size, hidden_size, bias=out_bias)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard ViT / DiT Xavier uniform initialization
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for multi-head self-attention.

        Args:
            x (torch.Tensor): Input token sequence of shape [B, N, D].

        Returns:
            torch.Tensor: Attended token sequence of shape [B, N, D].
        """
        B, N, D = x.shape
        if D != self.hidden_size:
            raise ValueError(
                f"Expected input feature dimension {self.hidden_size}, but got {D} (shape: {x.shape})."
            )

        # 1. Fused QKV Projection: [B, N, D] -> [B, N, 3*D]
        qkv = self.qkv(x)

        # 2. Reshape & Split into Q, K, V: [B, N, 3, H, d_head] -> 3 x [B, H, N, d_head]
        # Reshape to separate the 3 tensors and H attention heads
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # Permute to layout: (3, B, H, N, d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each is [B, H, N, d_head]

        # 3. Scaled Dot-Product Attention: [B, H, N, d_head]
        # Computes softmax(Q @ K.T / sqrt(d_head)) @ V via fast fused SDPA kernel
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        # 4. Concatenate Heads: [B, H, N, d_head] -> [B, N, H * d_head] = [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, D)

        # 5. Output Projection: [B, N, D] -> [B, N, D]
        out = self.proj(out)

        return out
