"""Adaptive Layer Normalization (adaLN-Zero) for Diffusion Transformers (DiT).

Implements the conditioning modulation mechanism that maps a global condition
vector c [B, D] into per-sample activation shifts (beta), scales (gamma), and
residual gating multipliers (alpha) with strict ZERO-INITIALIZATION.
"""

from typing import Tuple
import torch
import torch.nn as nn


def modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Applies affine scale and shift modulation to normalized activations.

    Formula:
        modulate(x, shift, scale) = x * (1 + scale) + shift

    Args:
        x (torch.Tensor): Normalized activation sequence of shape [B, N, D].
        shift (torch.Tensor): Shift vector beta of shape [B, D].
        scale (torch.Tensor): Scale vector gamma of shape [B, D].

    Returns:
        torch.Tensor: Modulated activations of shape [B, N, D].
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D activation tensor [B, N, D], but got shape {x.shape}.")
    if shift.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"Expected 2D modulation vectors [B, D], but got shift: {shift.shape}, scale: {scale.shape}."
        )

    # Unsqueeze dimension 1 to broadcast across sequence length N
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNZero(nn.Module):
    """Adaptive LayerNorm (adaLN-Zero) Modulation Layer.

    Projects conditioning vector c [B, D] via SiLU -> Linear(D -> num_modulations * D).
    For a standard DiT block, num_modulations = 6:
        - shift_attn (beta_1): [B, D]
        - scale_attn (gamma_1): [B, D]
        - gate_attn  (alpha_1): [B, D]
        - shift_mlp  (beta_2): [B, D]
        - scale_mlp  (gamma_2): [B, D]
        - gate_mlp   (alpha_2): [B, D]

    Args:
        hidden_size (int): Model embedding dimension D (e.g. 384).
        num_modulations (int): Number of modulation vectors to produce (default: 6).
        bias (bool): Whether to include bias in linear projection. Default: True.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        num_modulations: int = 6,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modulations = num_modulations

        self.act = nn.SiLU()
        self.linear = nn.Linear(hidden_size, num_modulations * hidden_size, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        # STRICT ZERO-INIT: Weight and bias initialized to exact zeros
        # Ensures that at step 0: beta=0, gamma=0, alpha=0 (identity residual mapping)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, c: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            c (torch.Tensor): Conditioning vector of shape [B, hidden_size].

        Returns:
            Tuple[torch.Tensor, ...]: Tuple of num_modulations tensors, each of shape [B, hidden_size].
        """
        B, D = c.shape
        if D != self.hidden_size:
            raise ValueError(
                f"Expected conditioning dimension {self.hidden_size}, but got {D} (shape: {c.shape})."
            )

        # 1. Non-linear activation & linear projection: [B, D] -> [B, num_modulations * D]
        params = self.linear(self.act(c))

        # 2. Chunk into individual modulation vectors: num_modulations x [B, D]
        chunks = params.chunk(self.num_modulations, dim=-1)
        return chunks
