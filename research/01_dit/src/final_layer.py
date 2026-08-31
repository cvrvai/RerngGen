"""Final Layer Head for Diffusion Transformers (DiT).

Applies conditioned Layer Normalization (adaLN with shift, scale) followed by
a zero-initialized linear projection to map contextual tokens [B, N, D] into
patch features [B, N, P^2 * C_out] ready for unpatchify.
"""

from typing import Tuple
import torch
import torch.nn as nn
from src.adaln import modulate


class FinalLayer(nn.Module):
    """DiT Final Layer Head.

    Architecture:
        x [B, N, D], c [B, D]
        1. (shift, scale) = adaLN_modulation(c) [B, 2*D] -> 2 x [B, D]
        2. x_norm = Modulate( norm_final(x), shift, scale )
        3. x_out = linear(x_norm) -> [B, N, P^2 * out_channels]

    Args:
        hidden_size (int): Model embedding dimension D (e.g. 384).
        patch_size (int): Spatial patch size P (default: 2).
        out_channels (int): Latent output channels C_out (default: 4).
        eps (float): Epsilon for LayerNorm (default: 1e-6).
    """

    def __init__(
        self,
        hidden_size: int = 384,
        patch_size: int = 2,
        out_channels: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.out_features = patch_size * patch_size * out_channels  # P^2 * C_out (e.g. 2*2*4 = 16)

        # LayerNorm without static affine parameters (scale & shift provided dynamically by adaLN)
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)

        # Linear projection to patch pixel volume
        self.linear = nn.Linear(hidden_size, self.out_features, bias=True)

        # adaLN modulation producing 2 vectors: shift_final, scale_final
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # STRICT ZERO-INIT:
        # 1. Zero-initialize final linear projection so initial predicted velocity field is IDENTICALLY ZERO (v_pred = 0)
        nn.init.zeros_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

        # 2. Zero-initialize adaLN modulation projection (shift = 0, scale = 0)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        if self.adaLN_modulation[-1].bias is not None:
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Output token sequence from final DiT block of shape [B, N, D].
            c (torch.Tensor): Global conditioning vector of shape [B, D].

        Returns:
            torch.Tensor: Projected patch token features of shape [B, N, P^2 * out_channels].
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

        # 1. Compute dynamic shift and scale from condition vector c
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        # 2. Modulate normalized activations: Modulate(LN(x), shift, scale)
        x = modulate(self.norm_final(x), shift, scale)

        # 3. Project to patch features [B, N, P^2 * out_channels]
        x = self.linear(x)

        return x
