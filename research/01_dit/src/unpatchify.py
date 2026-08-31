"""Unpatchify module for Diffusion Transformers (DiT).

Reconstructs spatial latent tensors [B, C, H, W] from flat patch token sequences
[B, N, P * P * C] by reshaping and rearranging intra-patch and inter-patch grids.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


def unpatchify(
    x: torch.Tensor,
    patch_size: int = 2,
    out_channels: int = 4,
    grid_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Reconstructs spatial latent [B, C, H, W] from patch tokens [B, N, P*P*C].

    Args:
        x (torch.Tensor): Projected token tensor of shape [B, N, P * P * out_channels].
        patch_size (int): Spatial size of each patch P. Default: 2.
        out_channels (int): Number of latent channels C. Default: 4.
        grid_size (Optional[Tuple[int, int]]): (H/P, W/P). If None, assumed square (sqrt(N), sqrt(N)).

    Returns:
        torch.Tensor: Reconstructed spatial latent tensor of shape [B, out_channels, H, W].
    """
    B, N, D_patch = x.shape
    P = patch_size
    C = out_channels
    expected_patch_dim = P * P * C

    if D_patch != expected_patch_dim:
        raise ValueError(
            f"Expected token feature dimension to be P*P*C = {P}*{P}*{C} = {expected_patch_dim}, "
            f"but got {D_patch} (shape: {x.shape})."
        )

    if grid_size is None:
        h = int(N**0.5)
        w = int(N**0.5)
        if h * w != N:
            raise ValueError(
                f"Sequence length N={N} is not a perfect square. Please supply explicit `grid_size=(h, w)`."
            )
    else:
        h, w = grid_size
        if h * w != N:
            raise ValueError(
                f"Supplied grid_size ({h}, {w}) with product {h*w} does not match sequence length N={N}."
            )

    # 1. Unfold sequence N into 2D grid of patches: [B, N, P*P*C] -> [B, h, w, P, P, C]
    x = x.view(B, h, w, P, P, C)

    # 2. Permute dimensions: [B, h, w, P_h, P_w, C] -> [B, C, h, P_h, w, P_w]
    # Dimension mapping:
    # 0: B, 1: h, 2: w, 3: P_h, 4: P_w, 5: C
    # -> (0, 5, 1, 3, 2, 4)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()

    # 3. Collapse (h, P_h) into H = h*P, and (w, P_w) into W = w*P
    # [B, C, h, P_h, w, P_w] -> [B, C, H, W]
    H = h * P
    W = w * P
    x = x.view(B, C, H, W)

    return x


class Unpatchify(nn.Module):
    """Unpatchify module wrapper for seamless model pipelining."""

    def __init__(
        self,
        patch_size: int = 2,
        out_channels: int = 4,
        latent_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.grid_size = (
            (latent_size[0] // patch_size, latent_size[1] // patch_size)
            if latent_size is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        active_grid = grid_size if grid_size is not None else self.grid_size
        return unpatchify(
            x,
            patch_size=self.patch_size,
            out_channels=self.out_channels,
            grid_size=active_grid,
        )
