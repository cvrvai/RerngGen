"""Patch Embedding layer for Diffusion Transformers (DiT).

Transforms a spatial latent tensor [B, C, H, W] into a sequence of
token vectors [B, N, D] via non-overlapping 2D patch projection.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Patch Embedding Layer.

    Args:
        latent_size (int or Tuple[int, int]): Spatial resolution of input latent (H, W).
        patch_size (int): Spatial size of each patch P. Default: 2.
        in_channels (int): Number of input latent channels C. Default: 4.
        hidden_size (int): Transformer token embedding dimension D. Default: 384.
        bias (bool): Whether to include bias in linear/conv projection. Default: True.
    """

    def __init__(
        self,
        latent_size: Union[int, Tuple[int, int]] = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 384,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(latent_size, int):
            self.latent_size: Tuple[int, int] = (latent_size, latent_size)
        else:
            self.latent_size = latent_size

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        H, W = self.latent_size
        if H % patch_size != 0 or W % patch_size != 0:
            raise ValueError(
                f"Latent dimensions ({H}, {W}) must be divisible by patch_size ({patch_size})."
            )

        self.grid_size: Tuple[int, int] = (H // patch_size, W // patch_size)
        self.num_patches: int = self.grid_size[0] * self.grid_size[1]

        # Non-overlapping 2D convolution acts as simultaneous patch extraction + linear projection
        # Kernel size = P, Stride = P guarantees each PxP patch is mapped to 1 token of dimension D
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard ViT/DiT weight initialization: Xavier uniform on conv kernel, zeros for bias
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.shape[0], -1))
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input latent tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: Patch token embeddings of shape [B, N, D], where N = (H/P) * (W/P).
        """
        B, C, H, W = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input channels {self.in_channels}, but got {C} (input shape: {x.shape})."
            )
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"Input spatial dimensions ({H}, {W}) must be divisible by patch_size ({self.patch_size})."
            )

        # 1. Non-overlapping Conv2d: [B, C, H, W] -> [B, D, H/P, W/P]
        x = self.proj(x)

        # 2. Flatten spatial grid into sequence: [B, D, H/P, W/P] -> [B, D, N]
        x = x.flatten(2)

        # 3. Transpose to standard Transformer sequence layout: [B, D, N] -> [B, N, D]
        x = x.transpose(1, 2)

        return x
