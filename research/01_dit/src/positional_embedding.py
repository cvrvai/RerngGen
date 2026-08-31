"""2D Sinusoidal Positional Embeddings for Diffusion Transformers (DiT).

Provides fixed (non-learnable) 2D sinusoidal coordinate embeddings to inject
spatial awareness (row y, column x) into flat patch token sequences [B, N, D].
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import numpy as np


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: np.ndarray,
    temperature: float = 10000.0,
) -> np.ndarray:
    """Computes 1D sinusoidal positional embeddings for an array of coordinates.

    Args:
        embed_dim (int): Dimensionality of each 1D coordinate embedding (must be even).
        pos (np.ndarray): 1D coordinate array of shape [N].
        temperature (float): Base wavelength scaling factor. Default: 10000.0.

    Returns:
        np.ndarray: Sinusoidal embeddings of shape [N, embed_dim].
    """
    if embed_dim % 2 != 0:
        raise ValueError(f"1D embedding dimension must be even, but got {embed_dim}.")

    # Number of frequency bands: half for sin, half for cos
    num_bands = embed_dim // 2
    omega = np.arange(num_bands, dtype=np.float64) / num_bands
    omega = 1.0 / (temperature**omega)  # [num_bands]

    pos = pos.reshape(-1)  # [N]
    out = np.einsum("m,d->md", pos, omega)  # [N, num_bands]

    emb_sin = np.sin(out)  # [N, num_bands]
    emb_cos = np.cos(out)  # [N, num_bands]

    return np.concatenate([emb_sin, emb_cos], axis=1)  # [N, embed_dim]


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Union[int, Tuple[int, int]],
    temperature: float = 10000.0,
) -> np.ndarray:
    """Computes fixed 2D sinusoidal grid positional embeddings.

    Allocates embed_dim / 2 features for the vertical (Y) grid axis,
    and embed_dim / 2 features for the horizontal (X) grid axis.

    Args:
        embed_dim (int): Total token embedding dimension D (must be divisible by 4).
        grid_size (int or Tuple[int, int]): Spatial patch grid resolution (h, w).
        temperature (float): Base wavelength scaling factor. Default: 10000.0.

    Returns:
        np.ndarray: Positional embedding matrix of shape [N, embed_dim] where N = h * w.
    """
    if embed_dim % 4 != 0:
        raise ValueError(
            f"Embedding dimension D must be divisible by 4 for 2D sin/cos (got {embed_dim})."
        )

    if isinstance(grid_size, int):
        grid_h = grid_w = grid_size
    else:
        grid_h, grid_w = grid_size

    # 1. Generate 2D coordinate meshgrid: y in [0, grid_h-1], x in [0, grid_w-1]
    grid_y = np.arange(grid_h, dtype=np.float64)
    grid_x = np.arange(grid_w, dtype=np.float64)
    mesh_y, mesh_x = np.meshgrid(grid_y, grid_x, indexing="ij")  # [grid_h, grid_w]

    mesh_y = mesh_y.reshape(-1)  # [N]
    mesh_x = mesh_x.reshape(-1)  # [N]

    # 2. Compute 1D sin/cos embeddings for Y and X coordinates (each gets D / 2 dimensions)
    dim_axis = embed_dim // 2
    emb_y = get_1d_sincos_pos_embed_from_grid(dim_axis, mesh_y, temperature=temperature)  # [N, D/2]
    emb_x = get_1d_sincos_pos_embed_from_grid(dim_axis, mesh_x, temperature=temperature)  # [N, D/2]

    # 3. Concatenate Y and X positional features
    emb_2d = np.concatenate([emb_y, emb_x], axis=1)  # [N, embed_dim]
    return emb_2d


class PositionalEmbedding2D(nn.Module):
    """2D Sinusoidal Positional Embedding module for DiT.

    Stores fixed 2D sinusoidal embeddings as a persistent non-trainable buffer [1, N, D].
    """

    def __init__(
        self,
        embed_dim: int = 384,
        grid_size: Union[int, Tuple[int, int]] = (16, 16),
        temperature: float = 10000.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = (grid_size, grid_size) if isinstance(grid_size, int) else grid_size
        self.temperature = temperature

        pos_embed = get_2d_sincos_pos_embed(
            embed_dim=self.embed_dim,
            grid_size=self.grid_size,
            temperature=self.temperature,
        )
        # Register as a non-learnable buffer of shape [1, N, D]
        pos_tensor = torch.from_numpy(pos_embed).float().unsqueeze(0)
        self.register_buffer("pos_embed", pos_tensor, persistent=False)

    @property
    def num_patches(self) -> int:
        return self.pos_embed.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds 2D positional embeddings to input token sequence.

        Args:
            x (torch.Tensor): Token sequence tensor of shape [B, N, D].

        Returns:
            torch.Tensor: Position-augmented tokens of shape [B, N, D].
        """
        B, N, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected D={self.embed_dim}, but got {D}."
            )
        if N != self.pos_embed.shape[1]:
            raise ValueError(
                f"Sequence length mismatch: expected N={self.pos_embed.shape[1]}, but got {N}."
            )

        return x + self.pos_embed.to(device=x.device, dtype=x.dtype)
