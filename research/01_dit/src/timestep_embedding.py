"""Timestep Embedding module for Diffusion Transformers (DiT).

Maps continuous diffusion/flow time scalar t in [0, 1] into a rich,
high-dimensional conditioning vector [B, D] via sinusoidal frequency
projection followed by a 2-layer SiLU MLP.
"""

import math
from typing import Optional
import torch
import torch.nn as nn


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int = 256,
    max_period: float = 10000.0,
) -> torch.Tensor:
    """Computes sinusoidal frequency features for scalar timesteps.

    Math:
        half_dim = embedding_dim // 2
        freqs = exp(-log(max_period) * arange(half_dim) / half_dim)
        args = timesteps * freqs
        embedding = [cos(args), sin(args)]

    Args:
        timesteps (torch.Tensor): 1D tensor of scalar timesteps [B] or [B, 1].
        embedding_dim (int): Dimensionality of the sinusoidal feature representation.
        max_period (float): Maximum frequency wavelength constant. Default: 10000.0.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape [B, embedding_dim].
    """
    if timesteps.ndim == 2 and timesteps.shape[1] == 1:
        timesteps = timesteps.squeeze(1)
    elif timesteps.ndim != 1:
        raise ValueError(
            f"Expected timesteps tensor to have shape [B] or [B, 1], but got {timesteps.shape}."
        )

    half = embedding_dim // 2
    # Frequency bands in log-linear space
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # Pad if odd dimension requested
    if embedding_dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
        )

    return embedding


class TimestepEmbedder(nn.Module):
    """Timestep Embedding MLP for DiT.

    Projects sinusoidal frequency features through a 2-layer MLP with SiLU activation:
    Linear(frequency_embedding_size -> hidden_size) -> SiLU -> Linear(hidden_size -> hidden_size)

    Args:
        hidden_size (int): Target Transformer embedding dimension D (e.g. 384).
        frequency_embedding_size (int): Dimension of initial sinusoidal features (default: 256).
        max_period (float): Maximum wavelength for sinusoidal frequencies. Default: 10000.0.
    """

    def __init__(
        self,
        hidden_size: int = 384,
        frequency_embedding_size: int = 256,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            t (torch.Tensor): Continuous or discrete timesteps of shape [B] or [B, 1].

        Returns:
            torch.Tensor: Timestep conditioning vectors of shape [B, hidden_size].
        """
        # 1. Project scalar t -> sinusoidal feature vector [B, frequency_embedding_size]
        t_freq = sinusoidal_timestep_embedding(
            timesteps=t,
            embedding_dim=self.frequency_embedding_size,
            max_period=self.max_period,
        )

        # 2. Match MLP parameter dtype and compute [B, frequency_embedding_size] -> [B, hidden_size]
        dtype = self.mlp[0].weight.dtype
        t_emb = self.mlp(t_freq.to(dtype=dtype))

        return t_emb
