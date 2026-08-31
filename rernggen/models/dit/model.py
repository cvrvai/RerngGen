"""Lightweight Diffusion Transformer (DiT) implementation for RerngGen training validation.

This module provides a minimal, technically correct, and transparent DiT model
built strictly with standard PyTorch primitives (adaLN-Zero conditioning, patchification,
multi-head self-attention, and continuous timestep/text conditioning).
"""

import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies adaptive LayerNorm scale and shift modulation."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations via sinusoidal embeddings and an MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        """Creates sinusoidal timestep embeddings matching standard continuous diffusion formulations."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class TextEmbedder(nn.Module):
    """Projects pooled text embeddings into the Transformer hidden dimension."""

    def __init__(self, text_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, text_embed: torch.Tensor) -> torch.Tensor:
        return self.proj(text_embed)


class PatchEmbed(nn.Module):
    """2D Image/Latent to Patch Embedding."""

    def __init__(
        self,
        latent_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 4,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if H != self.latent_size or W != self.latent_size:
            raise ValueError(
                f"Input latent spatial dimensions ({H}, {W}) do not match expected ({self.latent_size}, {self.latent_size})."
            )
        # [B, C, H, W] -> [B, D, H/P, W/P] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class DiTBlock(nn.Module):
    """Transformer block with adaptive LayerNorm (adaLN-Zero) conditioning."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Initialize adaLN modulation to zeros
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # adaLN parameters: shift1, scale1, gate1, shift2, scale2, gate2
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Modulated Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # Modulated MLP
        x_norm2 = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    """The final adaLN-Zero layer of DiT that projects tokens back to patch channels."""

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Initialize to zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TinyDiT(nn.Module):
    """Diffusion Transformer model parameterized for smoke tests and learning baselines.

    Forward Pipeline:
        x_t [B, in_channels, H, W]
        ├──> PatchEmbed -> [B, N, D]
        ├──> + Positional Embeddings -> [B, N, D]
        │
        t [B] ───────────> TimestepEmbedder -> [B, D] ──┐
        text_embed [B, T_dim] -> TextEmbedder ──────────┴──> Combined conditioning c [B, D]
        │
        ├──> Stack of DiTBlocks (adaLN-Zero modulation, Self-Attention, MLP) -> [B, N, D]
        ├──> FinalLayer -> [B, N, P^2 * out_channels]
        └──> Unpatchify -> Predicted Noise epsilon_hat [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        latent_size: int = 32,
        patch_size: int = 4,
        hidden_size: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        text_dim: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.text_dim = text_dim

        self.x_embedder = PatchEmbed(
            latent_size=latent_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )
        self.t_embedder = TimestepEmbedder(hidden_size=hidden_size)
        self.text_embedder = TextEmbedder(text_dim=text_dim, hidden_size=hidden_size)

        # Learnable 1D/2D positional embeddings
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # Final Layer
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initializes positional and embedding weights."""
        # Initialize patch embed like a linear layer
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize pos_embed
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes [B, N, P*P*C] back to spatial latents [B, C, H, W]."""
        c = self.out_channels
        p = self.patch_size
        h = w = self.latent_size // p
        if x.shape[1] != h * w:
            raise ValueError(f"Token count {x.shape[1]} does not match expected {h * w}.")

        # [B, H/P * W/P, P*P*C] -> [B, H/P, W/P, P, P, C]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        # [B, H/P, W/P, P, P, C] -> [B, C, H/P, P, W/P, P]
        x = torch.einsum("nhwpqc->nchpwq", x)
        # [B, C, H/P, P, W/P, P] -> [B, C, H, W]
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for noise prediction.

        Args:
            x (torch.Tensor): Noisy latent batch of shape [B, in_channels, H, W].
            t (torch.Tensor): Diffusion timestep indices of shape [B].
            text_embed (torch.Tensor): Pooled text embedding batch of shape [B, text_dim].

        Returns:
            torch.Tensor: Predicted noise epsilon_hat of shape [B, out_channels, H, W].
        """
        # Patchify and add positional embeddings
        x_tokens = self.x_embedder(x) + self.pos_embed  # [B, N, D]

        # Embed conditions
        t_emb = self.t_embedder(t)  # [B, D]
        text_cond = self.text_embedder(text_embed)  # [B, D]
        c = t_emb + text_cond  # Combined conditioning [B, D]

        # Apply DiT blocks
        for block in self.blocks:
            x_tokens = block(x_tokens, c)

        # Final projection and unpatchify
        out_tokens = self.final_layer(x_tokens, c)  # [B, N, P*P*C]
        out_latents = self.unpatchify(out_tokens)  # [B, C, H, W]
        return out_latents

    def get_parameter_count(self) -> Dict[str, int]:
        """Calculates total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TinyDiT":
        """Instantiates TinyDiT from a configuration dictionary."""
        return cls(
            in_channels=config.get("in_channels", config.get("input_channels", 4)),
            out_channels=config.get("out_channels", config.get("output_channels", 4)),
            latent_size=config.get("latent_size", 32),
            patch_size=config.get("patch_size", 4),
            hidden_size=config.get("hidden_size", 64),
            depth=config.get("depth", 2),
            num_heads=config.get("num_heads", 4),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            text_dim=config.get("text_dim", 512),
        )
