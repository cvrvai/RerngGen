"""Complete Diffusion Transformer (DiT) Architecture.

Integrates Patch Embedding, 2D Positional Embedding, Timestep Embedder,
adaLN-Zero Transformer Blocks, Final Layer Head, and Unpatchify into an
end-to-end continuous-time velocity predictor for Flow Matching.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn
import yaml

from src.block import DiTBlock
from src.final_layer import FinalLayer
from src.patch_embed import PatchEmbed
from src.positional_embedding import PositionalEmbedding2D
from src.timestep_embedding import TimestepEmbedder
from src.unpatchify import Unpatchify


class DiT(nn.Module):
    """Diffusion Transformer (DiT) Model.

    End-to-End Pipeline:
        x_t [B, C_in, H, W]
        ├──> PatchEmbed -> [B, N, D]
        ├──> + PositionalEmbedding2D -> [B, N, D]
        │
        t [B] ──> TimestepEmbedder -> [B, D] ──┐
        text_embed [B, D] (optional) ──────────┴──> Condition c = t_embed + text_embed [B, D]
        │
        ├──> Stack of depth DiTBlocks (adaLN-Zero, Attention, MLP) -> [B, N, D]
        ├──> FinalLayer -> [B, N, P^2 * C_out]
        └──> Unpatchify -> Predicted Velocity Field v_pred [B, C_out, H, W]

    Architectural Boundary Note:
        Text embedding projection (e.g. CLIP/T5 pooled embedding projected to hidden_size D)
        lives strictly upstream of DiT. DiT receives already-projected pooled text embeddings
        of shape [B, D] and combines them via condition addition: c = t_embed + text_embed.

    Args:
        in_channels (int): Latent input channels (default: 4).
        out_channels (int): Latent output channels (default: 4).
        latent_size (int): Spatial latent resolution H=W (default: 32).
        patch_size (int): Patch resolution P (default: 2).
        hidden_size (int): Transformer hidden dimension D (default: 384).
        depth (int): Number of stacked DiTBlocks (default: 8).
        num_heads (int): Number of self-attention heads (default: 6).
        mlp_ratio (float): Hidden expansion ratio for MLP (default: 4.0).
        time_scale (float): Scaling factor for normalized flow time (default: 1000.0).
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        latent_size: int = 32,
        patch_size: int = 2,
        hidden_size: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        time_scale: float = 1000.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.time_scale = time_scale

        # 1. Patch Embedding: [B, C_in, H, W] -> [B, N, D]
        self.x_embed = PatchEmbed(
            latent_size=latent_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
        )

        # 2. Fixed 2D Sinusoidal Grid Positional Embedding: [1, N, D]
        self.pos_embed = PositionalEmbedding2D(
            embed_dim=hidden_size,
            grid_size=self.x_embed.grid_size,
        )

        # 3. Timestep Conditioning Embedder: [B] in [0, 1] -> [B, D]
        self.t_embedder = TimestepEmbedder(
            hidden_size=hidden_size,
            time_scale=time_scale,
        )

        # 4. Transformer Blocks: Depth x DiTBlock
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        # 5. Final Layer Head: [B, N, D] -> [B, N, P^2 * C_out]
        self.final_layer = FinalLayer(
            hidden_size=hidden_size,
            patch_size=patch_size,
            out_channels=out_channels,
        )

        # 6. Unpatchify Module: [B, N, P^2 * C_out] -> [B, C_out, H, W]
        self.unpatchify = Unpatchify(
            patch_size=patch_size,
            out_channels=out_channels,
        )

    @classmethod
    def from_config(cls, config: Union[Dict[str, Any], str, Path]) -> "DiT":
        """Instantiate DiT from configuration dictionary or YAML file path.

        Args:
            config (Union[Dict[str, Any], str, Path]): Config dictionary or YAML file path.

        Returns:
            DiT: Configured model instance.
        """
        if isinstance(config, (str, Path)):
            with open(config, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = config

        model_params = config_dict.get("model", config_dict)
        return cls(
            in_channels=model_params.get("in_channels", 4),
            out_channels=model_params.get("out_channels", 4),
            latent_size=model_params.get("latent_size", 32),
            patch_size=model_params.get("patch_size", 2),
            hidden_size=model_params.get("hidden_size", 384),
            depth=model_params.get("depth", 8),
            num_heads=model_params.get("num_heads", 6),
            mlp_ratio=model_params.get("mlp_ratio", 4.0),
            time_scale=model_params.get("time_scale", 1000.0),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the complete DiT model.

        Args:
            x (torch.Tensor): Noisy latent image tensor [B, in_channels, H, W].
            t (torch.Tensor): Continuous flow timestep tensor [B] with values in [0, 1].
            text_embed (Optional[torch.Tensor]): Optional pooled text condition embeddings [B, hidden_size].

        Returns:
            torch.Tensor: Predicted velocity field tensor [B, out_channels, H, W].
        """
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(
                f"Expected input channels {self.in_channels}, but got {C} (shape: {x.shape})."
            )
        if H != self.latent_size or W != self.latent_size:
            raise ValueError(
                f"Expected spatial resolution ({self.latent_size}, {self.latent_size}), "
                f"but got ({H}, {W})."
            )

        # 1. Patchify & linear embed: [B, C_in, H, W] -> [B, N, D]
        x = self.x_embed(x)

        # 2. Add fixed 2D spatial sinusoidal positional embeddings: [B, N, D]
        x = self.pos_embed(x)

        # 3. Compute condition vector c: [B, D]
        c = self.t_embedder(t)
        if text_embed is not None:
            if text_embed.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: input x has batch size {B}, "
                    f"but text_embed has batch size {text_embed.shape[0]}."
                )
            if text_embed.shape[1] != self.hidden_size:
                raise ValueError(
                    f"Expected text_embed feature dimension {self.hidden_size}, "
                    f"but got {text_embed.shape[1]} (shape: {text_embed.shape})."
                )
            c = c + text_embed

        # 4. Pass through depth transformer blocks with adaLN-Zero modulation
        for block in self.blocks:
            x = block(x, c)

        # 5. Modulated Final Layer Head: [B, N, D] -> [B, N, P^2 * C_out]
        x = self.final_layer(x, c)

        # 6. Unpatchify into 2D velocity prediction: [B, N, P^2 * C_out] -> [B, C_out, H, W]
        v_pred = self.unpatchify(x)

        return v_pred
