"""Transformer MLP (Pointwise Feed-Forward Network) for Diffusion Transformers (DiT).

Implements the standard pointwise channel-mixing MLP:
Linear(D -> mlp_ratio * D) -> GELU(approximate="tanh") -> Linear(mlp_ratio * D -> D)
"""

from typing import Callable, Optional
import torch
import torch.nn as nn


class Mlp(nn.Module):
    """Transformer Pointwise Feed-Forward Network.

    Applies non-linear channel transformations independently at each token position.

    Args:
        in_features (int): Input feature dimension D (e.g. 384).
        hidden_features (Optional[int]): Expanded hidden dimension (default: 4 * in_features = 1536).
        out_features (Optional[int]): Output feature dimension (default: in_features = 384).
        act_layer (Callable[[], nn.Module]): Activation module constructor. Default: nn.GELU(approximate="tanh").
        bias (bool): Whether to include bias in linear layers. Default: True.
    """

    def __init__(
        self,
        in_features: int = 384,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Module] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features or (in_features * 4)
        self.out_features = out_features or in_features

        self.fc1 = nn.Linear(self.in_features, self.hidden_features, bias=bias)
        self.act = act_layer if act_layer is not None else nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(self.hidden_features, self.out_features, bias=bias)

        self._init_weights()

    def _init_weights(self) -> None:
        # Standard ViT / DiT Xavier uniform initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input token sequence of shape [B, N, in_features].

        Returns:
            torch.Tensor: Transformed token sequence of shape [B, N, out_features].
        """
        B, N, D = x.shape
        if D != self.in_features:
            raise ValueError(
                f"Expected input feature dimension {self.in_features}, but got {D} (shape: {x.shape})."
            )

        # 1. Expand channels: [B, N, D] -> [B, N, 4*D] = [B, 256, 1536]
        x = self.fc1(x)

        # 2. Non-linear activation: GELU(approximate="tanh")
        x = self.act(x)

        # 3. Contract back to model dimension: [B, N, 4*D] -> [B, N, D] = [B, 256, 384]
        x = self.fc2(x)

        return x
