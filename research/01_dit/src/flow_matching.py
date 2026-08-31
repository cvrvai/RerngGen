"""Frozen Baseline Flow Matching Objective for Diffusion Transformers (DiT).

Implements the continuous-time linear interpolation trajectory and straight-line
velocity target computation:
    x_t = (1 - t) * x_noise + t * x_data
    v_target = x_data - x_noise
    Loss = MSE(v_pred, v_target)
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_interpolated_state(
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Computes the continuous-time linear interpolation state x_t between noise and data.

    Formula:
        x_t = (1 - t) * x_noise + t * x_data

    Args:
        x_data (torch.Tensor): Real latent data state x_1 of shape [B, C, H, W].
        x_noise (torch.Tensor): Standard Gaussian noise state x_0 of shape [B, C, H, W].
        t (torch.Tensor): Continuous time tensor of shape [B] with values in [0, 1].

    Returns:
        torch.Tensor: Interpolated state x_t of shape [B, C, H, W].
    """
    if x_data.shape != x_noise.shape:
        raise ValueError(
            f"Shape mismatch: x_data {x_data.shape} vs x_noise {x_noise.shape}."
        )
    if x_data.ndim != 4:
        raise ValueError(f"Expected 4D latent tensor [B, C, H, W], got {x_data.shape}.")

    B = x_data.shape[0]
    t_flat = t.view(-1)
    if t_flat.shape[0] != B:
        raise ValueError(f"Batch size mismatch: latent B={B} vs timestep B={t_flat.shape[0]}.")

    # Broadcast t: [B] -> [B, 1, 1, 1] to match [B, C, H, W]
    t_broadcast = t_flat.view(B, 1, 1, 1)

    return (1.0 - t_broadcast) * x_noise + t_broadcast * x_data


def compute_target_velocity(
    x_data: torch.Tensor,
    x_noise: torch.Tensor,
) -> torch.Tensor:
    """Computes the analytical straight-line velocity target v_target = d(x_t)/dt.

    Formula:
        v_target = d/dt [(1 - t)*x_noise + t*x_data] = x_data - x_noise

    Args:
        x_data (torch.Tensor): Real latent data state x_1 of shape [B, C, H, W].
        x_noise (torch.Tensor): Standard Gaussian noise state x_0 of shape [B, C, H, W].

    Returns:
        torch.Tensor: Analytical target velocity field of shape [B, C, H, W].
    """
    if x_data.shape != x_noise.shape:
        raise ValueError(
            f"Shape mismatch: x_data {x_data.shape} vs x_noise {x_noise.shape}."
        )
    return x_data - x_noise


class FlowMatchingObjective(nn.Module):
    """Flow Matching Loss Computation Module.

    Calculates the mean squared error (MSE) between the model's predicted velocity
    field and the analytical straight-line target velocity.

    Architectural Decoupling:
        The DiT model is treated purely as a velocity predictor v_theta(x_t, t, text_embed)
        and remains entirely unaware of how x_t, x_noise, or v_target were constructed.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        model: nn.Module,
        x_data: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Executes a single Flow Matching loss step.

        Args:
            model (nn.Module): DiT velocity prediction model.
            x_data (torch.Tensor): Clean latent data batch [B, C, H, W].
            text_embed (Optional[torch.Tensor]): Optional pooled text condition [B, D].
            noise (Optional[torch.Tensor]): Optional externally supplied noise [B, C, H, W].
                                           If None, sampled from N(0, I).
            t (Optional[torch.Tensor]): Optional externally supplied timesteps [B].
                                       If None, sampled uniformly from Uniform(0, 1).

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
                - loss: Scalar MSE loss tensor.
                - metrics: Dictionary containing {"loss", "v_pred", "v_target", "x_t", "t"}.
        """
        if x_data.ndim != 4:
            raise ValueError(f"Expected 4D data tensor [B, C, H, W], got {x_data.shape}.")

        B = x_data.shape[0]

        # 1. Sample or validate standard Gaussian noise x_0 ~ N(0, I)
        if noise is None:
            noise = torch.randn_like(x_data)
        else:
            if noise.shape != x_data.shape:
                raise ValueError(
                    f"Shape mismatch: x_data {x_data.shape} vs noise {noise.shape}."
                )
            if noise.device != x_data.device:
                raise ValueError(
                    f"Device mismatch: x_data on {x_data.device} vs noise on {noise.device}."
                )
            if noise.dtype != x_data.dtype:
                raise ValueError(
                    f"Dtype mismatch: x_data is {x_data.dtype} vs noise is {noise.dtype}."
                )

        # 2. Sample or validate continuous timesteps t ~ Uniform(0, 1)
        if t is None:
            t = torch.rand(B, device=x_data.device, dtype=x_data.dtype)
        else:
            t_flat = t.view(-1)
            if t_flat.shape[0] != B:
                raise ValueError(
                    f"Batch size mismatch: x_data has {B} samples vs t has {t_flat.shape[0]}."
                )
            if t.device != x_data.device:
                raise ValueError(
                    f"Device mismatch: x_data on {x_data.device} vs t on {t.device}."
                )
            if t.dtype != x_data.dtype:
                raise ValueError(
                    f"Dtype mismatch: x_data is {x_data.dtype} vs t is {t.dtype}."
                )
            t = t_flat

        # 3. Construct linear interpolation state x_t = (1 - t)*x_noise + t*x_data
        x_t = compute_interpolated_state(x_data=x_data, x_noise=noise, t=t)

        # 4. Compute analytical target velocity v_target = x_data - x_noise
        v_target = compute_target_velocity(x_data=x_data, x_noise=noise)

        # 5. Model prediction v_pred = DiT(x_t, t, text_embed)
        v_pred = model(x_t, t, text_embed=text_embed)

        # 6. Mean Squared Error (MSE) loss
        loss = F.mse_loss(v_pred, v_target)

        metrics = {
            "loss": loss,
            "v_pred": v_pred,
            "v_target": v_target,
            "x_t": x_t,
            "t": t,
        }

        return loss, metrics
