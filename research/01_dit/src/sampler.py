"""Deterministic Euler ODE Sampler for Diffusion Transformers (DiT).

Implements forward numerical integration from t=0 (Gaussian noise) to t=1 (generated data latent):
    dt = 1.0 / num_steps
    x_{k+1} = x_k + dt * v_theta(x_k, t_k, text_embed)
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class EulerSampler:
    """Deterministic First-Order Euler ODE Sampler.

    Under our frozen baseline Flow Matching convention, generative integration proceeds
    forward in time from t=0 (Gaussian noise) to t=1 (data latent).
    Because the learned velocity field v_theta points in the direction of increasing t,
    each step advances forward via:
        x_{k+1} = x_k + dt * v_theta(x_k, t_k, text_embed)
    """

    def __init__(self, num_steps: int = 50) -> None:
        """Initializes the EulerSampler.

        Args:
            num_steps (int): Number of uniform integration steps. Must be >= 1.
        """
        if not isinstance(num_steps, int) or num_steps < 1:
            raise ValueError(f"num_steps must be a positive integer >= 1, got {num_steps}.")
        self.num_steps = num_steps

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        text_embed: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Integrates the generative ODE from t=0 to t=1 using the forward Euler method.

        Args:
            model (nn.Module): DiT velocity predictor v_theta(x, t, text_embed).
            noise (torch.Tensor): Initial noise latent at t=0 of shape [B, C, H, W].
            text_embed (Optional[torch.Tensor]): Optional pooled conditioning vector [B, D].
            num_steps (Optional[int]): Optional override for integration step count.
            return_trajectory (bool): If True, also returns intermediate latent states [x_0, ..., x_N].

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
                - Final generated latent state x_1 at t=1 [B, C, H, W].
                - (Optional) List of N+1 trajectory states [x_0, x_1, ..., x_N].
        """
        steps = num_steps if num_steps is not None else self.num_steps
        if not isinstance(steps, int) or steps < 1:
            raise ValueError(f"num_steps must be a positive integer >= 1, got {steps}.")

        if noise.ndim != 4:
            raise ValueError(f"Expected 4D noise latent [B, C, H, W], got {noise.shape}.")

        B = noise.shape[0]
        if text_embed is not None and text_embed.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: noise B={B} vs text_embed B={text_embed.shape[0]}."
            )

        dt = 1.0 / float(steps)
        x = noise.clone()
        trajectory: Optional[List[torch.Tensor]] = [x.clone()] if return_trajectory else None

        for step_idx in range(steps):
            t_scalar = step_idx / float(steps)
            t_tensor = torch.full((B,), t_scalar, device=noise.device, dtype=noise.dtype)

            # Predict velocity field at current continuous state x and time t
            v_pred = model(x, t_tensor, text_embed=text_embed)

            # Forward Euler step: x_{k+1} = x_k + dt * v_pred
            x = x + dt * v_pred

            if return_trajectory and trajectory is not None:
                trajectory.append(x.clone())

        if return_trajectory and trajectory is not None:
            return x, trajectory

        return x
