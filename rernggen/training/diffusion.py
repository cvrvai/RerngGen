"""Forward diffusion process and standard noise prediction loss computation for RerngGen."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionSchedule:
    """Discrete-time forward diffusion schedule with linear beta variance."""

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Precompute schedule coefficients
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _extract_into_tensor(self, arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extracts values from 1D schedule array indexed by timesteps and reshapes for broadcasting."""
        res = arr.to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res.unsqueeze(-1)
        return res.expand(broadcast_shape)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Diffuse the clean latent data x_0 to timestep t: x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps."""
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError(f"Noise shape {noise.shape} does not match x_start shape {x_start.shape}.")

        sqrt_alpha_bar = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_bar = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def training_loss(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        text_embed: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the mean squared error (MSE) noise prediction diffusion loss."""
        x_t, noise = self.q_sample(x_start=x_start, t=t, noise=noise)
        epsilon_hat = model(x_t, t, text_embed)
        loss = F.mse_loss(epsilon_hat, noise)
        return loss, epsilon_hat, noise
