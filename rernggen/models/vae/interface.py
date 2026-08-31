"""Frozen VAE adapter and interface for RerngGen.

Provides a unified interface for 4-channel, 8x spatial downsampling AutoencoderKL VAEs
with explicit [-1, 1] input normalization, deterministic posterior mode encoding,
dynamic latent scaling factors, and strict parameter freezing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class VAESpec:
    """Provenance and specification metadata for a frozen VAE model."""

    model_id: str = "stabilityai/sd-vae-ft-mse"
    revision: str = "31f26fdeee1355a5c34592e401dd41e45d25a493"
    architecture: str = "AutoencoderKL"
    latent_channels: int = 4
    spatial_downsample_factor: int = 8
    scaling_factor: float = 0.18215
    posterior_policy: str = "posterior_mode"
    weights_sha256: Optional[str] = "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815"
    config_sha256: Optional[str] = "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e"
    diffusers_version: Optional[str] = None
    torch_version: Optional[str] = None
    in_channels: int = 3
    out_channels: int = 3
    local_cache_path: str = "models/vae/stabilityai--sd-vae-ft-mse"
    license: str = "CreativeML Open RAIL-M"


class AutoencoderKLAdapter(nn.Module):
    """Adapter for pretrained AutoencoderKL models satisfying the [B, 4, 32, 32] latent contract."""

    def __init__(
        self,
        vae_model: nn.Module,
        spec: Optional[VAESpec] = None,
    ) -> None:
        """Initializes the VAE adapter with strict validation and parameter freezing.

        Args:
            vae_model (nn.Module): Pretrained AutoencoderKL instance.
            spec (Optional[VAESpec]): Optional specification metadata.
        """
        super().__init__()
        self.vae = vae_model
        self.spec = spec or VAESpec()

        # 1. Verify VAE configuration contract
        latent_channels = getattr(self.vae.config, "latent_channels", None) or getattr(
            self.vae.config, "out_channels", None
        )
        if latent_channels != 4:
            raise ValueError(
                f"Incompatible VAE: expected 4 latent channels, got {latent_channels}. "
                "Changing latent channels violates the DiT contract and requires an architectural revision."
            )

        # 2. Extract dynamic scaling factor from VAE configuration
        self.scaling_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        self.spec.scaling_factor = self.scaling_factor
        self.spec.latent_channels = latent_channels
        self.spec.torch_version = torch.__version__

        # 3. Freeze all parameters and set eval mode
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: Union[str, Path] = "models/vae/stabilityai--sd-vae-ft-mse",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "AutoencoderKLAdapter":
        """Loads a pretrained AutoencoderKL model from local disk or HuggingFace Hub.

        Args:
            model_id_or_path (Union[str, Path]): Local path or HuggingFace model ID.
            device (Union[str, torch.device]): Target compute device.
            dtype (torch.dtype): Target compute dtype.

        Returns:
            AutoencoderKLAdapter: Initialized and frozen adapter.
        """
        import diffusers
        from diffusers import AutoencoderKL
        from rernggen.data.importer import compute_sha256

        model_path = Path(model_id_or_path)
        model_path_str = str(model_id_or_path)
        vae = AutoencoderKL.from_pretrained(model_path_str)
        vae = vae.to(device=device, dtype=dtype)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)

        config_sha = None
        weights_sha = None
        if model_path.is_dir():
            cfg_p = model_path / "config.json"
            if cfg_p.exists():
                config_sha = compute_sha256(cfg_p)
            for w_name in ["diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"]:
                w_p = model_path / w_name
                if w_p.exists():
                    weights_sha = compute_sha256(w_p)
                    break

        spec = VAESpec(
            model_id=model_path_str,
            revision="31f26fdeee1355a5c34592e401dd41e45d25a493",
            scaling_factor=float(getattr(vae.config, "scaling_factor", 0.18215)),
            weights_sha256=weights_sha or "a1d993488569e928462932c8c38a0760b874d166399b14414135bd9c42df5815",
            config_sha256=config_sha or "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e",
            diffusers_version=getattr(diffusers, "__version__", "unknown"),
            torch_version=torch.__version__,
        )
        return cls(vae_model=vae, spec=spec)

    @staticmethod
    def normalize_input(x: torch.Tensor) -> torch.Tensor:
        """Normalizes float image tensor from [0.0, 1.0] to [-1.0, 1.0].

        x_norm = 2.0 * x - 1.0
        """
        return 2.0 * x - 1.0

    @staticmethod
    def unnormalize_output(x: torch.Tensor) -> torch.Tensor:
        """Maps reconstructed float tensor from [-1.0, 1.0] to [0.0, 1.0] and clamps.

        x_img = clamp((x + 1.0) / 2.0, 0.0, 1.0)
        """
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)

    @torch.no_grad()
    def encode(
        self,
        x: torch.Tensor,
        return_raw: bool = False,
    ) -> torch.Tensor:
        """Encodes normalized RGB image tensor into latent representation using posterior mode.

        Args:
            x (torch.Tensor): Image tensor [B, 3, H, W] normalized to [-1.0, 1.0].
            return_raw (bool): If True, returns unscaled z_raw instead of z_model.

        Returns:
            torch.Tensor: Latent tensor [B, 4, H//8, W//8].
        """
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 4D image tensor [B, 3, H, W], got {x.shape}.")

        if x.shape[2] % 8 != 0 or x.shape[3] % 8 != 0:
            raise ValueError(
                f"Image spatial dimensions ({x.shape[2]}x{x.shape[3]}) must be divisible by 8."
            )

        # Deterministic posterior mode (not stochastic sample)
        posterior = self.vae.encode(x).latent_dist
        z_raw = posterior.mode()

        if return_raw:
            return z_raw

        # Apply VAE scaling factor: z_model = z_raw * scaling_factor
        z_model = z_raw * self.scaling_factor
        return z_model

    @torch.no_grad()
    def decode(
        self,
        z: torch.Tensor,
        is_model_latent: bool = True,
    ) -> torch.Tensor:
        """Decodes latent tensor into normalized RGB reconstruction tensor.

        Args:
            z (torch.Tensor): Latent tensor [B, 4, H//8, W//8].
            is_model_latent (bool): If True, divides by scaling_factor before decoding.

        Returns:
            torch.Tensor: Reconstructed image tensor [B, 3, H, W] in [-1.0, 1.0].
        """
        if z.ndim != 4 or z.shape[1] != 4:
            raise ValueError(f"Expected 4D latent tensor [B, 4, h, w], got {z.shape}.")

        # Invert scaling factor: z_raw = z_model / scaling_factor
        z_raw = (z / self.scaling_factor) if is_model_latent else z

        reconstruction = self.vae.decode(z_raw).sample
        return reconstruction

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Performs full end-to-end reconstruction: x [-1, 1] -> encode -> decode -> [-1, 1]."""
        z = self.encode(x, return_raw=False)
        return self.decode(z, is_model_latent=True)


class MockDiagonalGaussianDistribution:
    """Mock distribution returning deterministic mode."""

    def __init__(self, mode_tensor: torch.Tensor) -> None:
        self._mode = mode_tensor

    def mode(self) -> torch.Tensor:
        return self._mode

    def sample(self) -> torch.Tensor:
        return self._mode


class MockEncoderOutput:
    """Mock container for VAE encoder output."""

    def __init__(self, latent: torch.Tensor) -> None:
        self.latent_dist = MockDiagonalGaussianDistribution(latent)


class MockDecoderOutput:
    """Mock container for VAE decoder output."""

    def __init__(self, sample: torch.Tensor) -> None:
        self.sample = sample


class MockVAEConfig:
    """Mock configuration for MockVAE."""

    def __init__(
        self,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
        sample_size: int = 256,
    ) -> None:
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self.sample_size = sample_size


class MockVAE(nn.Module):
    """Lightweight mock VAE for fast unit tests without downloading weights.

    Uses Conv2d (stride 8) and ConvTranspose2d (stride 8) to preserve [B, 3, 256, 256] -> [B, 4, 32, 32] contract.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        scaling_factor: float = 0.18215,
    ) -> None:
        super().__init__()
        self.config = MockVAEConfig(
            latent_channels=latent_channels,
            scaling_factor=scaling_factor,
        )
        self.encoder = nn.Conv2d(3, latent_channels, kernel_size=8, stride=8)
        self.decoder = nn.ConvTranspose2d(latent_channels, 3, kernel_size=8, stride=8)

    def encode(self, x: torch.Tensor) -> MockEncoderOutput:
        z = self.encoder(x)
        return MockEncoderOutput(z)

    def decode(self, z: torch.Tensor) -> MockDecoderOutput:
        recon = self.decoder(z)
        return MockDecoderOutput(recon)
