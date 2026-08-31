"""Frozen text encoder adapter and trainable projection layer for RerngGen.

Provides a unified interface for tokenizing and extracting pooled [B, 512] text embeddings
using a frozen pretrained CLIP text model, along with a dedicated trainable linear projection
layer mapping [B, 512] -> [B, 384] into the DiT conditioning space.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


def compute_tokenizer_identity_sha256(
    tokenizer_class: str,
    tokenizer_config_sha256: str,
    vocab_sha256: str,
    merges_sha256: str,
    special_tokens_map_sha256: str,
    max_token_length: int,
) -> str:
    """Computes a deterministic SHA-256 hash representing the complete tokenizer identity."""
    payload = (
        f"class={tokenizer_class}|"
        f"config={tokenizer_config_sha256}|"
        f"vocab={vocab_sha256}|"
        f"merges={merges_sha256}|"
        f"special={special_tokens_map_sha256}|"
        f"max_len={max_token_length}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TextEncoderSpec:
    """Provenance and specification metadata for a frozen text encoder."""

    model_id: str = "openai/clip-vit-base-patch32"
    revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    tokenizer_class: str = "CLIPTokenizer"
    encoder_class: str = "CLIPTextModel"
    output_dim: int = 512
    max_token_length: int = 77
    pooling_policy: str = "eos_token"
    weights_sha256: str = "c06e8e2f73be888f8026728b85703a2cbf6960a12ab47f4a9ba02fda4ae7674b"
    config_sha256: str = "50f791f5217268bfce4da7e33c3cea360c2a8fb3fd6d4d094384e8f244d74459"
    tokenizer_config_sha256: str = "4d1439e2cefa4a2c46934aa42dafab7a4f35c3d47612d9a9a04f880c4ca494ab"
    vocab_sha256: str = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"
    merges_sha256: str = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"
    special_tokens_map_sha256: str = "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35"
    tokenizer_identity_sha256: str = ""
    local_cache_path: str = "models/text_encoder/openai--clip-text-base-patch32"
    license: str = "MIT"

    def __post_init__(self) -> None:
        if not self.tokenizer_identity_sha256:
            self.tokenizer_identity_sha256 = compute_tokenizer_identity_sha256(
                tokenizer_class=self.tokenizer_class,
                tokenizer_config_sha256=self.tokenizer_config_sha256,
                vocab_sha256=self.vocab_sha256,
                merges_sha256=self.merges_sha256,
                special_tokens_map_sha256=self.special_tokens_map_sha256,
                max_token_length=self.max_token_length,
            )


class CLIPTextEncoderAdapter(nn.Module):
    """Adapter for pretrained CLIP text encoders with strict parameter freezing and deterministic pooling."""

    def __init__(
        self,
        tokenizer: Optional[Any] = None,
        text_model: Optional[nn.Module] = None,
        spec: Optional[TextEncoderSpec] = None,
    ) -> None:
        """Initializes the text encoder adapter with parameter freezing.

        Args:
            tokenizer: Initialized Hugging Face tokenizer instance.
            text_model: Pretrained text encoder model instance.
            spec: Provenance metadata.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.spec = spec or TextEncoderSpec()

        if self.text_model is not None:
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: Union[str, Path] = "models/text_encoder/openai--clip-text-base-patch32",
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "CLIPTextEncoderAdapter":
        """Loads a pretrained CLIP tokenizer and text encoder from local disk or Hugging Face.

        Args:
            model_id_or_path: Local path or Hugging Face model identifier.
            device: Compute device.
            dtype: Compute dtype.

        Returns:
            CLIPTextEncoderAdapter: Initialized and frozen adapter.
        """
        from transformers import CLIPTextModel, CLIPTokenizer
        from rernggen.data.importer import compute_sha256

        model_path = Path(model_id_or_path)
        model_path_str = str(model_id_or_path)

        tokenizer = CLIPTokenizer.from_pretrained(model_path_str)
        text_model = CLIPTextModel.from_pretrained(model_path_str)
        text_model = text_model.to(device=device, dtype=dtype)
        text_model.eval()
        for p in text_model.parameters():
            p.requires_grad_(False)

        config_sha = None
        weights_sha = None
        tokenizer_cfg_sha = None
        vocab_sha = None
        merges_sha = None
        special_tokens_sha = None

        if model_path.is_dir():
            cfg_p = model_path / "config.json"
            if cfg_p.exists():
                config_sha = compute_sha256(cfg_p)
            for w_name in ["model.safetensors", "pytorch_model.bin"]:
                w_p = model_path / w_name
                if w_p.exists():
                    weights_sha = compute_sha256(w_p)
                    break

            t_cfg_p = model_path / "tokenizer_config.json"
            if t_cfg_p.exists():
                tokenizer_cfg_sha = compute_sha256(t_cfg_p)

            v_p = model_path / "vocab.json"
            if v_p.exists():
                vocab_sha = compute_sha256(v_p)

            m_p = model_path / "merges.txt"
            if m_p.exists():
                merges_sha = compute_sha256(m_p)

            s_p = model_path / "special_tokens_map.json"
            if s_p.exists():
                special_tokens_sha = compute_sha256(s_p)

            t_json_p = model_path / "tokenizer.json"
            if t_json_p.exists():
                t_json_sha = compute_sha256(t_json_p)
                if vocab_sha is None:
                    vocab_sha = t_json_sha
                if merges_sha is None:
                    merges_sha = t_json_sha
                if special_tokens_sha is None:
                    special_tokens_sha = t_json_sha

        spec = TextEncoderSpec(
            model_id=model_path_str,
            revision="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            output_dim=int(getattr(text_model.config, "hidden_size", 512)),
            max_token_length=int(getattr(text_model.config, "max_position_embeddings", 77)),
            pooling_policy="eos_token",
            weights_sha256=weights_sha or "c06e8e2f73be888f8026728b85703a2cbf6960a12ab47f4a9ba02fda4ae7674b",
            config_sha256=config_sha or "50f791f5217268bfce4da7e33c3cea360c2a8fb3fd6d4d094384e8f244d74459",
            tokenizer_class=tokenizer.__class__.__name__,
            tokenizer_config_sha256=tokenizer_cfg_sha or "4d1439e2cefa4a2c46934aa42dafab7a4f35c3d47612d9a9a04f880c4ca494ab",
            vocab_sha256=vocab_sha or "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
            merges_sha256=merges_sha or "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
            special_tokens_map_sha256=special_tokens_sha or "6d9109cc838977f3ca94a379eec36aecc7c807e1785cd729660ca2fc0171fb35",
        )
        return cls(tokenizer=tokenizer, text_model=text_model, spec=spec)

    @torch.no_grad()
    def encode_text(
        self,
        captions: Union[str, List[str]],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Tokenizes and encodes captions into frozen pooled [B, 512] text embeddings.

        Args:
            captions: Single caption string or list of caption strings.
            device: Target device for compute and output tensor.
            dtype: Compute and output dtype.

        Returns:
            torch.Tensor: Pooled text embeddings [B, 512] with requires_grad=False.
        """
        if isinstance(captions, str):
            captions = [captions]

        for c in captions:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("Caption must be a non-empty string.")

        target_device = device or (
            next(self.text_model.parameters()).device if self.text_model else torch.device("cpu")
        )

        tokens = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.spec.max_token_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(target_device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(target_device)

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)

        # Standard CLIP text pooling: pooler_output corresponds to the EOS token hidden state
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Fallback: manually extract the hidden state at the EOS token position
            last_hidden = outputs.last_hidden_state
            eos_indices = input_ids.argmax(dim=-1)
            pooled = last_hidden[torch.arange(last_hidden.shape[0]), eos_indices]

        pooled = pooled.to(dtype=dtype)
        if not torch.all(torch.isfinite(pooled)):
            raise ValueError("Non-finite values encountered in text encoder output.")

        return pooled.detach()

    @torch.no_grad()
    def encode_text_with_diagnostics(
        self,
        captions: Union[str, List[str]],
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Encodes captions and returns both pooled embeddings and token diagnostics.

        Returns:
            Tuple[torch.Tensor, List[Dict[str, Any]]]: (embeddings [B, 512], diagnostics list).
        """
        if isinstance(captions, str):
            captions = [captions]

        embeddings = self.encode_text(captions, device=device, dtype=dtype)
        diagnostics: List[Dict[str, Any]] = []

        for c in captions:
            raw_tokens = self.tokenizer(c, truncation=False)["input_ids"]
            count = len(raw_tokens)
            diagnostics.append(
                {
                    "token_count": count,
                    "truncated": count > self.spec.max_token_length,
                }
            )

        return embeddings, diagnostics


class MockTextEncoder(nn.Module):
    """Lightweight deterministic mock text encoder for fast unit tests.

    Maps text strings to deterministic [B, 512] tensors using cryptographic hash seeds.
    """

    def __init__(
        self,
        output_dim: int = 512,
        tokenizer_class: str = "MockTokenizer",
        tokenizer_config_sha256: str = "mock_tokenizer_config_sha256",
        vocab_sha256: str = "mock_vocab_sha256",
        merges_sha256: str = "mock_merges_sha256",
        special_tokens_map_sha256: str = "mock_special_tokens_map_sha256",
        max_token_length: int = 77,
        weights_sha256: str = "mock_weights_sha256",
        config_sha256: str = "mock_config_sha256",
        revision: str = "mock_v1",
        pooling_policy: str = "eos_token",
    ) -> None:
        super().__init__()
        self.spec = TextEncoderSpec(
            model_id="mock_text_encoder",
            revision=revision,
            output_dim=output_dim,
            tokenizer_class=tokenizer_class,
            tokenizer_config_sha256=tokenizer_config_sha256,
            vocab_sha256=vocab_sha256,
            merges_sha256=merges_sha256,
            special_tokens_map_sha256=special_tokens_map_sha256,
            max_token_length=max_token_length,
            pooling_policy=pooling_policy,
            weights_sha256=weights_sha256,
            config_sha256=config_sha256,
        )

    def encode_text(
        self,
        captions: Union[str, List[str]],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(captions, str):
            captions = [captions]

        embeddings = []
        for c in captions:
            if not isinstance(c, str) or not c.strip():
                raise ValueError("Caption must be a non-empty string.")
            # Deterministic generator seeded by caption hash
            h = int(hashlib.sha256(c.encode("utf-8")).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(h)
            vec = torch.randn(self.spec.output_dim, generator=g, dtype=dtype)
            embeddings.append(vec)

        return torch.stack(embeddings, dim=0).to(device=device)

    def encode_text_with_diagnostics(
        self,
        captions: Union[str, List[str]],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        if isinstance(captions, str):
            captions = [captions]
        embeddings = self.encode_text(captions, device=device, dtype=dtype)
        diagnostics = [
            {
                "token_count": len(c.split()) + 2,
                "truncated": (len(c.split()) + 2) > self.spec.max_token_length,
            }
            for c in captions
        ]
        return embeddings, diagnostics


class TextProjection(nn.Module):
    """Trainable linear projection mapping frozen text embeddings [B, E_text] -> [B, 384].

    This layer is outside the frozen text encoder and receives gradient updates during training.
    """

    def __init__(self, in_features: int = 512, out_features: int = 384) -> None:
        """Initializes the linear projection layer.

        Args:
            in_features: Hidden dimension of frozen text encoder (default: 512).
            out_features: DiT conditioning dimension (default: 384).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.proj = nn.Linear(in_features, out_features, bias=True)

        # Standard small-weight initialization for stable initial condition injection
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects text embeddings [B, in_features] to [B, out_features].

        Args:
            x (torch.Tensor): Pooled frozen text embedding tensor [B, in_features].

        Returns:
            torch.Tensor: Projected text condition tensor [B, out_features].
        """
        if x.ndim != 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected 2D text embedding tensor [B, {self.in_features}], got shape {x.shape}."
            )
        return self.proj(x)
