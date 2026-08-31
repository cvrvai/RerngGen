"""Deterministic frozen text embedding cache generator and loader for RerngGen.

Extracts pooled [512] frozen text embeddings from versioned captions once,
persisting them as standalone safetensors files with cryptographic provenance tracking,
diagnostics, and atomic manifest management.
"""

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union
from safetensors.torch import load_file, save_file
import torch
from rernggen.data.captions import CaptionManager
from rernggen.data.importer import compute_sha256
from rernggen.data.schema import TextEmbeddingCacheReport, TextEmbeddingRecord
from rernggen.models.text.interface import CLIPTextEncoderAdapter


class TextEmbeddingCacheGenerator:
    """Generates and manages versioned permanent text embedding caches from caption manifests."""

    def __init__(
        self,
        text_encoder_adapter: CLIPTextEncoderAdapter,
        cache_version: str = "clip_b32_v001",
        dataset_root: Union[str, Path] = "datasets",
    ) -> None:
        """Initializes the text embedding cache generator.

        Args:
            text_encoder_adapter (CLIPTextEncoderAdapter): Frozen text encoder adapter.
            cache_version (str): Subdirectory identifier under cache/text_embeds/.
            dataset_root (Union[str, Path]): Root path for datasets. Default: "datasets".
        """
        self.adapter = text_encoder_adapter
        self.cache_version = cache_version
        self.dataset_root = Path(dataset_root)

    def generate_cache(
        self,
        dataset_id: str = "khmer_story_cartoon_v001",
        caption_version: str = "captions_v001",
        force: bool = False,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> TextEmbeddingCacheReport:
        """Generates permanent frozen text embedding files for all captions in the version.

        Args:
            dataset_id (str): Dataset ID.
            caption_version (str): Caption version to encode.
            force (bool): If True, re-encodes embeddings even if already cached. Default: False.
            device (Union[str, torch.device]): Device for text encoder inference.
            dtype (torch.dtype): Compute dtype for text encoder inference.

        Returns:
            TextEmbeddingCacheReport: Execution summary metrics and manifest records.
        """
        start_time = time.time()
        dataset_dir = self.dataset_root / dataset_id
        cap_manager = CaptionManager(dataset_root=self.dataset_root)
        captions = cap_manager.load_captions(dataset_id=dataset_id, version=caption_version)

        cache_dir = dataset_dir / "cache" / "text_embeds" / self.cache_version
        cache_manifest_path = cache_dir / "manifest.jsonl"
        cache_dir.mkdir(parents=True, exist_ok=True)

        report = TextEmbeddingCacheReport(
            dataset_id=dataset_id,
            caption_version=caption_version,
            cache_version=self.cache_version,
            cache_dir=cache_dir,
            manifest_path=cache_manifest_path,
            total_captions_in_dataset=len(captions),
            text_encoder_provenance={
                "model_id": self.adapter.spec.model_id,
                "revision": self.adapter.spec.revision,
                "output_dim": self.adapter.spec.output_dim,
                "pooling_policy": self.adapter.spec.pooling_policy,
                "weights_sha256": self.adapter.spec.weights_sha256,
                "config_sha256": self.adapter.spec.config_sha256,
                "tokenizer_class": self.adapter.spec.tokenizer_class,
                "tokenizer_config_sha256": self.adapter.spec.tokenizer_config_sha256,
                "vocab_sha256": self.adapter.spec.vocab_sha256,
                "merges_sha256": self.adapter.spec.merges_sha256,
                "special_tokens_map_sha256": self.adapter.spec.special_tokens_map_sha256,
                "max_token_length": self.adapter.spec.max_token_length,
                "tokenizer_identity_sha256": self.adapter.spec.tokenizer_identity_sha256,
            },
        )

        # Load existing cache manifest for idempotency
        existing_cache_map: Dict[str, Dict[str, Any]] = {}
        if cache_manifest_path.exists() and not force:
            with open(cache_manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        existing_cache_map[rec["image_id"]] = rec

        final_records: List[TextEmbeddingRecord] = []

        for cap_rec in captions:
            image_id = cap_rec.image_id
            embed_filename = f"{image_id}.safetensors"
            embed_file = cache_dir / embed_filename
            embed_rel = f"cache/text_embeds/{self.cache_version}/{embed_filename}"

            # Idempotency check: verify existing embedding matches caption hash, text encoder, and tokenizer identity
            if (
                not force
                and image_id in existing_cache_map
                and embed_file.exists()
            ):
                cached_rec = existing_cache_map[image_id]
                if (
                    cached_rec.get("caption_sha256") == cap_rec.caption_sha256
                    and cached_rec.get("text_encoder_revision") == self.adapter.spec.revision
                    and cached_rec.get("text_encoder_weights_sha256") == self.adapter.spec.weights_sha256
                    and cached_rec.get("text_encoder_config_sha256") == self.adapter.spec.config_sha256
                    and cached_rec.get("tokenizer_class") == self.adapter.spec.tokenizer_class
                    and cached_rec.get("tokenizer_config_sha256") == self.adapter.spec.tokenizer_config_sha256
                    and cached_rec.get("vocab_sha256") == self.adapter.spec.vocab_sha256
                    and cached_rec.get("merges_sha256") == self.adapter.spec.merges_sha256
                    and cached_rec.get("special_tokens_map_sha256") == self.adapter.spec.special_tokens_map_sha256
                    and cached_rec.get("max_token_length") == self.adapter.spec.max_token_length
                    and cached_rec.get("tokenizer_identity_sha256") == self.adapter.spec.tokenizer_identity_sha256
                    and cached_rec.get("pooling_policy") == self.adapter.spec.pooling_policy
                    and cached_rec.get("cache_version") == self.cache_version
                    and compute_sha256(embed_file) == cached_rec.get("embedding_sha256")
                ):
                    report.valid_cache_hits += 1
                    report.total_cache_bytes += embed_file.stat().st_size
                    final_records.append(TextEmbeddingRecord(**cached_rec))
                    continue

            tmp_embed_file = None
            try:
                # 1. Encode caption with frozen text encoder -> [1, 512] + diagnostics
                pooled_embed, diag_list = self.adapter.encode_text_with_diagnostics(
                    captions=[cap_rec.caption],
                    device=device,
                    dtype=dtype,
                )
                embed_tensor = pooled_embed.squeeze(0).contiguous().to(torch.float32).cpu()
                diag = diag_list[0] if diag_list else {"token_count": 0, "truncated": False}

                if not torch.all(torch.isfinite(embed_tensor)):
                    raise ValueError(f"Non-finite values encountered in text embedding for {image_id}")

                # 2. Calculate tensor diagnostics
                min_val = float(embed_tensor.min().item())
                max_val = float(embed_tensor.max().item())
                mean_val = float(embed_tensor.mean().item())
                std_val = float(embed_tensor.std().item())
                l2_norm = float(torch.linalg.norm(embed_tensor).item())

                # 3. Atomic safetensors write
                tmp_embed_file = cache_dir / f"{image_id}_tmp_{os.getpid()}_{time.time_ns()}.safetensors"
                save_file({"embedding": embed_tensor}, tmp_embed_file)
                os.replace(tmp_embed_file, embed_file)
                tmp_embed_file = None

                embed_size = embed_file.stat().st_size
                report.total_cache_bytes += embed_size
                embed_sha = compute_sha256(embed_file)

                # 4. Build TextEmbeddingRecord with exact governance passthrough & truncation visibility
                emb_rec = TextEmbeddingRecord(
                    image_id=image_id,
                    dataset_id=dataset_id,
                    dataset_version="v001",
                    caption_version=caption_version,
                    caption_sha256=cap_rec.caption_sha256,
                    text_encoder_id=self.adapter.spec.model_id,
                    text_encoder_revision=self.adapter.spec.revision,
                    text_encoder_weights_sha256=self.adapter.spec.weights_sha256,
                    text_encoder_config_sha256=self.adapter.spec.config_sha256,
                    tokenizer_class=self.adapter.spec.tokenizer_class,
                    tokenizer_config_sha256=self.adapter.spec.tokenizer_config_sha256,
                    vocab_sha256=self.adapter.spec.vocab_sha256,
                    merges_sha256=self.adapter.spec.merges_sha256,
                    special_tokens_map_sha256=self.adapter.spec.special_tokens_map_sha256,
                    tokenizer_identity_sha256=self.adapter.spec.tokenizer_identity_sha256,
                    max_token_length=self.adapter.spec.max_token_length,
                    pooling_policy=self.adapter.spec.pooling_policy,
                    embedding_shape=list(embed_tensor.shape),
                    embedding_dtype=str(embed_tensor.dtype).replace("torch.", ""),
                    embedding_sha256=embed_sha,
                    embedding_relative_path=embed_rel,
                    min_val=round(min_val, 4),
                    max_val=round(max_val, 4),
                    mean_val=round(mean_val, 4),
                    std_val=round(std_val, 4),
                    l2_norm=round(l2_norm, 4),
                    token_count=diag["token_count"],
                    truncated=diag["truncated"],
                    training_allowed=cap_rec.training_allowed,
                    commercial_allowed=cap_rec.commercial_allowed,
                    license_id=cap_rec.license_id,
                    cache_version=self.cache_version,
                    status="CACHED",
                )

                final_records.append(emb_rec)
                report.embeddings_created += 1

            except Exception as e:
                if tmp_embed_file and tmp_embed_file.exists():
                    tmp_embed_file.unlink(missing_ok=True)
                report.failures += 1
                report.failure_details.append({"image_id": image_id, "error": str(e)})

        # 5. Atomic Manifest Write
        if final_records:
            tmp_manifest = cache_dir / f"manifest_tmp_{os.getpid()}_{time.time_ns()}.jsonl"
            with open(tmp_manifest, "w", encoding="utf-8") as f:
                for r in final_records:
                    f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            os.replace(tmp_manifest, cache_manifest_path)

        report.records = final_records
        report.elapsed_time_seconds = time.time() - start_time
        return report


class TextEmbeddingCacheLoader:
    """Fast, parameter-free text embedding loader for training pipelines.

    Reads standalone safetensors files directly from disk without instantiating or invoking a text encoder.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        cache_version: str = "clip_b32_v001",
    ) -> None:
        """Initializes the text embedding loader.

        Args:
            dataset_dir (Union[str, Path]): Path to dataset directory.
            cache_version (str): Cache version identifier.
        """
        self.dataset_dir = Path(dataset_dir)
        self.cache_version = cache_version
        self.cache_dir = self.dataset_dir / "cache" / "text_embeds" / cache_version
        self.manifest_path = self.cache_dir / "manifest.jsonl"

    def load_manifest(self) -> List[TextEmbeddingRecord]:
        """Loads all text embedding records from the cache manifest."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Text embedding manifest not found: {self.manifest_path}")

        records: List[TextEmbeddingRecord] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(TextEmbeddingRecord(**json.loads(line)))
        return records

    def load_embedding(
        self,
        image_id_or_record: Union[str, TextEmbeddingRecord],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Loads a cached text embedding [512] directly into memory without invoking a text encoder.

        Args:
            image_id_or_record (Union[str, TextEmbeddingRecord]): Image ID or TextEmbeddingRecord.
            device (Union[str, torch.device]): Target device.
            dtype (torch.dtype): Target dtype.

        Returns:
            torch.Tensor: Frozen text embedding [512].
        """
        if isinstance(image_id_or_record, TextEmbeddingRecord):
            embed_path = self.dataset_dir / image_id_or_record.embedding_relative_path
        else:
            embed_path = self.cache_dir / f"{image_id_or_record}.safetensors"

        if not embed_path.exists():
            raise FileNotFoundError(f"Cached text embedding file not found: {embed_path}")

        tensors = load_file(str(embed_path))
        if "embedding" not in tensors:
            raise KeyError(f"Expected key 'embedding' in safetensors file {embed_path}")

        embed = tensors["embedding"].to(device=device, dtype=dtype)
        return embed
