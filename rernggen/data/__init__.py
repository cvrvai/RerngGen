"""Dataset intake, preprocessing, latent caching, captioning, and paired DataLoader for RerngGen."""

from rernggen.data.captions import CaptionManager, compute_caption_sha256
from rernggen.data.dataset import (
    GovernanceMode,
    PairedLatentTextDataset,
    create_paired_dataloader,
    paired_collate_fn,
)
from rernggen.data.importer import DatasetImporter, import_image_directory
from rernggen.data.latent_cache import LatentCacheLoader, LatentCacheGenerator
from rernggen.data.preprocessor import ImagePreprocessor, preprocess_dataset
from rernggen.data.schema import (
    CaptionRecord,
    ImportReport,
    LatentCacheReport,
    LatentRecord,
    PreprocessingReport,
    ProcessedRecord,
    TextEmbeddingCacheReport,
    TextEmbeddingRecord,
)
from rernggen.data.text_cache import TextEmbeddingCacheGenerator, TextEmbeddingCacheLoader

__all__ = [
    "DatasetImporter",
    "import_image_directory",
    "ImagePreprocessor",
    "preprocess_dataset",
    "LatentCacheGenerator",
    "LatentCacheLoader",
    "CaptionManager",
    "compute_caption_sha256",
    "TextEmbeddingCacheGenerator",
    "TextEmbeddingCacheLoader",
    "PairedLatentTextDataset",
    "GovernanceMode",
    "create_paired_dataloader",
    "paired_collate_fn",
    "ManifestRecord",
    "ProcessedRecord",
    "LatentRecord",
    "CaptionRecord",
    "TextEmbeddingRecord",
    "ImportReport",
    "PreprocessingReport",
    "LatentCacheReport",
    "TextEmbeddingCacheReport",
]
