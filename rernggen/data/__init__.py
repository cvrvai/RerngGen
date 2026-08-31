"""Dataset intake, preprocessing, latent caching, captioning, and text embedding management for RerngGen."""

from rernggen.data.captions import CaptionManager, compute_caption_sha256
from rernggen.data.importer import DatasetImporter, import_image_directory
from rernggen.data.latent_cache import LatentCacheGenerator, LatentCacheLoader
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
