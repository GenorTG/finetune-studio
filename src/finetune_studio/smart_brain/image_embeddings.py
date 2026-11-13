"""Cross-modal image/text embeddings for Smart Brain image retrieval.

The multilingual Sentence-Transformers CLIP text model is aligned with the
matching CLIP image encoder. Both are deliberately separate from the corpus'
text embedder: their vectors live in a different space and must never be
compared with ``vectors.npy``.
"""
from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from finetune_studio.smart_brain.devices import resolve_device

IMAGE_MODEL = "sentence-transformers/clip-ViT-B-32"
TEXT_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


@dataclass(frozen=True)
class ImageAsset:
    """An extracted image with stable document and page/slide provenance."""

    document_id: str
    filename: str
    source: str
    locator: str
    mime_type: str
    width: int
    height: int
    data: bytes

    @property
    def asset_id(self) -> str:
        digest = hashlib.sha256(self.data).hexdigest()
        identity = f"{self.document_id}\0{self.locator}\0{digest}".encode()
        return hashlib.sha256(identity).hexdigest()[:24]

    @property
    def suffix(self) -> str:
        return ".png" if self.data.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "source": self.source,
            "locator": self.locator,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "path": f"image_assets/{self.asset_id}{self.suffix}",
        }


@dataclass(frozen=True)
class ImageEmbedder:
    encode_images: Callable[[list[Any]], np.ndarray]
    encode_texts: Callable[[str | list[str]], np.ndarray]
    dim: int
    image_model: str = IMAGE_MODEL
    text_model: str = TEXT_MODEL

    def metadata(self) -> dict[str, Any]:
        return {
            "image_model": self.image_model,
            "text_model": self.text_model,
            "dimension": self.dim,
            "normalize": True,
            "similarity": "cosine",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def load_image_embedder(device: str = "auto") -> ImageEmbedder:
    """Load the paired CLIP image encoder and multilingual text encoder."""
    from sentence_transformers import SentenceTransformer

    resolved = resolve_device(device)
    image_model = SentenceTransformer(IMAGE_MODEL, device=resolved)
    text_model = SentenceTransformer(TEXT_MODEL, device=resolved)
    image_dim = int(image_model.get_embedding_dimension())
    text_dim = int(text_model.get_embedding_dimension())
    if image_dim != text_dim:
        raise ValueError(
            f"Image/text embedding dimensions differ ({image_dim} vs {text_dim}); "
            "the encoders do not share a vector space."
        )

    def encode_images(images: list[Any]) -> np.ndarray:
        if not images:
            return np.zeros((0, image_dim), dtype=np.float32)
        vectors = image_model.encode(
            images, normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(texts: str | list[str]) -> np.ndarray:
        single = isinstance(texts, str)
        values = [texts] if single else list(texts)
        if not values:
            return np.zeros((0, text_dim), dtype=np.float32)
        vectors = text_model.encode(
            values, normalize_embeddings=True, show_progress_bar=False,
        )
        result = np.asarray(vectors, dtype=np.float32)
        return result[0] if single else result

    return ImageEmbedder(encode_images, encode_texts, image_dim)


def image_from_bytes(data: bytes) -> Any:
    """Decode an image safely as RGB without retaining a file handle."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.convert("RGB")


def cached_image_embedder(device: str = "auto") -> ImageEmbedder:
    """Load through the Smart Brain model cache (shared lifecycle with text retrieval)."""
    from finetune_studio.smart_brain.model_cache import cached_image_embedder as cached

    return cached(device=device)
