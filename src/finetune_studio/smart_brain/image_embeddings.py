"""Image/text embeddings for the Smart Brain image layer (Image search).

Pictures (standalone image files and the figures inside docx / pptx / pdf / xlsx files) are embedded with a CLIP-class
image encoder; a question is embedded with the paired text encoder, an uploaded picture with the image encoder. The
vectors live in their own space (``image_vectors.npy``) and are never compared with the passage vectors.

Models are pinned to a Hub commit and copied into the shared model store (``shared_models/image_models``) by the build
job, so an index and the package exported from it read the same weights. A search never downloads: no model on disk
= a clear error asking for a build.

The encoder itself (load + encode) and the ranking live in ``standalone_server`` (``open_image_encoder`` /
``rank_images``): the exported package's ``server.py`` runs the same code on the same files.
"""
from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from finetune_studio.smart_brain.devices import resolve_device

THUMB_SIDE = 320          # longest side of a stored thumbnail (px)
EMBED_BATCH = 16
MAX_QUERY_BYTES = 12 * 1024 * 1024
_SNAPSHOT_FILES = ["*.json", "*.txt", "model.safetensors", "*/model.safetensors", "*/*.json", "*/*.txt"]


@dataclass(frozen=True)
class ImageModelSpec:
    """One selectable image model: a CLIP image encoder and the text encoder aligned with it (pinned revisions)."""

    id: str
    label: str
    image_repo: str
    image_revision: str
    text_repo: str             # "" = the image repo's own text tower
    text_revision: str
    licence: str
    size_mb: int
    languages: str
    dim: int = 512

    def snapshots(self) -> list[tuple[str, str]]:
        out = [(self.image_repo, self.image_revision)]
        if self.text_repo:
            out.append((self.text_repo, self.text_revision))
        return out

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "licence": self.licence, "size_mb": self.size_mb,
                "languages": self.languages, "dim": self.dim, "image_repo": self.image_repo,
                "image_revision": self.image_revision, "text_repo": self.text_repo or self.image_repo,
                "text_revision": self.text_revision or self.image_revision}


IMAGE_MODELS: dict[str, ImageModelSpec] = {
    "clip-vit-b32-multilingual": ImageModelSpec(
        id="clip-vit-b32-multilingual",
        label="CLIP ViT-B/32 + multilingual text (50+ languages)",
        image_repo="sentence-transformers/clip-ViT-B-32",
        image_revision="327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
        text_repo="sentence-transformers/clip-ViT-B-32-multilingual-v1",
        text_revision="58edf8cada9e398793dca955574a48cbb7f18be2",
        licence="MIT (OpenAI CLIP image encoder) + Apache-2.0 (multilingual text encoder)",
        size_mb=1150, languages="50+ languages"),
    "clip-vit-b32": ImageModelSpec(
        id="clip-vit-b32",
        label="CLIP ViT-B/32 (English questions only, smaller)",
        image_repo="sentence-transformers/clip-ViT-B-32",
        image_revision="327ab6726d33c0e22f920c83f2ff9e4bd38ca37f",
        text_repo="", text_revision="",
        licence="MIT (OpenAI CLIP)", size_mb=605, languages="English"),
}
DEFAULT_IMAGE_MODEL = "clip-vit-b32-multilingual"


def spec_for(model_id: str | None) -> ImageModelSpec:
    key = (model_id or "").strip() or DEFAULT_IMAGE_MODEL
    try:
        return IMAGE_MODELS[key]
    except KeyError:
        raise ValueError(f"unknown image model {key!r}; choose one of: {', '.join(IMAGE_MODELS)}") from None


def model_dirs(spec: ImageModelSpec) -> tuple[Path, Path | None]:
    """The shared-store folders of the model's image and (separate) text encoder."""
    from finetune_studio.data import shared_models as sm

    image = sm.snapshot_dir(spec.image_repo, spec.image_revision)
    text = sm.snapshot_dir(spec.text_repo, spec.text_revision) if spec.text_repo else None
    return image, text


def on_disk(model_id: str | None = None) -> bool:
    from finetune_studio.data import shared_models as sm

    spec = spec_for(model_id)
    return all(sm.has_snapshot(repo, rev) for repo, rev in spec.snapshots())


def ensure_model(model_id: str | None = None) -> tuple[Path, Path | None]:
    """Download (once) the pinned snapshots into the shared store. Build jobs only: never from a request."""
    from finetune_studio.data import shared_models as sm

    spec = spec_for(model_id)
    for repo, rev in spec.snapshots():
        sm.register_snapshot(repo, rev, _SNAPSHOT_FILES)
    return model_dirs(spec)


@dataclass(frozen=True)
class ImageEmbedder:
    encode_images: Callable[[list[Any]], np.ndarray]
    encode_texts: Callable[[str | list[str]], np.ndarray]
    dim: int
    model_id: str = DEFAULT_IMAGE_MODEL

    @property
    def spec(self) -> ImageModelSpec:
        return spec_for(self.model_id)

    def metadata(self) -> dict[str, Any]:
        s = self.spec
        return {"model": s.id, "image_model": f"{s.image_repo}@{s.image_revision}",
                "text_model": f"{s.text_repo or s.image_repo}@{s.text_revision or s.image_revision}",
                "dimension": self.dim, "normalize": True, "similarity": "cosine", "licence": s.licence,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def load_image_embedder(model_id: str | None = None, device: str = "auto") -> ImageEmbedder:
    """Load the model's encoders from the shared store (``ensure_model`` first; nothing is downloaded here)."""
    from finetune_studio.smart_brain.standalone_server import open_image_encoder

    spec = spec_for(model_id)
    if not on_disk(spec.id):
        raise FileNotFoundError(f"the image model {spec.id} is not on this machine yet: build the Smart Brain with "
                                "Image search on (the build downloads it)")
    image_dir, text_dir = model_dirs(spec)
    enc = open_image_encoder(image_dir, text_dir, resolve_device(device))

    def encode_texts(texts: str | list[str]) -> np.ndarray:
        single = isinstance(texts, str)
        out = enc.encode_texts([texts] if single else list(texts))
        return out[0] if single else out

    return ImageEmbedder(enc.encode_images, encode_texts, enc.dim, spec.id)


def cached_image_embedder(model_id: str | None = None, device: str = "auto") -> ImageEmbedder:
    """Load through the Smart Brain model cache (``release_brain_models`` frees it with the text models)."""
    from finetune_studio.smart_brain.model_cache import cached_image_embedder as cached

    return cached(model_id=model_id, device=device)


def image_from_bytes(data: bytes) -> Any:
    from finetune_studio.smart_brain.standalone_server import decode_image

    return decode_image(data)


def _image_format(data: bytes) -> tuple[str, str]:
    """(suffix, mime) of the bytes as PIL reads them."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
    except Exception:  # noqa: BLE001 - unreadable bytes: thumbnail() reports them; the suffix only names a file
        fmt = ""
    return {"PNG": (".png", "image/png"), "JPEG": (".jpg", "image/jpeg"), "WEBP": (".webp", "image/webp"),
            "GIF": (".gif", "image/gif"), "TIFF": (".tif", "image/tiff"),
            "BMP": (".bmp", "image/bmp")}.get(fmt, (".png", "image/png"))


def thumbnail(data: bytes, side: int = THUMB_SIDE) -> bytes:
    """A JPEG at most ``side`` px on its longest side (transparency on white)."""
    im = image_from_bytes(data)
    im.thumbnail((side, side))
    out = io.BytesIO()
    im.save(out, "JPEG", quality=82, optimize=True)
    return out.getvalue()


@dataclass(frozen=True)
class ImageAsset:
    """One picture with its source document, where in it (page / slide / paragraph / sheet) and what the parse read."""

    document_id: str
    filename: str
    source: str
    locator: str
    mime_type: str
    width: int
    height: int
    data: bytes
    figure: int = 0            # the figure number of the parsed text's ``[Figure N · …]`` block (0 = none)
    kind: str = "picture"
    context: str = ""          # text around the picture (caption, slide text, page text)
    description: str = ""      # the parse's reading of it: OCR or the Vision reader
    read_by: str = ""          # "OCR, not checked", the reader's name, or "" (nothing read it)
    source_sha: str = ""       # the source's parsed-text hash at extraction (incremental builds)
    file_id: str = ""          # the studio's library file (open / download the original; never in a package)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    @property
    def asset_id(self) -> str:
        identity = f"{self.document_id}\0{self.locator}\0{self.figure}\0{self.sha256}".encode()
        return hashlib.sha256(identity).hexdigest()[:24]

    @property
    def suffix(self) -> str:
        return _image_format(self.data)[0]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id, "document_id": self.document_id, "filename": self.filename,
            "source": self.source, "locator": self.locator, "figure": self.figure, "kind": self.kind,
            "mime_type": self.mime_type, "width": self.width, "height": self.height, "sha256": self.sha256,
            "bytes": len(self.data), "context": self.context[:1200], "description": self.description[:4000],
            "read_by": self.read_by, "source_sha": self.source_sha, "file_id": self.file_id,
            "path": f"image_assets/{self.asset_id}{self.suffix}",
            "thumbnail": f"image_assets/thumbs/{self.asset_id}.jpg",
        }
