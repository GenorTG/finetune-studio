"""Human-readable labels for model paths (HF cache, local dirs, repo ids)."""

from __future__ import annotations

import re

_HF_CACHE_RE = re.compile(r"models--(.+?)/snapshots/")


def model_label(path: str | None) -> str:
    """Render a short display name for a model path or Hugging Face id.

    HF hub cache paths ``.../models--<org>--<repo>/snapshots/<hash>`` become
    ``<org>/<repo>``. Short ``org/repo`` ids are kept as-is. Otherwise the
    last non-empty path segment is used.
    """
    if path is None:
        return ""
    raw = str(path).strip().rstrip("/\\")
    if not raw:
        return ""

    match = _HF_CACHE_RE.search(raw.replace("\\", "/"))
    if match:
        return match.group(1).replace("--", "/")

    normalized = raw.replace("\\", "/")
    # Plain Hugging Face id: org/repo (no absolute path)
    if (
        not normalized.startswith("/")
        and normalized.count("/") == 1
        and not normalized.startswith(".")
    ):
        return normalized

    parts = [p for p in normalized.split("/") if p]
    if not parts:
        return raw
    return parts[-1]
