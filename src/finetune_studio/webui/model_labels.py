"""Human-readable labels for model paths (HF cache, local dirs, repo ids)."""

from __future__ import annotations

import re

_HF_CACHE_RE = re.compile(r"models--(.+?)/snapshots/")
_RUN_RE = re.compile(r"(^|/)runs/[^/]+(/|$)")


def model_label(path: str | None) -> str:
    """Render a short display name for a model path or Hugging Face id.

    HF hub cache paths ``.../models--<org>--<repo>/snapshots/<hash>`` become
    ``<org>/<repo>``, the app's ``hf_models/<org>__<repo>`` downloads too, and
    a run's output gets its project/run name. Short ``org/repo`` ids are kept
    as-is. Otherwise the last non-empty path segment is used.
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
    # The app's own downloads live in hf_models/<org>__<repo>.
    if len(parts) >= 2 and parts[-2] == "hf_models" and "__" in parts[-1]:
        return parts[-1].replace("__", "/", 1)
    # A training run's output (…/runs/<rid>/merged): the folder name alone is just 'merged'.
    if _RUN_RE.search(normalized):
        from finetune_studio.naming import display_for_path
        return display_for_path(raw) or parts[-1]
    return parts[-1]
