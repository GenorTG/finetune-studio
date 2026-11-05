"""HuggingFace cache location + tokenizer loading — one honest source of truth.

HF itself resolves the hub cache as ``HF_HUB_CACHE`` > ``$HF_HOME/hub`` >
``~/.cache/huggingface/hub``. Every place in the app that needs that path goes
through here so ``HF_HOME`` / ``HF_HUB_CACHE`` are honored everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def hf_home() -> Path:
    """HF root dir: ``$HF_HOME`` or ``~/.cache/huggingface``."""
    env = os.environ.get("HF_HOME", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "huggingface"


def hf_hub_cache() -> Path:
    """HF hub cache dir: ``$HF_HUB_CACHE`` > ``$HF_HOME/hub`` > default."""
    env = os.environ.get("HF_HUB_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    return hf_home() / "hub"


def is_in_hub_cache(path: str) -> bool:
    """True when ``path`` lives under the effective hub cache (or default layout)."""
    norm = str(path).replace("\\", "/").lower()
    if "huggingface/hub" in norm:
        return True
    root = str(hf_hub_cache()).replace("\\", "/").lower().rstrip("/")
    return norm == root or norm.startswith(root + "/")


def hf_cache_source() -> str:
    """Which setting decided the cache path (for the Settings page)."""
    if os.environ.get("HF_HUB_CACHE", "").strip():
        return "HF_HUB_CACHE"
    if os.environ.get("HF_HOME", "").strip():
        return "HF_HOME"
    return "default (~/.cache/huggingface)"


def trust_remote_code_for(path: str) -> bool:
    """Whether to run the model repo's own Python code: only when transformers has no implementation of its type.

    Some repos ship code alongside a type transformers implements itself. That code is pinned to the transformers of
    its day and breaks on a newer one: Phi-4-mini's ``modeling_phi3.py`` fails with "cannot import name 'LossKwargs'".
    """
    import json
    from pathlib import Path

    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    try:
        model_type = json.loads((Path(path) / "config.json").read_text(encoding="utf-8")).get("model_type")
    except (OSError, ValueError):
        return True   # a repo id or an unreadable config: the loader decides from what it downloads
    return model_type not in CONFIG_MAPPING


def load_tokenizer(path: str, **kwargs: Any):
    """``AutoTokenizer.from_pretrained`` with the Mistral-regex fix applied.

    transformers 4.57 warns on every load of a Mistral-family tokenizer unless
    ``fix_mistral_regex`` is passed explicitly (and tokenizes incorrectly
    without it). Older/newer versions that reject the kwarg fall back cleanly.
    """
    from transformers import AutoTokenizer

    kwargs.setdefault("trust_remote_code", trust_remote_code_for(path))
    try:
        return AutoTokenizer.from_pretrained(path, fix_mistral_regex=True, **kwargs)
    except TypeError:
        return AutoTokenizer.from_pretrained(path, **kwargs)
