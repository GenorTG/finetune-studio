"""Cross-process signal that the WebUI's in-memory model list needs a rescan."""
from __future__ import annotations

import os
import time
from pathlib import Path

from finetune_studio.data.fs.paths import root as fts_root


def _generation_file() -> Path:
    return fts_root() / "run" / "model-registry-generation"


def read_generation() -> int:
    """Return the latest invalidation generation, or zero when none exists."""
    try:
        return int(_generation_file().read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return 0


def invalidate_model_registry() -> int:
    """Atomically publish a new generation for other app processes to observe."""
    path = _generation_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    generation = max(time.time_ns(), read_generation() + 1)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{generation}\n", encoding="ascii")
    temporary.replace(path)
    return generation
