"""JSON + pandas helpers.

Single responsibility: tiny I/O utilities shared by every other module in this package.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_json(path: Path, data: dict, indent: int = 2) -> None:
    path.write_text(json.dumps(data, indent=indent, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def try_import_pandas():
    """Lazily import pandas, raising a clear error if it's missing."""
    try:
        import pandas as pd
        return pd
    except ImportError as e:
        raise RuntimeError(
            "pyarrow + pandas required for chunk parquet. Install: pip install pandas pyarrow"
        ) from e
