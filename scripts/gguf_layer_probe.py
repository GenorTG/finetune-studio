"""Dump GGUF header keys relevant to layer counts (diagnostic helper)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gguf import GGUFReader


def dump_gguf_headers(path: Path) -> None:
    """Print GGUF header keys relevant to layer counts."""
    r = GGUFReader(path)
    hits = [
        k for k in r.fields
        if "block" in k.lower() or "context_length" in k.lower() or "layer" in k.lower()
    ][:12]
    for key in hits:
        f = r.fields[key]
        try:
            arr = r.data[f.data_start_index:f.data_start_index + f.length]
            print(key, "=", int(arr.astype(np.int64)[0]))
        except (IndexError, TypeError, ValueError):
            print(key, "(non-numeric)")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: gguf_layer_probe.py <path-to.gguf>", file=sys.stderr)
        return 1
    dump_gguf_headers(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
