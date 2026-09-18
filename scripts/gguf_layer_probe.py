"""Dump GGUF header keys relevant to layer counts (diagnostic helper)."""
from __future__ import annotations

import sys

import numpy as np
from gguf import GGUFReader

path = sys.argv[1]
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
    except Exception:
        print(key, "(non-numeric)")
