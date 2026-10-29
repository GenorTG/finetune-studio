"""Convert downloaded base models to GGUF with the app's own converter (finetune_studio.training.gguf_convert).

    .venv/bin/python tests/corpus/quillmoor/bench/base_gguf.py Qwen__Qwen3.5-4B:q8_0 unsloth__Llama-3.1-8B-Instruct:q4_k_m ...

Output: ~/.finetune-studio/bench-gguf/<model>/model-<quant>.gguf (deleted with the benchmark's cleanup). CPU only.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HF = Path.home() / ".finetune-studio" / "hf_models"
OUT = Path.home() / ".finetune-studio" / "bench-gguf"


def main(argv: list[str]) -> int:
    from finetune_studio.training.gguf_convert import convert_merged_to_gguf
    for spec in argv:
        name, _, quant = spec.partition(":")
        quant = quant or "q8_0"
        t0 = time.time()
        res = convert_merged_to_gguf(str(HF / name), str(OUT / name), [quant])
        print(json.dumps({"model": name, "quant": quant, "ok": res.get("ok"), "seconds": round(time.time() - t0),
                          "files": res.get("files"), "error": (res.get("error") or "")[:400]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
