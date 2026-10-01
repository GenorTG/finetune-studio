"""Advanced quantization — imatrix-based GGUF.

imatrix GGUF uses an importance matrix for higher quality quantization
than basic llama.cpp quantize. Tool discovery is shared with ``gguf_convert``.
"""

from __future__ import annotations

import os
import subprocess
import sys

from finetune_studio.training.gguf_convert import (
    find_gguf_convert_script,
    find_llama_quantize,
    normalize_gguf_quant,
)


def quantize_gguf_imatrix(
    model_path: str,
    output_dir: str,
    imatrix_path: str,
    quants: list[str] | None = None,
) -> dict:
    """Quantize a merged HF model to GGUF using a precomputed importance matrix.

    The importance matrix tells the quantizer which weights matter more,
    so it can allocate more bits to important weights and fewer to unimportant.

    Args:
        model_path: Path to the merged model (safetensors)
        output_dir: Where to save the GGUF files (under ``<output_dir>/gguf``)
        imatrix_path: Path to an existing llama.cpp importance-matrix file
        quants: List of quant types (default: q4_k_m, q5_k_m, q8_0)

    Returns:
        {output_dir, exported: {quant: {path, size} | {error}}, size_bytes, method}
        or {error} when tools / the imatrix file are missing or conversion fails.
    """
    if quants is None:
        quants = ["q4_k_m", "q5_k_m", "q8_0"]

    if not imatrix_path or not os.path.isfile(imatrix_path):
        return {"error": f"imatrix file not found: {imatrix_path or '(none given)'}"}

    convert_script = find_gguf_convert_script()
    if not convert_script:
        return {"error": "llama.cpp convert script not found"}
    quant_bin = find_llama_quantize()
    if not quant_bin:
        return {"error": "llama-quantize not found"}

    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    # Step 1: Convert to F16 GGUF
    f16_file = os.path.join(gguf_dir, "model-f16.gguf")
    cmd = [sys.executable, convert_script, model_path, "--outfile", f16_file, "--outtype", "f16"]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, check=False,
    )
    if result.returncode != 0:
        return {"error": f"convert failed: {result.stderr[:200]}"}

    # Step 2: Quantize with imatrix. llama-quantize parses options only before
    # the positional args, so --imatrix must come first.
    exported = {}
    for quant in quants:
        quant_clean = normalize_gguf_quant(quant)
        out_file = os.path.join(gguf_dir, f"model-{quant_clean}.gguf")
        cmd = [quant_bin, "--imatrix", imatrix_path, f16_file, out_file, quant_clean.upper()]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200, check=False,
        )
        if result.returncode == 0 and os.path.isfile(out_file):
            exported[quant] = {"path": out_file, "size": os.path.getsize(out_file)}
        else:
            exported[quant] = {"error": result.stderr[:100]}

    return {
        "output_dir": gguf_dir,
        "exported": exported,
        "size_bytes": sum(v.get("size", 0) for v in exported.values()),
        "method": "gguf_imatrix",
    }
