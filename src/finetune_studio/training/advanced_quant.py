"""Advanced quantization — AWQ, GPTQ, and imatrix-based GGUF.

Provides multiple quantization backends beyond basic llama.cpp quantize:
- AWQ: Activation-aware quantization, better for GPU inference
- GPTQ: Post-training quantization with group quantization
- imatrix GGUF: Uses importance matrix for higher quality quantization
"""

from __future__ import annotations

import os
import subprocess


def is_awq_available() -> bool:
    """Check if autoawq is installed."""
    try:
        import autoawq  # noqa: F401
        return True
    except ImportError:
        return False


def is_gptq_available() -> bool:
    """Check if auto-gptq is installed."""
    try:
        import auto_gptq  # noqa: F401
        return True
    except ImportError:
        return False


def quantize_awq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    version: str = "GEMM",
) -> dict:
    """Quantize a model using AWQ (Activation-aware Weight Quantization).

    Args:
        model_path: Path to the model (safetensors)
        output_dir: Where to save the quantized model
        bits: Bit width (4 is standard)
        group_size: Quantization group size
        version: AWQ kernel version (GEMM or GEMV)

    Returns:
        {output_dir, size_bytes, size_human, bits, group_size}
    """
    from autoawq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    model = AutoAWQForCausalLM.from_pretrained(model_path, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }

    model.quantize(tokenizer, quant_config=quant_config)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    size = _dir_size(output_dir)
    return {
        "output_dir": output_dir,
        "size_bytes": size,
        "size_human": _human_size(size),
        "bits": bits,
        "group_size": group_size,
        "method": "awq",
    }


def quantize_gptq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
) -> dict:
    """Quantize a model using GPTQ (Post-training Quantization).

    Args:
        model_path: Path to the model (safetensors)
        output_dir: Where to save the quantized model
        bits: Bit width (4 is standard)
        group_size: Quantization group size
        damp_percent: Damping percentage for numerical stability

    Returns:
        {output_dir, size_bytes, size_human, bits, group_size}
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=False,
        static_groups=True,
        sym=True,
        true_sequential=True,
    )

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Use calibration data (simple default)
    examples = []
    for i in range(128):
        text = f"This is calibration example number {i} for GPTQ quantization."
        examples.append(tokenizer(text))

    model.quantize(examples)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    size = _dir_size(output_dir)
    return {
        "output_dir": output_dir,
        "size_bytes": size,
        "size_human": _human_size(size),
        "bits": bits,
        "group_size": group_size,
        "method": "gptq",
    }


def quantize_gguf_imatrix(
    model_path: str,
    output_dir: str,
    imatrix_path: str,
    quants: list[str] | None = None,
) -> dict:
    """Quantize a model to GGUF using an importance matrix for higher quality.

    The importance matrix tells the quantizer which weights matter more,
    so it can allocate more bits to important weights and fewer to unimportant.

    Args:
        model_path: Path to the merged model (safetensors)
        output_dir: Where to save the GGUF files
        imatrix_path: Path to the importance matrix file
        quants: List of quant types (default: all common ones)

    Returns:
        {output_dir, exported: {quant: {path, size}}}
    """
    if quants is None:
        quants = ["q4_k_m", "q5_k_m", "q8_0"]

    gguf_dir = os.path.join(output_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    # Find llama.cpp convert script
    convert_script = None
    candidates = [
        os.path.expanduser("~/llama.cpp/convert.py"),
        os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
        "/usr/local/bin/convert-hf-to-gguf.py",
    ]
    for c in candidates:
        if os.path.isfile(c):
            convert_script = c
            break

    if not convert_script:
        return {"error": "llama.cpp not found"}

    # Step 1: Convert to F16 GGUF
    f16_file = os.path.join(gguf_dir, "model-f16.gguf")
    cmd = ["python3", convert_script, model_path, "--outfile", f16_file, "--outtype", "f16"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        return {"error": f"convert failed: {result.stderr[:200]}"}

    # Step 2: Quantize with imatrix
    quant_bin = os.path.join(os.path.dirname(convert_script), "quantize")
    if not os.path.isfile(quant_bin):
        quant_bin = "quantize"

    exported = {}
    for quant in quants:
        quant_clean = quant.lower().replace("-", "_").replace(".", "_")
        out_file = os.path.join(gguf_dir, f"model-{quant_clean}.gguf")
        cmd = [quant_bin, f16_file, out_file, quant_clean, "--imatrix", imatrix_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if result.returncode == 0 and os.path.isfile(out_file):
            exported[quant] = {"path": out_file, "size": os.path.getsize(out_file)}
        else:
            exported[quant] = {"error": result.stderr[:100]}

    return {
        "output_dir": gguf_dir,
        "exported": exported,
        "method": "gguf_imatrix",
    }


def generate_imatrix(
    model_path: str,
    output_path: str,
    calibration_data: str | None = None,
    n_ctx: int = 512,
) -> dict:
    """Generate an importance matrix for GGUF quantization.

    The importance matrix is computed by running the model on calibration
    data and measuring which activations are most important.

    Args:
        model_path: Path to the F16 GGUF model
        output_path: Where to save the imatrix file
        calibration_data: Path to calibration text data (default: use model's own tokenizer)
        n_ctx: Context length for calibration

    Returns:
        {imatrix_path, size_bytes}
    """
    # Find llama.cpp's imatrix binary
    imatrix_bin = None
    candidates = [
        os.path.expanduser("~/llama.cpp/imatrix"),
        "/usr/local/bin/imatrix",
    ]
    for c in candidates:
        if os.path.isfile(c):
            imatrix_bin = c
            break

    if not imatrix_bin:
        return {"error": "llama.cpp imatrix not found"}

    if not calibration_data:
        # Use a default calibration text
        calibration_data = _default_calibration_path()

    cmd = [imatrix_bin, "-m", model_path, "-f", calibration_data, "-o output_path, "--ctx", str(n_ctx)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        return {"error": f"imatrix failed: {result.stderr[:200]}"}

    size = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    return {"imatrix_path": output_path, "size_bytes": size}


def _default_calibration_path() -> str:
    """Get or create a default calibration text file."""
    path = "/tmp/fts_calibration.txt"
    if not os.path.isfile(path):
        # Write a simple calibration text
        with open(path, "w") as f:
            for i in range(200):
                f.write(f"This is calibration sentence number {i}. " * 20 + "\n")
    return path


def _dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"
