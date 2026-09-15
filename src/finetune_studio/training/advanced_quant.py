"""Advanced quantization — GPTQ and imatrix-based GGUF.

Provides multiple quantization backends beyond basic llama.cpp quantize:
- GPTQ: Post-training quantization with group quantization
  (prefers modern ``gptqmodel``, falls back to ``auto_gptq``)
- imatrix GGUF: Uses importance matrix for higher quality quantization
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Literal

GptqBackendName = Literal["gptqmodel", "auto_gptq"]

_CALIBRATION_EXAMPLE_COUNT = 128


def is_gptqmodel_available() -> bool:
    """Return True when the modern ``gptqmodel`` package imports."""
    try:
        import gptqmodel  # noqa: F401
        return True
    except ImportError:
        return False


def is_auto_gptq_available() -> bool:
    """Return True when legacy ``auto_gptq`` imports."""
    try:
        import auto_gptq  # noqa: F401
        return True
    except ImportError:
        return False


def is_gptq_available() -> bool:
    """Return True when any supported GPTQ backend is importable."""
    return is_gptqmodel_available() or is_auto_gptq_available()


def is_optimum_available() -> bool:
    """Return True when ``optimum`` imports (needed for HF GPTQ inference)."""
    try:
        import optimum  # noqa: F401
        return True
    except ImportError:
        return False


def preferred_gptq_backend() -> GptqBackendName | None:
    """Prefer ``gptqmodel``; fall back to ``auto_gptq``; else None."""
    if is_gptqmodel_available():
        return "gptqmodel"
    if is_auto_gptq_available():
        return "auto_gptq"
    return None


def gptq_missing_backend_message() -> str:
    """Actionable error when neither GPTQ backend is installed."""
    return (
        "Neither gptqmodel nor auto_gptq is installed. "
        "Install with: uv pip install -e '.[gptq]' "
        "(gptqmodel + optimum for Transformers load), "
        "or choose format=merged / gguf instead."
    )


def gptq_optimum_inference_hint() -> str:
    """Hint when export backend is present but HF GPTQ load needs optimum."""
    return (
        "GPTQ export backend is installed, but Transformers GPTQ loading "
        "needs optimum. Install with: uv pip install -e '.[gptq]' "
        "(includes optimum)."
    )


def gptq_dependency_hints() -> dict[str, Any]:
    """Capability flags + actionable hints for export vs inference deps."""
    export_ok = is_gptq_available()
    optimum_ok = is_optimum_available()
    hints: list[str] = []
    if not export_ok:
        hints.append(gptq_missing_backend_message())
    elif not optimum_ok:
        hints.append(gptq_optimum_inference_hint())
    return {
        "gptq_export": export_ok,
        "optimum": optimum_ok,
        "gptq_inference_hf": export_ok and optimum_ok,
        "backend": preferred_gptq_backend(),
        "hints": hints,
        "hint": hints[0] if hints else "",
    }


def verify_gptq_artifacts(gptq_dir: str) -> dict[str, Any]:
    """Require a non-empty GPTQ export dir (config + weight file).

    Returns ``ok``, ``files``, ``output_dir``, ``size_bytes``, ``error``.
    """
    if not os.path.isdir(gptq_dir):
        return {
            "ok": False,
            "files": [],
            "output_dir": gptq_dir,
            "size_bytes": 0,
            "error": f"GPTQ directory missing: {gptq_dir}",
        }
    weight_suffixes = (".safetensors", ".bin")
    files: list[str] = []
    try:
        names = os.listdir(gptq_dir)
    except OSError as e:
        return {
            "ok": False,
            "files": [],
            "output_dir": gptq_dir,
            "size_bytes": 0,
            "error": f"Cannot read GPTQ directory {gptq_dir}: {e}",
        }

    has_config = "config.json" in names or "quantize_config.json" in names
    for name in names:
        path = os.path.join(gptq_dir, name)
        try:
            if not os.path.isfile(path) or os.path.getsize(path) <= 0:
                continue
        except OSError:
            continue
        lower = name.lower()
        if lower.endswith(weight_suffixes):
            files.append(path)

    if not has_config or not files:
        return {
            "ok": False,
            "files": files,
            "output_dir": gptq_dir,
            "size_bytes": _dir_size(gptq_dir),
            "error": (
                "No non-empty GPTQ artifacts found (need config.json / "
                "quantize_config.json and at least one weight file). "
                "Install gptqmodel or auto-gptq and retry, or choose "
                "format=merged."
            ),
        }
    return {
        "ok": True,
        "files": files,
        "output_dir": gptq_dir,
        "size_bytes": _dir_size(gptq_dir),
        "error": None,
    }


def calibration_example_texts(
    count: int = _CALIBRATION_EXAMPLE_COUNT,
) -> list[str]:
    """Default GPTQ calibration sentences (shared across backends)."""
    n = max(1, int(count))
    return [
        f"This is calibration example number {i} for GPTQ quantization."
        for i in range(n)
    ]


def quantize_gptq(
    model_path: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.01,
) -> dict[str, Any]:
    """Quantize a model using GPTQ (Post-training Quantization).

    Prefers the modern ``gptqmodel`` backend when installed; otherwise uses
    ``auto_gptq``. Both paths share the same calibration example texts.

    Args:
        model_path: Path to the model (safetensors)
        output_dir: Where to save the quantized model
        bits: Bit width (4 is standard)
        group_size: Quantization group size
        damp_percent: Damping percentage for numerical stability

    Returns:
        {output_dir, size_bytes, size_human, bits, group_size, method, backend}
    """
    backend = preferred_gptq_backend()
    if backend is None:
        raise ImportError(gptq_missing_backend_message())

    os.makedirs(output_dir, exist_ok=True)
    texts = calibration_example_texts()

    if backend == "gptqmodel":
        result = _quantize_with_gptqmodel(
            model_path=model_path,
            output_dir=output_dir,
            bits=bits,
            group_size=group_size,
            damp_percent=damp_percent,
            calibration_texts=texts,
        )
    else:
        result = _quantize_with_auto_gptq(
            model_path=model_path,
            output_dir=output_dir,
            bits=bits,
            group_size=group_size,
            damp_percent=damp_percent,
            calibration_texts=texts,
        )

    _copy_runtime_sidecars(model_path, output_dir)
    size = _dir_size(output_dir)
    result.update(
        {
            "output_dir": output_dir,
            "size_bytes": size,
            "size_human": _human_size(size),
            "bits": bits,
            "group_size": group_size,
            "method": "gptq",
            "backend": backend,
        }
    )
    return result


def _quantize_with_gptqmodel(
    model_path: str,
    output_dir: str,
    bits: int,
    group_size: int,
    damp_percent: float,
    calibration_texts: list[str],
) -> dict[str, Any]:
    """Run GPTQ via ``gptqmodel`` (GPTQModel + GPTQConfig/QuantizeConfig)."""
    import inspect

    from gptqmodel import GPTQModel

    try:
        from gptqmodel import GPTQConfig as _Config
    except ImportError:  # older gptqmodel
        from gptqmodel import QuantizeConfig as _Config

    wanted: dict[str, Any] = {
        "bits": bits,
        "group_size": group_size,
        "damp_percent": damp_percent,
        "desc_act": False,
        "static_groups": True,
        "sym": True,
        "true_sequential": True,
    }
    accepted = set(inspect.signature(_Config).parameters)
    quantize_config = _Config(
        **{k: v for k, v in wanted.items() if k in accepted}
    )
    model = GPTQModel.load(
        model_path,
        quantize_config,
        trust_remote_code=True,
    )
    quantize_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "calibration_data_min_length": 1,
    }
    quantize_params = set(inspect.signature(model.quantize).parameters)
    model.quantize(
        calibration_texts,
        **{k: v for k, v in quantize_kwargs.items() if k in quantize_params},
    )
    model.save(output_dir)
    return {}


def _quantize_with_auto_gptq(
    model_path: str,
    output_dir: str,
    bits: int,
    group_size: int,
    damp_percent: float,
    calibration_texts: list[str],
) -> dict[str, Any]:
    """Run GPTQ via legacy ``auto_gptq``."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    quantize_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        damp_percent=damp_percent,
        desc_act=False,
        static_groups=True,
        sym=True,
        true_sequential=True,
    )

    model = AutoGPTQForCausalLM.from_pretrained(
        model_path, quantize_config, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True,
    )
    examples = [tokenizer(text) for text in calibration_texts]
    model.quantize(examples)
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {}


def _copy_runtime_sidecars(model_path: str, output_dir: str) -> None:
    """Copy optional prompt / chat-template files next to the GPTQ weights."""
    for name in ("system_prompt.txt", "chat_template.jinja"):
        src = os.path.join(model_path, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, name))


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
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, check=False,
    )
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

    cmd = [imatrix_bin, "-m", model_path, "-f", calibration_data, "-o", output_path, "--ctx", str(n_ctx)]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, check=False,
    )

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
            f.writelines(
                f"This is calibration sentence number {i}. " * 20 + "\n"
                for i in range(200)
            )
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
