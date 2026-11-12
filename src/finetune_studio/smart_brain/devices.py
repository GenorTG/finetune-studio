"""Device resolution for RAG embedders / rerankers (Studio side).

Policy: ``"auto"`` (the default everywhere) = GPU first through the
vendor-neutral accelerator layer (CUDA / ROCm / XPU / MPS), CPU only when the
host has no usable GPU. An explicit ``cpu`` / ``cuda:N`` / ``xpu`` / ``mps``
request is honoured as-is. The shipped standalone package carries its own
inline copy of this logic (``standalone_server._resolve_device``) because it
runs without finetune_studio installed.
"""
from __future__ import annotations

AUTO = ("", "auto")
# Encode batch size per device class: a GPU amortises launch cost over bigger
# batches; CPU stays small to bound RAM and latency.
GPU_BATCH = 64
CPU_BATCH = 16


def resolve_device(requested: str | None = "auto") -> str:
    """Map a user/config device string to a concrete torch device string."""
    req = (requested or "auto").strip().lower()
    if req not in AUTO:
        return req
    from finetune_studio.accel import get_accelerator
    return get_accelerator().torch_device  # "cpu" when no GPU is usable


def encode_batch_size(device: str) -> int:
    return CPU_BATCH if device.split(":")[0] == "cpu" else GPU_BATCH


def gpu_memory_report() -> str:
    """`` (2.1 of 24.0 GB free; held by: ComfyUI python 17.4 GB, llama-server 4.3 GB)`` or ``""`` (no nvidia-smi).

    Plain words for an out-of-memory error: which programs use the card this process was given
    (``CUDA_VISIBLE_DEVICES``, else GPU 0). Read-only: never stops anything."""
    import os
    import shutil
    import subprocess

    if not shutil.which("nvidia-smi"):
        return ""
    try:
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, check=True).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,process_name,used_memory",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
    rows = [[p.strip() for p in line.split(",")] for line in gpus.splitlines() if line.count(",") == 3]
    if not rows:
        return ""
    want = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")[0].strip()
    card = next((r for r in rows if want and (r[0] == want or r[1] == want or r[1].endswith(want))), rows[0])
    holders = []
    for line in apps.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] != card[1] or not parts[2].isdigit() or int(parts[2]) < 512:
            continue
        holders.append((int(parts[2]), f"{_process_label(parts[1])} {int(parts[2]) / 1024:.1f} GB"))
    holders.sort(reverse=True)
    out = f"GPU {card[0]}: {float(card[2]) / 1024:.1f} of {float(card[3]) / 1024:.1f} GB free"
    if holders:
        out += "; held by: " + ", ".join(h for _, h in holders[:4])
    return f" ({out})"


def _process_label(exe: str) -> str:
    """A readable name: the program, plus the app folder for a Python process (``ComfyUI python``)."""
    import os

    base = os.path.basename(exe) or exe
    if base.startswith("python"):
        parts = [p for p in exe.split("/") if p and p not in (".venv", "bin", "venv", "home")]
        app = next((p for p in reversed(parts[:-1]) if not p.startswith(("python", "."))), "")
        return f"{app} python" if app else base
    return base

