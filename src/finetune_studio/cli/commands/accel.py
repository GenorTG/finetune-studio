"""`fts accel` — which accelerator Finetune Studio computes on, and why.

Exit status is nonzero ONLY on a degraded GPU host (GPU hardware present but
compute fell back to CPU), so it doubles as a health probe in scripts.
"""
from __future__ import annotations

import json
import sys


def format_report(d: dict, llama_warnings: list[str] | None = None) -> list[str]:
    llama_warnings = llama_warnings or []
    acc, torch, llama = d["accelerator"], d["torch"], d["llama_cpp"]
    lines = [
        f"Accelerator : {acc['torch_device']} — {acc['name']} ({acc['runtime']})",
    ]
    if acc["is_gpu"]:
        lines.append(f"Memory      : {acc['free_gb']:.1f} / {acc['total_gb']:.1f} GiB free"
                     f" · bf16={'yes' if acc['supports_bf16'] else 'no'}"
                     f" · {acc['device_count']} device(s)")
    lines.append("PyTorch     : " + (torch.get("error") or
                 f"{torch['version']} (cuda={torch.get('cuda')}, hip={torch.get('hip')}, xpu={torch.get('xpu')})"))
    if llama.get("installed"):
        be = ", ".join(llama.get("backends") or []) or "none (CPU build)"
        lines.append(f"llama.cpp   : {llama.get('version', '')} · GPU offload "
                     f"{'yes' if llama.get('gpu_offload') else 'NO'} · backends: {be}")
        for w in llama_warnings:
            lines.append(f"DEGRADED    : llama.cpp — {w}")
    else:
        lines.append("llama.cpp   : not installed")
    lines.append("Hardware    : " + (", ".join(d["hardware"]) or "no GPU found"))
    if d["policy"]:
        lines.append("Policy env  : " + " ".join(f"{k}={v}" for k, v in d["policy"].items()))
    if acc["degraded_reason"]:
        lines.append(f"DEGRADED    : {acc['degraded_reason']}")
    return lines


def cmd_accel(args) -> None:
    from finetune_studio import accel

    d = accel.describe()
    _, llama_warnings = accel.llama_gpu_kwargs()
    if not (d["accelerator"]["is_gpu"] and d["llama_cpp"]["installed"]):
        llama_warnings = []   # CPU-only host / no llama.cpp: nothing is degraded
    d["llama_cpp"]["warnings"] = llama_warnings
    if getattr(args, "json", False):
        print(json.dumps(d, indent=2))
    else:
        print("\n".join(format_report(d, llama_warnings)))
    if d["accelerator"]["degraded_reason"] or llama_warnings:
        sys.exit(1)
