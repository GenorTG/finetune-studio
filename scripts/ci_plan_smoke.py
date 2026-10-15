#!/usr/bin/env python3
"""Smoke `bash install.sh --plan` on fake hardware for every vendor; stdlib only, runs without a venv.

Usage:
    python3 scripts/ci_plan_smoke.py

Each case writes an ``FTS_ACCEL_FIXTURE`` JSON (the same hook tests/test_accel_plan.py uses), runs the
real shell installer in ``--plan`` mode and checks that stdout is the plan JSON and nothing else, with
the vendor / torch backend / llama.cpp backend the planner must pick. CI runs it on Linux and on macOS
(bash 3.2, BSD userland), which is the one place the installer's portability is exercised for real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

SMI_QUERY = "nvidia-smi --query-gpu=index,name,driver_version,compute_cap --format=csv,noheader"
NVCC = "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.4, V13.4.92\n"

NVIDIA = {
    "system": "Linux", "machine": "x86_64",
    "which": ["nvidia-smi", "nvcc"],
    "commands": {
        SMI_QUERY: "0, NVIDIA GeForce RTX 3090, 580.1, 8.6\n",
        "nvidia-smi": "| NVIDIA-SMI 580.1  Driver Version: x   CUDA UMD Version: 13.0 |\n",
        "/usr/local/cuda/bin/nvcc --version": NVCC, "nvcc --version": NVCC,
    },
    "files": {"/usr/local/cuda/bin/nvcc": "x",
              "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13": "x",
              "/usr/local/cuda/lib64/libcudart.so.12": "x"},
}
AMD = {
    "system": "Linux", "machine": "x86_64",
    "which": ["rocm-smi", "rocminfo"],
    "files": {"/opt/rocm/.info/version": "6.4.1", "/dev/kfd": "", "/opt/rocm/bin/hipcc": "x"},
    "commands": {"rocminfo": "Name: gfx1100\nName: AMD Ryzen CPU\n",
                 "rocm-smi --showproductname": "Card series: Radeon RX 7900 XTX\n"},
}
INTEL = {
    "system": "Linux", "machine": "x86_64",
    "which": ["sycl-ls"],
    "commands": {"sycl-ls": "[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) B580 Graphics\n"},
}
APPLE = {"system": "Darwin", "machine": "arm64"}
NONE = {"system": "Linux", "machine": "x86_64"}

# name, fixture, installer flags, expected plan fields
CASES: list[tuple[str, dict[str, Any], list[str], dict[str, str]]] = [
    ("nvidia", NVIDIA, [], {"vendor": "nvidia", "torch_backend": "cuda", "llama_backend": "cuda"}),
    ("amd", AMD, [], {"vendor": "amd", "torch_backend": "hip", "llama_backend": "hip"}),
    ("intel", INTEL, [], {"vendor": "intel", "torch_tag": "xpu"}),
    ("apple", APPLE, [], {"vendor": "apple", "torch_backend": "mps", "llama_backend": "metal"}),
    ("none", NONE, [], {"vendor": "none", "torch_tag": "cpu"}),
    ("nvidia --cpu", NVIDIA, ["--cpu"], {"vendor": "none", "torch_tag": "cpu"}),
    ("none --gpu amd", NONE, ["--gpu", "amd"], {"vendor": "amd"}),
]


def run_case(tmp: Path, name: str, fixture: dict[str, Any], flags: list[str], expect: dict[str, str]) -> str | None:
    """Return a failure message, or None when the plan matches."""
    fx = tmp / "fixture.json"
    fx.write_text(json.dumps(fixture))
    # Only PATH/HOME and the fixture reach the installer: a runner's own GPU env must not leak in.
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", str(tmp)),
           "FTS_ACCEL_FIXTURE": str(fx), "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run(["bash", str(ROOT / "install.sh"), "--plan", *flags], capture_output=True, text=True,
                       env=env, cwd=ROOT, check=False, timeout=60)
    if r.returncode != 0:
        return f"exit {r.returncode}: {r.stderr.strip()[-400:]}"
    try:
        plan = json.loads(r.stdout)        # stdout must be the plan and nothing else
    except json.JSONDecodeError as exc:
        return f"stdout is not pure JSON ({exc}): {r.stdout[:200]!r}"
    wrong = {k: (plan.get(k), v) for k, v in expect.items() if plan.get(k) != v}
    return f"got/expected {wrong}" if wrong else None


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory() as td:
        for name, fixture, flags, expect in CASES:
            err = run_case(Path(td), name, fixture, flags, expect)
            print(f"{'FAIL' if err else 'ok  '} install.sh --plan [{name}]" + (f"  {err}" if err else ""))
            failed += err is not None
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
