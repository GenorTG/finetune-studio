#!/usr/bin/env python3
"""Exact versions a fresh install gets: the stack that was tested, recorded in ``pins/``.

    python3 scripts/pin_versions.py write    record this checkout's .venv, Heretic overlay and llama.cpp (make pins)
    python3 scripts/pin_versions.py check    list what differs from the pins; exit 1 when something does

Moving to newer versions: `FTS_UNPINNED=1 bash update.sh --repair` on a branch, test it, `make pins`, PR (CONTRIBUTING.md).
Stdlib only: install.sh and update.sh run it with whatever Python there is.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS = ROOT / "pins"
TORCH_FAMILY = ("torch", "torchvision", "torchaudio")
# Decided by which PyTorch wheel the GPU needs, not by us: pinned here they would fail on every other wheel index.
_WHEEL_DEPS = ("nvidia-", "cuda-")
_HEADER = "# Tested stack, written by scripts/pin_versions.py (make pins): do not edit by hand.\n"


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def base_version(version: str) -> str:
    """``2.15.0+cu132`` -> ``2.15.0``: the build tag follows the GPU."""
    return version.split("+", 1)[0]


_LIST = ("import importlib.metadata as m, json, sysconfig\n"
         "print(json.dumps({d.metadata['Name']: d.version\n"
         "    for d in m.distributions(path=[sysconfig.get_paths()['purelib']]) if d.metadata['Name']}))")
_EDITABLE = ("finetune-studio", "heretic-llm")


def freeze(python: Path) -> dict[str, str]:
    """``{normalized name: version}`` in that environment's own site-packages (Heretic's overlay also sees the app's)."""
    out = subprocess.run([str(python), "-c", _LIST], capture_output=True, text=True, check=True).stdout
    found = {normalize(n): v for n, v in json.loads(out.strip().splitlines()[-1]).items()}
    return {n: v for n, v in found.items() if n not in _EDITABLE}


def read_pins(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    found: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            found[normalize(name)] = version
    return found


def _render(versions: dict[str, str], note: str) -> str:
    return _HEADER + f"# {note}\n" + "".join(f"{n}=={versions[n]}\n" for n in sorted(versions))


def app_pins(installed: dict[str, str]) -> dict[str, str]:
    return {n: v for n, v in installed.items() if n not in TORCH_FAMILY and not n.startswith(_WHEEL_DEPS)}


def torch_pins(installed: dict[str, str]) -> dict[str, str]:
    return {n: base_version(installed[n]) for n in TORCH_FAMILY if n in installed}


def heretic_python() -> Path:
    root = Path(os.environ.get("FTS_ROOT") or Path.home() / ".finetune-studio")
    return root / "heretic" / "venv" / "bin" / "python"


def write(venv: Path, llama_dir: Path) -> int:
    python = venv / "bin" / "python"
    installed = freeze(python)
    torch_full = installed.get("torch", "")
    if "+" not in torch_full:
        print(f"refusing: torch {torch_full or '(none)'} has no GPU build tag; record the tested GPU stack", file=sys.stderr)
        return 2
    version = subprocess.run([str(python), "-c", "import sys;print('%d.%d'%sys.version_info[:2])"], capture_output=True,
                             text=True, check=True).stdout.strip()
    note = f"recorded {datetime.date.today()} on Python {version}, torch {torch_full}"
    PINS.mkdir(exist_ok=True)
    (PINS / "app.txt").write_text(_render(app_pins(installed), note), encoding="utf-8")
    (PINS / "torch.txt").write_text(_render(torch_pins(installed), note), encoding="utf-8")
    overlay = heretic_python()
    if overlay.exists():
        (PINS / "heretic.txt").write_text(_render(freeze(overlay), note), encoding="utf-8")
    else:
        print("no Heretic overlay here: pins/heretic.txt left as it is", file=sys.stderr)
    if (llama_dir / ".git").exists():
        commit = subprocess.run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        (PINS / "llama-cpp.txt").write_text(f"{_HEADER}# recorded {datetime.date.today()}\n{commit}\n", encoding="utf-8")
    print(f"wrote {PINS}: {note}")
    return 0


def drift(pins: dict[str, str], installed: dict[str, str]) -> list[str]:
    """``name installed -> pinned`` for what is installed at another version. Absent packages are the plan's call."""
    return [f"{n} {installed[n]} -> {want}" for n, want in sorted(pins.items())
            if n in installed and base_version(installed[n]) != base_version(want)]


def check(venv: Path) -> int:
    installed = freeze(venv / "bin" / "python")
    found = drift({**read_pins(PINS / "app.txt"), **read_pins(PINS / "torch.txt")}, installed)
    overlay = heretic_python()
    if overlay.exists():
        found += [f"heretic: {d}" for d in drift(read_pins(PINS / "heretic.txt"), freeze(overlay))]
    if not found:
        print("installed versions match the tested pins.")
        return 0
    print(f"{len(found)} package(s) differ from the tested pins (installed -> tested):")
    for line in found:
        print(f"  {line}")
    return 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("write", "check"))
    ap.add_argument("--venv", default=str(ROOT / ".venv"))
    ap.add_argument("--llama-cpp", default=str(ROOT / ".llama.cpp"))
    a = ap.parse_args(argv)
    if a.action == "write":
        return write(Path(a.venv), Path(a.llama_cpp))
    return check(Path(a.venv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
