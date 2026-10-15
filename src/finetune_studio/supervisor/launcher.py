"""How was this supervisor started: by the finetune-studio systemd unit, or by hand?"""
from __future__ import annotations

from pathlib import Path

UNIT_NAME = "finetune-studio.service"


def detect(cgroup_file: Path = Path("/proc/self/cgroup")) -> str:
    """``systemd`` when this process lives in the finetune-studio unit's cgroup, else ``manual``.

    ``INVOCATION_ID`` cannot tell: any shell started by another systemd service inherits it, so a
    supervisor started by hand from such a shell would claim to be the unit.
    """
    try:
        text = cgroup_file.read_text(encoding="utf-8")
    except OSError:
        return "manual"
    return "systemd" if any(line.rstrip().endswith("/" + UNIT_NAME) for line in text.splitlines()) else "manual"
