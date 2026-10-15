"""The public tree names no private machine, home path or LAN address, and holds no secret-shaped string.

Hosts and paths come from config (``FTS_BASE``, ``FTS_ROOT``); comments say "the GPU host" / "the dev box". Patterns are
spelled in pieces so this file does not match itself.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRIVATE = re.compile("|".join([
    "genor" "box", "fan" "-dragon", "stealth" "-dragon",
    r"/ho" r"me/" "genor",                             # the maintainers' real home dirs
    r"\b192\.168\.\d+\.\d+",
    r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+\b",   # tailnet (CGNAT) addresses
]), re.IGNORECASE)
# The fictional Korvane corpus invents 10.x subnets as test facts; outside it a 10.x address is a real one.
LAN10 = re.compile(r"\b10\.\d+\.\d+\.\d+\b")
FICTION = "tests/corpus/"
SECRET = re.compile("|".join([
    r"\bsk-[A-Za-z0-9_-]{20,}", r"\bhf_[A-Za-z0-9]{30,}", r"\bghp_[A-Za-z0-9]{30,}", r"\bgithub_pat_[A-Za-z0-9_]{30,}",
    "-----BEGIN [A-Z ]*PRIV" "ATE KEY-----", r"\btsk" r"ey-[A-Za-z0-9-]{10,}",
]))


def _tracked_text_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
    files = []
    for name in out:
        p = ROOT / name
        if name and p.is_file() and p.stat().st_size < 5_000_000:
            files.append(p)
    return files


def _hits(pattern: re.Pattern[str], skip: str | None = None) -> list[str]:
    found = []
    for p in _tracked_text_files():
        if skip and str(p.relative_to(ROOT)).startswith(skip):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.split("\n"), 1):
            if pattern.search(line):
                found.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:120]}")
    return found


def test_no_private_hosts_paths_or_lan_addresses() -> None:
    hits = _hits(PRIVATE) + _hits(LAN10, skip=FICTION)
    assert not hits, "private host/path/address in the public tree:\n" + "\n".join(hits[:30])


def test_no_secret_shaped_strings() -> None:
    hits = _hits(SECRET)
    assert not hits, "secret-shaped string in the public tree:\n" + "\n".join(hits[:30])
