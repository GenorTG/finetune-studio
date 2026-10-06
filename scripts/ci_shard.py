#!/usr/bin/env python3
"""Print the pytest files of one CI shard: sorted test files, ``position % count == index``.

Usage:
    python scripts/ci_shard.py INDEX COUNT        # one path per line
    pytest -q $(python scripts/ci_shard.py 0 5)

Excluded on purpose (see EXCLUDED): ``tests/test_vram.py`` needs a physical GPU and
a downloaded model. Stdlib only, so it runs before the venv exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = frozenset({"tests/test_vram.py"})


def all_test_files(root: Path = ROOT) -> list[str]:
    """Every collected test module, repo-relative and sorted (the stable partition input)."""
    found = (p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py"))
    return sorted(p for p in found if p not in EXCLUDED)


def shard(index: int, count: int, files: list[str] | None = None) -> list[str]:
    """The files of shard ``index`` (0-based) out of ``count``; round-robin keeps shards balanced."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard index {index} must be in [0, {count})")
    return [f for i, f in enumerate(all_test_files() if files is None else files) if i % count == index]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        files = shard(int(argv[0]), int(argv[1]))
    except ValueError as exc:
        print(f"ci_shard: {exc}", file=sys.stderr)
        return 2
    print("\n".join(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
