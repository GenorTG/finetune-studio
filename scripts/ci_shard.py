#!/usr/bin/env python3
"""Print the pytest files of one CI shard: test files packed into ``count`` shards by expected run time.

Usage:
    python scripts/ci_shard.py INDEX COUNT        # one path per line
    pytest -q $(python scripts/ci_shard.py 0 5)

Packing is longest-processing-time-first over WEIGHTS (seconds measured on a hosted runner; any file not
listed costs DEFAULT_WEIGHT, roughly its collection/import cost): heaviest file first, each onto the
currently lightest shard, ties to the lowest index. The result depends only on the file list, so every
shard computes the same partition. Round-robin by position left one shard at 120 s next to four at
40-60 s because a single file (test_rag_audit) is ~70 s. Re-measure with `pytest --durations=0` (sum
setup+call+teardown per file) when a shard's wall time in the CI job summary drifts, and edit WEIGHTS.

Excluded on purpose (see EXCLUDED), each with a reason a CPU-only runner cannot meet.
Stdlib only, so it runs before the venv exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = frozenset({
    "tests/test_vram.py",              # profiles a real model on a physical GPU
    # Standalone playwright scripts against http://fan-dragon:7860 (zero pytest tests, `__main__` only);
    # collecting them just fails on `import playwright`, which is not a project dependency.
    "tests/test_breakpoints.py",
    "tests/test_breakpoints_visual.py",
    "tests/test_phase_bd.py",
    "tests/test_phase_bd_api.py",
})

# Measured seconds per file on a GitHub-hosted ubuntu runner (CI run 37432258997, `--durations=0`); only files that matter.
WEIGHTS: dict[str, float] = {
    "tests/test_rag_audit.py": 68.0,
    "tests/test_rag_encrypted_package.py": 14.0,
    "tests/test_rag_mcp_package.py": 11.0,
    "tests/test_accel_device_index.py": 7.0,
    "tests/test_training_checkpoint_eval.py": 6.0,
    "tests/test_config_honesty.py": 5.0,
    "tests/test_rag_model_cache_wiring.py": 3.5,
    "tests/test_fresh_run_regressions.py": 2.5,
    "tests/test_export_capabilities.py": 2.0,
}
DEFAULT_WEIGHT = 0.9


def weight(path: str) -> float:
    return WEIGHTS.get(path, DEFAULT_WEIGHT)


def all_test_files(root: Path = ROOT) -> list[str]:
    """Every collected test module, repo-relative and sorted (the stable partition input)."""
    found = (p.relative_to(root).as_posix() for p in (root / "tests").rglob("test_*.py"))
    return sorted(p for p in found if p not in EXCLUDED)


def pack(count: int, files: list[str] | None = None) -> list[list[str]]:
    """All files packed into ``count`` shards (each shard sorted), lightest-shard-first per heaviest file."""
    if count < 1:
        raise ValueError(f"shard count {count} must be >= 1")
    shards: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for f in sorted(all_test_files() if files is None else files, key=lambda p: (-weight(p), p)):
        i = loads.index(min(loads))
        shards[i].append(f)
        loads[i] += weight(f)
    return [sorted(s) for s in shards]


def shard(index: int, count: int, files: list[str] | None = None) -> list[str]:
    """The files of shard ``index`` (0-based) out of ``count``."""
    if count < 1 or not 0 <= index < count:
        raise ValueError(f"shard index {index} must be in [0, {count})")
    return pack(count, files)[index]


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
