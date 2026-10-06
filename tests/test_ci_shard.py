"""CI test sharding: every test file lands in exactly one shard, GPU-only files are excluded."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
COUNT = 5


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ci_shard", ROOT / "scripts" / "ci_shard.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shards_partition_every_test_file_exactly_once() -> None:
    mod = _load()
    everything = mod.all_test_files()
    shards = [mod.shard(i, COUNT) for i in range(COUNT)]
    flat = [f for s in shards for f in s]
    assert sorted(flat) == everything and len(flat) == len(set(flat))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_gpu_only_files_are_excluded_and_unit_dir_is_included() -> None:
    mod = _load()
    files = mod.all_test_files()
    assert "tests/test_vram.py" not in files
    assert "tests/unit/test_engine_fixes.py" in files
    assert files == sorted(files)


@pytest.mark.parametrize("index,count", [(-1, 5), (5, 5), (0, 0)])
def test_bad_shard_arguments_are_rejected(index: int, count: int) -> None:
    with pytest.raises(ValueError):
        _load().shard(index, count)
