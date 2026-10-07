"""run_group_subprocess: tail capture, timeout, cancel, no orphan children."""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from finetune_studio.training.proc_runner import (
    SubprocessCancelled,
    SubprocessFailed,
    SubprocessTimeout,
    run_group_subprocess,
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); only a real process counts as orphaned.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def _spawn_grandchild_script(pidfile: Path) -> list[str]:
    """sh that leaves a ``sleep 300`` grandchild in the same group, then waits."""
    return ["sh", "-c", f"sleep 300 & echo $! > {pidfile}; wait"]


def test_success_returns_tails() -> None:
    res = run_group_subprocess(
        [sys.executable, "-c", "import sys; print('out'); sys.stderr.write('err')"],
        timeout=20,
    )
    assert res.returncode == 0
    assert "out" in res.stdout_tail and "err" in res.stderr_tail


def test_failure_reports_the_stderr_tail() -> None:
    code = "import sys; sys.stderr.write('boom: tensor shape mismatch'); sys.exit(7)"
    with pytest.raises(SubprocessFailed) as exc:
        run_group_subprocess([sys.executable, "-c", code], timeout=20)
    assert exc.value.returncode == 7
    assert "boom: tensor shape mismatch" in str(exc.value)


def test_output_tail_is_bounded_and_keeps_the_end() -> None:
    code = "import sys\nfor i in range(20000): sys.stderr.write('line %d\\n' % i)\nsys.exit(1)"
    with pytest.raises(SubprocessFailed) as exc:
        run_group_subprocess([sys.executable, "-c", code], timeout=30)
    assert "line 19999" in exc.value.tail
    assert "line 0\n" not in exc.value.tail
    assert len(exc.value.tail) <= 1000


def test_timeout_kills_the_whole_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild.pid"
    t0 = time.monotonic()
    with pytest.raises(SubprocessTimeout, match="timed out"):
        run_group_subprocess(_spawn_grandchild_script(pidfile), timeout=1)
    assert time.monotonic() - t0 < 15
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild), "timeout left an orphan grandchild running"


def test_cancel_kills_the_group_and_raises(tmp_path: Path) -> None:
    pidfile = tmp_path / "grandchild.pid"
    flag = threading.Event()
    threading.Timer(0.6, flag.set).start()
    with pytest.raises(SubprocessCancelled):
        run_group_subprocess(
            _spawn_grandchild_script(pidfile), timeout=60, cancelled=flag.is_set,
        )
    grandchild = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)


def test_on_tick_fires_while_the_child_runs() -> None:
    ticks: list[float] = []
    run_group_subprocess(
        [sys.executable, "-c", "import time; time.sleep(1.2)"],
        timeout=20, on_tick=lambda: ticks.append(time.monotonic()),
    )
    assert len(ticks) >= 3
