"""Job files under ``$FTS_ROOT/run/jobs/<id>/``: the spec, the live state, the exit record, the pid."""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,79}$")
KEEP_FINISHED_S = 7 * 24 * 3600


def jobs_dir() -> Path:
    root = Path(os.environ.get("FTS_ROOT") or str(Path.home() / ".finetune-studio"))
    return root / "run" / "jobs"


def job_dir(job_id: str) -> Path:
    if not _ID.match(job_id):
        raise ValueError(f"invalid job id {job_id!r}")
    return jobs_dir() / job_id


def new_id(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


@dataclass
class JobSpec:
    id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    gpu: bool = False                 # takes the GPU lease (one GPU job at a time; inference model unloaded)
    stop_grace_s: float = 60.0        # SIGTERM -> SIGKILL (training needs time to save)
    title: str = ""
    project_id: str | None = None
    created_at: float = field(default_factory=time.time)
    # The main DB the launching process uses (absolute); the job opens the same one whatever its cwd.
    db_path: str = ""


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except ValueError:
        return None                       # a torn write never happens (os.replace), but never crash a reader


def write_spec(spec: JobSpec) -> Path:
    d = job_dir(spec.id)
    if (d / "spec.json").exists():
        raise FileExistsError(f"job {spec.id} already exists")
    _write_json(d / "spec.json", asdict(spec))
    return d


def read_spec(job_id: str) -> JobSpec:
    raw = _read_json(job_dir(job_id) / "spec.json")
    if raw is None:
        raise FileNotFoundError(f"no job {job_id!r}")
    return JobSpec(**raw)


def publish_state(job_id: str, state: dict[str, Any]) -> None:
    _write_json(job_dir(job_id) / "state.json", {**state, "updated_at": time.time()})


def read_state(job_id: str) -> dict[str, Any] | None:
    return _read_json(job_dir(job_id) / "state.json")


def write_exit(job_id: str, record: dict[str, Any], *, overwrite: bool = False) -> None:
    path = job_dir(job_id) / "exit.json"
    if path.exists() and not overwrite:
        return                            # the runner's own record wins over the supervisor's guess
    _write_json(path, {**record, "at": record.get("at", time.time())})


def read_exit(job_id: str) -> dict[str, Any] | None:
    return _read_json(job_dir(job_id) / "exit.json")


def write_pid(job_id: str, pid: int) -> None:
    (job_dir(job_id) / "pid").write_text(str(pid), encoding="utf-8")


def read_pid(job_id: str) -> int | None:
    try:
        return int((job_dir(job_id) / "pid").read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def pid_alive(job_id: str) -> bool:
    """The recorded pid runs THIS job (a recycled pid running something else does not count)."""
    pid = read_pid(job_id)
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return "finetune_studio.jobs.runner" in cmdline and job_id in cmdline


def release_gpu(job_id: str) -> None:
    """The job is past its GPU work (only its exit is left): the next GPU job may start. A finished export's
    process is still exiting when the page already posts the next selected format."""
    (job_dir(job_id) / "gpu_released").touch()


def gpu_released(job_id: str) -> bool:
    try:
        return (job_dir(job_id) / "gpu_released").exists()
    except ValueError:
        return False


def list_ids() -> list[str]:
    d = jobs_dir()
    if not d.is_dir():
        return []
    return sorted((p.name for p in d.iterdir() if (p / "spec.json").exists()),
                  key=lambda i: (d / i / "spec.json").stat().st_mtime)


def prune(now: float | None = None) -> int:
    """Forget finished jobs older than a week (their results live in the DB)."""
    now = now or time.time()
    removed = 0
    for job_id in list_ids():
        ex = read_exit(job_id)
        if ex and now - float(ex.get("at", now)) > KEEP_FINISHED_S:
            shutil.rmtree(job_dir(job_id), ignore_errors=True)
            removed += 1
    return removed
