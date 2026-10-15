"""Judge jobs (and test runs / comparisons, same machinery) run as their own job process (supervisor P3, 2026-10-15):
the job runs in another pid, writes the web app's DB (the spec carries its absolute path), reads the providers DB
the web app uses (FTS_DB), and a cancel from the web side reaches it as SIGTERM. Real processes here (the rest of the
suite runs jobs in-process, ``jobs.launch``); the judge is an OpenAI-compatible provider served by this test."""
from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.webui import testing_jobs
from tests.testing_run_support import judge_reply, make_project_run, saved_run


class _Judge(BaseHTTPRequestHandler):
    delay_s = 0.0
    calls = 0

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        type(self).calls += 1
        time.sleep(type(self).delay_s)
        body = json.dumps({"choices": [{"message": {"content": judge_reply("pass")}, "finish_reason": "stop"}]})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *_a: object) -> None:
        return


@pytest.fixture
def real_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Real job processes, cwd in a temp dir, and an API judge provider served from this process. Yields its id."""
    import finetune_studio.models.manager as mgr_mod
    from finetune_studio.models.manager import get_manager

    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    monkeypatch.setenv("FTS_DB", str(mgr_mod._DB_PATH))      # the job process cannot see the monkeypatched path
    monkeypatch.chdir(tmp_path)                                # a stray cwd-relative DB would land here
    _Judge.delay_s, _Judge.calls = 0.0, 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Judge)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    get_manager().upsert_provider(id="proc-judge", name="Process judge", kind="openai_compat", model_id="judge-x",
                                  base_url=f"http://127.0.0.1:{server.server_address[1]}/v1", api_key="k",
                                  extra={"retries": 0, "timeout": 30})
    yield "proc-judge"
    for spec in testing_jobs._live_specs():
        launch.stop(spec.id)
        launch.wait(spec.id, timeout=30)
    server.shutdown()


def _job_id(bid: str) -> str:
    return next(j["job_id"] for j in testing_jobs.active_jobs() if j["benchmark_id"] == bid)


def _wait_row(bid: str, until, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = db.get_benchmark(bid) or {}
        if until(row):
            return row
        time.sleep(0.1)
    raise AssertionError(f"timed out: {row}")


def test_a_judge_job_runs_in_its_own_process_against_the_web_apps_db(real_jobs: str, tmp_path: Path) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [("Q1", "k", "a"), ("Q2", "k", "b")])
    row = testing_jobs.launch_judge(bid, real_jobs)
    assert row["judge_status"] == "running"
    jid = _job_id(bid)
    spec = store.read_spec(jid)
    assert spec.kind == "judge" and spec.gpu is False and spec.db_path == os.path.abspath(db.connection.settings.db_path)
    d = launch.wait(jid, timeout=60)
    assert d["exit"]["status"] == "done", d
    assert d["pid"] and d["pid"] != os.getpid()
    done = db.get_benchmark(bid)
    assert done["judge_status"] == "done" and done["judge_model"] == "judge-x" and done["scores"]["passed"] == 2
    assert _Judge.calls == 2
    assert not (tmp_path / "data").exists(), "the job used a cwd-relative DB instead of the web app's"
    assert testing_jobs.running_benchmark_ids() == set()


def test_cancel_from_the_web_side_stops_the_judge_process_and_keeps_the_verdicts(real_jobs: str) -> None:
    _pid, rid = make_project_run()
    bid = saved_run(rid, [(f"Q{i}", "k", "a") for i in range(40)])
    _Judge.delay_s = 0.3
    testing_jobs.launch_judge(bid, real_jobs)
    jid = _job_id(bid)
    _wait_row(bid, lambda r: (r.get("judge_done") or 0) >= 1)
    assert testing_jobs.is_active(bid, "judge") and testing_jobs.running_benchmark_ids() == {bid}
    with pytest.raises(testing_jobs.TestingBusy):              # the same run is never judged twice at once
        testing_jobs.launch_judge(bid, real_jobs)
    # a web restart now would leave this row alone: its job is alive
    assert db.reconcile_stale_benchmarks(keep=testing_jobs.running_benchmark_ids()) == 0
    assert testing_jobs.cancel(bid) is True
    assert any(j["stopping"] for j in testing_jobs.active_jobs() if j["benchmark_id"] == bid)
    d = launch.wait(jid, timeout=60)
    assert d["exit"]["status"] == "stopped", d
    row = db.get_benchmark(bid)
    assert row["judge_status"] == "cancelled" and 1 <= row["scores"]["passed"] < 40
    assert testing_jobs.cancel(bid) is False and testing_jobs.active_jobs() == []
