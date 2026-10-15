"""HF downloads run as their own job process (2026-10-15): the download runs in another pid, writes the web app's DB
(the spec carries its absolute path, whatever the job's cwd), reports live byte counts, and a cancel from the web side
reaches it as SIGTERM. Real processes against a fake Hub on localhost (``HF_ENDPOINT``): nothing is downloaded from
the internet. The rest of the suite runs jobs in-process (``jobs.launch``)."""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from finetune_studio import db
from finetune_studio.jobs import launch, store
from finetune_studio.webui import download_jobs

CHUNK = 64 * 1024


class _FakeHub(BaseHTTPRequestHandler):
    """``/<org>/<repo>/resolve/<rev>/<file>``: the headers ``hf_hub_download`` needs, then ``size`` bytes, one
    chunk every ``delay_s``."""

    size = 4 * CHUNK
    delay_s = 0.0

    def _head(self) -> None:
        self.send_response(200)
        self.send_header("X-Repo-Commit", "a" * 40)
        self.send_header("ETag", '"fake-etag-1"')
        self.send_header("Content-Length", str(self.size))
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802
        self._head()

    def do_GET(self) -> None:  # noqa: N802
        self._head()
        try:
            sent = 0
            while sent < self.size:
                n = min(CHUNK, self.size - sent)
                self.wfile.write(b"x" * n)
                self.wfile.flush()
                sent += n
                time.sleep(self.delay_s)
        except (BrokenPipeError, ConnectionResetError):
            pass                                  # the job was stopped mid-download

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[type[_FakeHub]]:
    handler = type("Hub", (_FakeHub,), {})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.delenv("FTS_JOBS_INPROCESS", raising=False)
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{server.server_port}")   # read by the job process
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "1")
    monkeypatch.chdir(tmp_path)                   # a stray cwd-relative DB would land here
    yield handler
    for j in launch.list_jobs(download_jobs.KIND, running_only=True):
        launch.stop(j["id"])
    server.shutdown()


def _wait_row(did: str, until, timeout: float = 60) -> dict:
    deadline = time.monotonic() + timeout
    row: dict = {}
    while time.monotonic() < deadline:
        row = db.get_hf_download(did) or {}
        if until(row):
            return row
        time.sleep(0.1)
    raise AssertionError(f"timed out: {row}; job: {launch.describe(download_jobs.job_id(did))}")


def test_a_download_runs_in_its_own_process_against_the_web_apps_db(hub, tmp_path) -> None:
    dest = tmp_path / "hf_models" / "Org__Tiny"
    row = download_jobs.start(repo_id="Org/Tiny", filename="model.bin", revision="main", dest=str(dest))
    jid = download_jobs.job_id(row["id"])
    assert store.read_spec(jid).db_path == os.path.abspath(db.connection.settings.db_path)
    assert store.read_spec(jid).gpu is False
    final = _wait_row(row["id"], lambda r: r.get("status") in download_jobs.TERMINAL_STATUSES)
    assert final["status"] == "completed", final
    assert final["bytes_done"] == final["bytes_total"] == hub.size
    assert Path(final["path"]) == dest / "model.bin" and (dest / "model.bin").stat().st_size == hub.size
    d = launch.wait(jid, timeout=30)
    assert d["exit"]["status"] == "done" and d["pid"] != os.getpid()
    assert not (tmp_path / "data").exists(), "the job used a cwd-relative DB instead of the web app's"


def test_progress_while_running_and_cancel_from_the_web_side_stops_the_job(hub, tmp_path) -> None:
    hub.size, hub.delay_s = 200 * CHUNK, 0.1      # ~20 s at full length: only a cancel ends it in time
    row = download_jobs.start(repo_id="Org/Big", filename="model.bin", revision="main",
                              dest=str(tmp_path / "hf_models" / "Org__Big"))
    jid = download_jobs.job_id(row["id"])
    live = _wait_row(row["id"], lambda r: r.get("status") == "downloading" and (r.get("bytes_done") or 0) > 0)
    assert live["bytes_total"] == hub.size and live["bytes_done"] < hub.size
    assert download_jobs.running_download_ids() == {row["id"]}
    # A web restart now would leave it alone: the startup reconcile keeps rows whose job is alive.
    assert download_jobs.reconcile_on_startup() == 0
    assert db.get_hf_download(row["id"])["status"] == "downloading"
    assert download_jobs.cancel(row["id"]) == "cancelled"
    assert launch.wait(jid, timeout=30)["exit"]["status"] == "stopped"
    final = db.get_hf_download(row["id"])
    assert final["status"] == "cancelled" and final["bytes_done"] < hub.size, final
    assert download_jobs.running_download_ids() == set()
    assert download_jobs.cancel(row["id"]) == "cancelled"        # a finished download is left as it is
