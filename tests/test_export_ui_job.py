"""Export page: live job panel, re-attach after reload, locked controls (markup contract)."""
from __future__ import annotations

import re

from finetune_studio import db


def _page(client) -> str:
    pid = client.post("/api/projects", json={"name": "UI Job"}).json()["id"]
    run = db.create_run(project_id=pid, name="r", base_model="Qwen/Qwen3-0.6B", data_path="/d")
    db.update_run(run["id"], status="done", output_path="/nonexistent")
    r = client.get(f"/projects/{pid}/export")
    assert r.status_code == 200
    return r.text


def test_page_has_the_job_panel_and_cancel_button(client) -> None:
    html = _page(client)
    for marker in ('id="export-job"', 'id="export-job-state"', 'id="export-job-bar"',
                   'id="export-job-detail"', 'id="export-job-error"', 'id="export-cancel-btn"'):
        assert marker in html, marker


def test_script_follows_the_job_via_sse_and_reattaches(client) -> None:
    html = _page(client)
    assert "/exports/active" in html          # re-attach after a reload
    assert "'/events'" in html                # SSE, not a held-open POST
    assert "/cancel" in html
    assert "localStorage" in html             # remaining formats survive a reload
    assert "setLocked(true" in html           # conflicting controls disabled while a job runs
    assert "server_now" in html               # elapsed time uses the server's clock
    for phase in ("merging", "converting", "quantizing"):
        assert phase in html


def test_script_does_not_hold_the_post_open_for_the_whole_export(client) -> None:
    html = _page(client)
    script = html[html.index("// ── Export job UI"):]
    # The only fetch of the export URL is the queueing POST; completion is observed through
    # the job stream, never by awaiting that response.
    posts = re.findall(r"fetch\(API \+ '/runs/'[^)]*\)", script)
    assert len(posts) == 1
    assert "follow(d.export_id)" in script


def test_inline_script_is_wrapped_so_spa_reentry_cannot_redeclare_consts(client) -> None:
    html = _page(client)
    script = html[html.index("// ── Export job UI"):]
    assert script.lstrip().split("\n", 1)[1].lstrip().startswith("// An export is a server-side job")
    assert "(function () {" in script.split("// An export is a server-side job", 1)[1][:900]
    assert "window.startExport" in script
    assert not re.search(r"^(const|let) ", script.split("Pre-select a run when navigated")[0], re.M)
