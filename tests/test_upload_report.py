"""The upload result names what did not go in and holds no server paths (walkthrough 2026-11-13 m9, m11)."""
from __future__ import annotations

import json
from pathlib import Path


def test_upload_report_names_failures_and_duplicates_without_server_paths(client) -> None:
    pid = client.post("/api/projects", json={"name": "agent-test-upload-report"}).json()["id"]
    files = [("files", ("faq.md", b"# FAQ\n\nDelivery every Tuesday.\n" * 5, "text/markdown")),
             ("files", ("copy.md", b"# FAQ\n\nDelivery every Tuesday.\n" * 5, "text/markdown")),
             ("files", ("empty.txt", b"", "text/plain"))]
    r = client.post(f"/api/projects/{pid}/files/upload", files=files)
    assert r.status_code == 200, r.text
    rows = {x["filename"]: x for x in r.json()["report"]}
    assert rows["empty.txt"]["status"] == "error" and rows["empty.txt"]["error"] == "empty upload"
    assert rows["copy.md"]["status"] == "duplicate" and rows["copy.md"]["duplicate_of"] == "faq.md"
    assert "raw_path" not in json.dumps(r.json())
    html = (Path(__file__).resolve().parents[1] / "src/finetune_studio/webui/templates/project_data.html").read_text()
    assert "' Not added: ' + list(failed)" in html and "Same content as a file already here, skipped" in html
