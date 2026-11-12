"""Is this project's knowledge ready to ship? One list, from the live state of each step of the main path.

``state``: done | todo | review (usable, someone should look) | problem (fix before shipping). ``ready`` needs readable
files, no lint errors or unreviewed superseded-document findings, and a fresh index; identity, a test and a built package
are shown but never block.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from finetune_studio import db
from finetune_studio.data.fs.paths import brain_index_dir, project_dir
from finetune_studio.knowledge import index as knowledge_index
from finetune_studio.knowledge.lint import lint
from finetune_studio.knowledge.sources import list_sources
from finetune_studio.smart_brain import identity as brain_identity


def _last_package_path(pid: str) -> Path:
    return project_dir(pid) / "brain_last_package.json"


def record_package(pid: str, *, filename: str, size: int, encrypted: bool, identity: bool, verified: dict[str, Any]) -> None:
    path = _last_package_path(pid)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"filename": filename, "size": size, "encrypted": encrypted, "identity": identity,
                               "verified": verified, "built_at": time.time()}), encoding="utf-8")
    os.replace(tmp, path)


def _last_package(pid: str) -> dict[str, Any] | None:
    try:
        return json.loads(_last_package_path(pid).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _index_time(pid: str) -> float:
    try:
        return float(json.loads((brain_index_dir(pid) / "manifest.json").read_text(encoding="utf-8")).get("updated_at") or 0)
    except (OSError, ValueError, TypeError):
        return 0.0


def _step(key: str, label: str, state: str, detail: str, href: str) -> dict[str, str]:
    return {"key": key, "label": label, "state": state, "detail": detail, "href": href}


def _test_step(pid: str, index_time: float) -> dict[str, str]:
    runs = [b for b in db.list_benchmarks_for_project(pid) if b.get("kind") == "brain" and b.get("status") == "done"]
    if not runs:
        return _step("test", "Test the answers", "todo", "Not tested yet: write a quiz from your documents.", "/testing")
    scores = runs[0].get("scores") or {}
    total, passed = scores.get("total"), scores.get("passed")
    result = f"{passed} of {total} passed" if total and scores.get("judged") else "answers saved, not judged yet"
    if float(runs[0].get("ran_at") or 0) < index_time:
        return _step("test", "Test the answers", "review", f"Last test ({result}) is older than the index.", "/testing")
    return _step("test", "Test the answers", "done", result, "/testing")


def _package_step(pid: str, index_time: float) -> dict[str, str]:
    last = _last_package(pid)
    if not last:
        return _step("package", "Build the package", "todo", "Not built yet.", "/export#package-form")
    check = last.get("verified") or {}
    if check.get("ok") is False:
        return _step("package", "Build the package", "problem", f"{last['filename']}: {check.get('detail')}", "/export#package-form")
    if float(last.get("built_at") or 0) < index_time:
        return _step("package", "Build the package", "review", f"{last['filename']} is older than the index.", "/export#package-form")
    return _step("package", "Build the package", "done",
                 last["filename"] + (" · opened and searched" if check.get("ok") else ""), "/export#package-form")


def readiness(pid: str) -> dict[str, Any]:
    """Blocking (runs the knowledge check): async callers use a thread."""
    sources = list_sources(pid)
    n_ready = sum(s.ready for s in sources)
    report = lint(pid)
    superseded = sum(1 for f in report["findings"]
                     if f["rule"] == "old_version" and not f["ignored"])
    st = knowledge_index.state(pid)
    index_time = _index_time(pid)
    saved = brain_identity.load(pid)
    if superseded:
        check = _step("check", "Check the knowledge", "problem",
                      f"{superseded} superseded document(s) must be removed or explicitly accepted before packaging.",
                      "/brain#brain-lint")
    elif report["errors"]:
        check = _step("check", "Check the knowledge", "problem", f"{report['errors']} to fix, {report['warnings']} to review.", "/brain#brain-lint")
    elif report["warnings"]:
        check = _step("check", "Check the knowledge", "review", f"{report['warnings']} to review.", "/brain#brain-lint")
    else:
        check = _step("check", "Check the knowledge", "done" if n_ready else "todo", "Nothing to fix." if n_ready else "Add files first.", "/brain#brain-lint")
    steps = [
        _step("files", "Add your files", "done" if n_ready else "todo", f"{n_ready} of {len(sources)} read." if sources else "No files yet.", "/data"),
        check,
        _step("identity", "Say who it speaks for", "done" if brain_identity.is_set(saved) else "todo",
              saved["name"] or ("Set." if brain_identity.is_set(saved) else "Optional: company, personality, how to answer."), "/brain#brain-identity"),
        _step("index", "Build the Smart Brain", "done" if st.fresh else ("review" if st.built else "todo"),
              "Up to date." if st.fresh else ("Your files changed since the build." if st.built else "Not built yet."), "/brain"),
        _test_step(pid, index_time),
        _package_step(pid, index_time),
    ]
    return {"ready": bool(n_ready) and not report["errors"] and not superseded and st.fresh, "steps": steps}
