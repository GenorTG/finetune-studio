"""``knowledge.migrate``: projects move off ``qa/sources/*.json`` once, with nothing lost and nothing left behind."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finetune_studio import db
from finetune_studio.data.fs.chunks import write_chunks
from finetune_studio.data.fs.paths import file_dir, project_dir
from finetune_studio.knowledge import migrate, sources
from tests.knowledge_helpers import add_source

TEXT = "The gate of Saltmere closes at 22:00. The keeper is Ada Morrow."


def _old_layout(pid: str, name: str, *, parsed: bool = True) -> sources.Source:
    """A library file the way the old code left it: parse on disk, a JSON manifest, no parse row."""
    text = f"{TEXT} ({name})"
    src = add_source(pid, name, text, parse=False)
    db.file_parses.delete(src.file_id)
    d = file_dir(pid, src.sha256)
    if parsed:
        (d / "parsed.txt").write_text(text, encoding="utf-8")
        write_chunks(pid, src.sha256, [text])
    (d / "metadata.json").write_text("{}", encoding="utf-8")
    (d / name).write_text(text, encoding="utf-8")                       # the old duplicate byte copy
    manifests = project_dir(pid) / "qa" / "sources"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"{src.id}.json").write_text(json.dumps({
        "id": src.id, "file_id": src.file_id, "filename": name, "status": "ready" if parsed else "queued",
        "parser": "text_v1", "edited_at": 5.0 if parsed else None}), encoding="utf-8")
    return src


def test_ready_and_unparsed_manifests_become_parse_rows_and_old_files_go() -> None:
    pid = db.create_project(name="mig", description="")["id"]
    ready = _old_layout(pid, "ready.txt")
    pending = _old_layout(pid, "pending.txt", parsed=False)
    assert migrate.needs_migration(pid)
    stats = migrate.migrate_project(pid)
    assert stats["ready"] == 1 and stats["queued"] == 1 and stats["orphans"] == 0
    got = sources.get_source(pid, ready.id)
    assert got.ready and got.parser == "text_v1" and got.edited_at == 5.0 and got.chunk_count == 1
    assert sources.get_source(pid, pending.id).status == sources.QUEUED
    assert not migrate.needs_migration(pid)
    left = {p.name for p in file_dir(pid, ready.sha256).iterdir()}
    assert left == {"parsed.txt", "chunks"}
    assert migrate.migrate_project(pid)["sources"] == 0                 # idempotent


def test_a_hand_edit_override_file_becomes_the_source_text() -> None:
    pid = db.create_project(name="mig-edit", description="")["id"]
    src = _old_layout(pid, "edited.txt")
    override = src.raw_path + ".parsed.md"
    with open(override, "w", encoding="utf-8") as fh:
        fh.write("Edited by hand: the gate closes at 22:30.")
    assert migrate.migrate_project(pid)["edits"] == 1
    got = sources.get_source(pid, src.id)
    assert "22:30" in sources.parsed_text(pid, got) and got.parser == sources.MANUAL_EDIT
    assert not Path(override).exists()


def test_a_source_that_bypassed_the_library_is_adopted() -> None:
    pid = db.create_project(name="mig-orphan", description="")["id"]
    raw = file_dir(pid, "ab" * 32) / "legacy.txt"
    raw.write_text(TEXT, encoding="utf-8")
    sha = hashlib.sha256(TEXT.encode()).hexdigest()
    manifests = project_dir(pid) / "qa" / "sources"
    manifests.mkdir(parents=True)
    (manifests / f"{sha[:12]}.json").write_text(json.dumps({
        "id": sha[:12], "filename": "legacy.txt", "status": "ready", "data_path": str(raw)}), encoding="utf-8")
    stats = migrate.migrate_project(pid)
    assert stats["orphans"] == 0
    got = sources.get_source(pid, sha[:12])
    assert got is not None and got.filename == "legacy.txt"          # same bytes = same source id, pairs still match


def test_migrate_all_moves_only_old_projects() -> None:
    old = db.create_project(name="old", description="")["id"]
    new = db.create_project(name="new", description="")["id"]
    _old_layout(old, "a.txt")
    add_source(new, "b.txt", TEXT)
    assert migrate.migrate_all() == 1 and migrate.migrate_all() == 0
