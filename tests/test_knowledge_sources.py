"""``knowledge.sources``: each project file is parsed once; edits, Vision readings and Reparse go through it."""
from __future__ import annotations

import pytest

from finetune_studio import db
from finetune_studio.data.fs import file_library as fl
from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.knowledge import sources
from tests.knowledge_helpers import add_source

TEXT = "The gate of Saltmere closes at 22:00.\n\nThe keeper is Ada Morrow."


@pytest.fixture
def pid() -> str:
    return db.create_project(name="sources", description="")["id"]


def test_stage_then_parse_writes_text_passages_and_state(pid: str) -> None:
    staged = add_source(pid, "gate.txt", TEXT, parse=False)
    assert staged.status == sources.QUEUED and not staged.ready
    src = sources.parse(pid, staged.id)
    assert src.ready and src.id == src.sha256[:12] and src.parser and src.char_count == len(TEXT)
    assert sources.parsed_text(pid, src) == TEXT
    assert "22:00" in "".join(sources.passages(pid, src.sha256)) and src.chunk_count == len(sources.passages(pid, src.sha256))
    assert [s.id for s in sources.list_sources(pid)] == [src.id]
    assert sources.source_for_file(pid, src.file_id) == src and sources.names(pid) == {src.id: "gate.txt"}


def test_parse_is_idempotent_and_force_reparses(pid: str) -> None:
    src = add_source(pid, "gate.txt", TEXT)
    edited = sources.set_text(pid, src.id, TEXT + "\n\nA hand-written note.")
    assert sources.parse(pid, src.id) == edited                      # not forced: the edit stays
    back = sources.parse(pid, src.id, force=True)
    assert back.edited_at is None and "hand-written" not in sources.parsed_text(pid, back)


def test_a_file_without_text_is_an_error_source_with_the_reason(pid: str) -> None:
    src = add_source(pid, "blank.txt", "   \n  ")
    assert src.status == sources.ERROR and "no text" in src.error and not src.ready
    with pytest.raises(sources.SourceError) as e:
        sources.view(pid, src.file_id)
    assert e.value.status == 422


def test_view_states(pid: str) -> None:
    queued = add_source(pid, "later.txt", TEXT, parse=False)
    with pytest.raises(sources.SourceError) as e:
        sources.view(pid, queued.file_id)
    assert e.value.status == 409
    meta = fl.write_uploaded_file(pid, b"\x00\x01", "blob.bin")
    with pytest.raises(sources.SourceError) as e:
        sources.stage(pid, meta.file_id)
    assert e.value.status == 422 and not sources.is_parseable("blob.bin")
    with pytest.raises(sources.SourceError) as e:
        sources.view(pid, "nope")
    assert e.value.status == 404


def test_set_text_refuses_empty_text(pid: str) -> None:
    src = add_source(pid, "gate.txt", TEXT)
    with pytest.raises(sources.SourceError) as e:
        sources.set_text(pid, src.id, "  ")
    assert e.value.status == 400


def test_purge_forgets_pairs_mining_and_the_parse_but_trash_keeps_them(pid: str) -> None:
    src = add_source(pid, "gate.txt", TEXT)
    qa_fs.write_qa_pair(pid, {"id": "q1", "source_id": src.id, "chunk_idx": 1, "question": "When?", "answer": "22:00.",
                              "status": "approved"})
    db.mining.insert_items(pid, src.id, "job", [{"chunk_idx": 1, "origin": "agent", "kind": "fact", "text": "t",
                                                 "values": ["22:00"]}])
    fl.soft_delete_file(pid, src.file_id)
    assert qa_fs.list_qa_pairs(pid) and sources.get_source(pid, src.id, include_deleted=True) is not None
    assert sources.get_source(pid, src.id) is None                  # a trashed file is not a live source
    fl.restore_file(pid, src.file_id)
    assert sources.get_source(pid, src.id) == src
    fl.soft_delete_file(pid, src.file_id)
    fl.purge_file(pid, src.file_id)
    assert not qa_fs.list_qa_pairs(pid) and not db.mining.list_items(pid, src.id)
    assert not sources.source_dir(pid, src, create=False).exists()


def test_resume_unfinished_parses_what_a_restart_cut_off(pid: str) -> None:
    queued = add_source(pid, "gate.txt", TEXT, parse=False)
    assert sources.resume_unfinished() == 1
    assert sources.get_source(pid, queued.id).ready
    assert sources.resume_unfinished() == 0
