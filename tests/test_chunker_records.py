"""Structured source records remain intact when they fit a passage."""
from __future__ import annotations

from finetune_studio.data.prep.chunker import chunk_text


def _splits(records: list[str], chunks: list[str]) -> int:
    return sum(not any(record in chunk for chunk in chunks) for record in records)


def test_xml_records_stay_whole_and_merge_only_within_budget() -> None:
    records = [f'<item>\n  <name>N{i}</name>\n  <value>v{i}</value>\n</item>' for i in range(20)]
    chunks = chunk_text("\n".join(records), target_chars=240, overlap=0)
    assert _splits(records, chunks) == 0
    assert all(len(c) <= 240 for c in chunks)


def test_json_objects_stay_whole() -> None:
    records = [f'{{\n  "id": {i},\n  "name": "record {i}",\n  "value": "value {i}"\n}}' for i in range(20)]
    chunks = chunk_text(",\n".join(records), target_chars=240, overlap=0)
    assert _splits(records, chunks) == 0


def test_yaml_list_records_stay_whole() -> None:
    records = [f'- id: {i}\n  name: record {i}\n  value: value {i}' for i in range(20)]
    chunks = chunk_text("\n".join(records), target_chars=240, overlap=0)
    assert _splits(records, chunks) == 0


def test_oversized_record_falls_back_losslessly_and_markdown_bullets_keep_behavior() -> None:
    body = " ".join(f'field{i}: value{i}.' for i in range(120))
    record = f'<item>\n  {body}\n</item>'
    chunks = chunk_text(record, target_chars=300, overlap=0)
    assert "<item>" in "\n".join(chunks) and "</item>" in "\n".join(chunks)
    assert all(f'field{i}: value{i}.' in "\n".join(chunks) for i in range(120))
    bullets = "\n".join(f'- ordinary markdown bullet {i}' for i in range(20))
    assert "\n".join(chunk_text(bullets, target_chars=240, overlap=0)) == bullets


def test_nested_table_groups_per_row() -> None:
    rows = [f'<row>\n  <cell>r{i}a</cell>\n  <cell>r{i}b</cell>\n</row>' for i in range(12)]
    chunks = chunk_text("\n".join(rows), target_chars=240, overlap=0)
    assert _splits(rows, chunks) == 0



def test_chunk_manifest_records_chunker_version(tmp_path, monkeypatch) -> None:
    from finetune_studio.data.fs import paths
    from finetune_studio.data.fs.chunks import write_chunks
    from finetune_studio.data.prep.chunker import CHUNKER_VERSION

    monkeypatch.setattr(paths, "_ROOT", tmp_path)
    monkeypatch.setattr(paths, "_PROJECTS", tmp_path / "projects")
    paths._PROJECTS.mkdir()
    write_chunks("p1", "abc", ["one passage"])
    version_file = paths.file_dir("p1", "abc") / "chunks" / "chunker-version.json"
    assert version_file.read_text(encoding="utf-8").strip() == str(CHUNKER_VERSION)


def test_audit_marks_unversioned_existing_chunks_for_explicit_reparse(tmp_path, monkeypatch) -> None:
    import hashlib
    from finetune_studio.data import audit as data_audit
    from finetune_studio.knowledge.sources import Source

    raw = tmp_path / "source.txt"
    raw.write_text("Legacy source text.", encoding="utf-8")
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    file_root = tmp_path / "files"
    chunks_dir = file_root / "chunks"
    chunks_dir.mkdir(parents=True)
    parsed = "Legacy source text."
    (file_root / "parsed.txt").write_text(parsed, encoding="utf-8")
    (chunks_dir / "0000.txt").write_text(parsed, encoding="utf-8")
    monkeypatch.setattr(data_audit, "file_dir", lambda *_args, **_kwargs: file_root)
    source = Source(id=digest[:12], file_id="f1", version=1, sha256=digest, filename="source.txt",
                    mime_type="text/plain", raw_path=str(raw), size_bytes=raw.stat().st_size, uploaded_at=0.0,
                    status="ready", text_sha="", parser="", char_count=len(parsed), chunk_count=1, error="",
                    edited_at=None, deleted=False)

    result = data_audit.audit_source("p1", source)

    assert "chunker_version_mismatch_reparse_required" in result["errors"]
    assert result["chunker_version"] == 0
