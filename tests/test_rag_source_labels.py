"""RAG source / citation display names for opaque parsed.txt / chunk files."""
from __future__ import annotations

import json
from pathlib import Path

from finetune_studio.data.rag_portable.source_labels import (
    display_name_for_path,
    is_opaque_source_name,
    prettify_source_label,
    should_ingest_source_file,
)
from finetune_studio.webui.routes.project_rag import list_indexed_docs


def test_opaque_name_detection() -> None:
    assert is_opaque_source_name("parsed.txt")
    assert is_opaque_source_name("parsed.json")
    assert is_opaque_source_name("0000.txt")
    assert is_opaque_source_name("chunks/0012.txt")
    assert not is_opaque_source_name("policy.pdf")
    assert not is_opaque_source_name("notes.txt")


def test_display_name_from_metadata(tmp_path: Path) -> None:
    fd = tmp_path / "abc123def456"
    fd.mkdir()
    (fd / "metadata.json").write_text(
        json.dumps({"original_filename": "Acme Policy 2024.pdf"}),
        encoding="utf-8",
    )
    (fd / "parsed.txt").write_text("body", encoding="utf-8")
    chunks = fd / "chunks"
    chunks.mkdir()
    (chunks / "0000.txt").write_text("chunk0", encoding="utf-8")

    assert display_name_for_path(fd / "parsed.txt") == "Acme Policy 2024.pdf (parsed)"
    assert display_name_for_path(chunks / "0000.txt") == "Acme Policy 2024.pdf (chunk 0)"


def test_should_ingest_skips_chunks_prefers_parsed(tmp_path: Path) -> None:
    fd = tmp_path / "deadbeefcafe"
    fd.mkdir()
    (fd / "metadata.json").write_text(
        json.dumps({"original_filename": "report.docx"}),
        encoding="utf-8",
    )
    parsed = fd / "parsed.txt"
    parsed.write_text("parsed body", encoding="utf-8")
    original = fd / "report.docx"
    original.write_bytes(b"raw")
    chunk = fd / "chunks" / "0001.txt"
    chunk.parent.mkdir()
    chunk.write_text("c", encoding="utf-8")
    loose = tmp_path / "loose-notes.txt"
    loose.write_text("loose", encoding="utf-8")

    assert should_ingest_source_file(parsed) is True
    assert should_ingest_source_file(original) is False
    assert should_ingest_source_file(chunk) is False
    assert should_ingest_source_file(loose) is True


def test_should_ingest_skips_raw_when_parsed_exists(tmp_path: Path) -> None:
    """File-library raw twin must not double-index beside files/<sha12>/parsed.txt."""
    files = tmp_path / "files"
    sha12 = "b04b26d1abef"
    file_id = sha12 + "6be9"
    parsed_dir = files / sha12
    parsed_dir.mkdir(parents=True)
    (parsed_dir / "metadata.json").write_text(
        json.dumps({"original_filename": "helios_notes.txt"}),
        encoding="utf-8",
    )
    parsed = parsed_dir / "parsed.txt"
    parsed.write_text("canonical notes body", encoding="utf-8")

    raw = files / "raw" / "other" / f"{file_id}_helios_notes.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("canonical notes body", encoding="utf-8")

    # Raw without a parsed twin still ingests (standalone / unparsed upload).
    orphan_raw = files / "raw" / "other" / "aaaaaaaaaaaa1111_orphan_notes.txt"
    orphan_raw.write_text("orphan only", encoding="utf-8")

    # Loose project text outside files/raw stays eligible.
    standalone = files / "standalone_memo.txt"
    standalone.write_text("memo", encoding="utf-8")

    assert should_ingest_source_file(parsed) is True
    assert should_ingest_source_file(raw) is False
    assert should_ingest_source_file(orphan_raw) is True
    assert should_ingest_source_file(standalone) is True


def test_prettify_falls_back_without_path() -> None:
    assert "parsed" in prettify_source_label("parsed.txt").lower()
    assert "chunk" in prettify_source_label("0000.txt").lower()
    assert prettify_source_label("manual_notes.txt") == "manual_notes.txt"


def test_list_indexed_docs_prettifies_opaque_parquet(
    tmp_path: Path, monkeypatch
) -> None:
    import pandas as pd

    pid = "prettypid01"
    corpus = tmp_path / "corpora" / pid
    corpus.mkdir(parents=True)
    (corpus / "sources").mkdir()
    src_dir = tmp_path / "projects" / pid / "files" / "aabbccddeeff"
    src_dir.mkdir(parents=True)
    (src_dir / "metadata.json").write_text(
        json.dumps({"original_filename": "Handbook.md"}),
        encoding="utf-8",
    )
    parsed = src_dir / "parsed.txt"
    parsed.write_text("handbook text " * 10, encoding="utf-8")
    (corpus / "sources" / "docabc.txt").write_text("handbook text " * 10, encoding="utf-8")

    df = pd.DataFrame([
        {
            "id": "docabc_0",
            "document_id": "docabc",
            "chunk_index": 0,
            "source": str(parsed),
            "filename": "parsed.txt",
            "text": "handbook text",
        }
    ])
    df.to_parquet(corpus / "chunks.parquet", index=False)
    (corpus / "manifest.json").write_text(
        json.dumps({"updated_at": 1_700_000_000.0}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "finetune_studio.webui.routes.project_rag.corpus_dir",
        lambda p: corpus if p == pid else tmp_path / "other" / p,
    )
    docs = list_indexed_docs(pid)
    assert len(docs) == 1
    assert docs[0]["name"] == "Handbook.md (parsed)"
    assert "parsed.txt" not in docs[0]["name"] or "Handbook" in docs[0]["name"]
