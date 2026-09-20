"""Tests for the hostable RAG package: builder + standalone server.

Covers: build_package archive contents, parquet→jsonl conversion, the
standalone server's keyword (BM25) search via --query, and the MCP stdio
handshake (initialize / tools/list / tools/call rag_search).

The standalone server is executed as a real subprocess with the test
interpreter — numpy is the only dependency it needs, and no embedding
endpoint is configured, so everything runs offline and deterministically.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.rag_portable import (
    ChunkSettings,
    EmbeddingModelInfo,
    Manifest,
    RagSettings,
    write_json,
)
from finetune_studio.data.rag_portable.bm25 import BM25Index
from finetune_studio.data.rag_portable.mcp_package import build_package

_TEXTS = [
    "The Ledger-Keepers of Vaelindrath swear an oath of salt before the seal.",
    "Warehousing at the Highmere Stair costs nine crowns per season.",
    "A meteoric salt shower over Emberfall is remembered as the Emberfall.",
]
_IDS = ["c1", "c2", "c3"]


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    """A real (tiny) PortableRAG corpus dir with all on-disk artifacts."""
    pd = pytest.importorskip("pandas")
    corpus = tmp_path / "corpus"
    (corpus / "sources").mkdir(parents=True)
    for i, (cid, text) in enumerate(zip(_IDS, _TEXTS)):
        (corpus / "sources" / f"{cid}.txt").write_text(text, encoding="utf-8")

    df = pd.DataFrame(
        {
            "id": _IDS,
            "text": _TEXTS,
            "document_id": _IDS,
            "chunk_index": [0, 0, 0],
            "filename": ["ledger.txt", "highmere.txt", "emberfall.txt"],
            "source": ["ledger.txt", "highmere.txt", "emberfall.txt"],
        }
    )
    df.to_parquet(corpus / "chunks.parquet", index=False)

    vectors = np.eye(3, dtype=np.float32)
    np.save(corpus / "vectors.npy", vectors)
    write_json(corpus / "vectors.idx.json", {cid: i for i, cid in enumerate(_IDS)})

    bm25 = BM25Index.build(_TEXTS)
    write_json(corpus / "bm25.json", bm25.to_dict())

    manifest = Manifest(
        name="Test Corpus",
        version="2",
        created_at=1.0,
        updated_at=2.0,
        embedding_model=EmbeddingModelInfo(name="test-embed", dim=3),
        chunk_settings=ChunkSettings(size=400, overlap=80),
        rag_settings=RagSettings(embedder="test-embed", reranker="",
                                 rerank_enabled=False, hybrid_enabled=True),
        documents=3,
        chunks=3,
    )
    write_json(corpus / "manifest.json", manifest.to_json())
    return corpus


def _extract(archive: Path, dest: Path) -> Path:
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    roots = [p for p in dest.iterdir() if p.is_dir()]
    assert len(roots) == 1, f"expected one package root, got {roots}"
    return roots[0]


def test_build_package_contents(corpus_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "pkg" / "test-rag-package.tar.gz"
    archive = build_package(corpus_dir, out, name="Test Corpus")
    assert archive == out and out.is_file()

    root = _extract(out, tmp_path / "x")
    assert root.name == "test-corpus-rag"
    for f in ("server.py", "install.sh", "run-http.sh", "run-mcp.sh",
              "requirements.txt", "README.md", "mcp-config.example.json"):
        assert (root / f).is_file(), f"missing {f}"
    assert (root / "install.sh").stat().st_mode & 0o111, "install.sh must be executable"
    assert (root / "requirements.txt").read_text().strip() == "numpy"

    # parquet must be gone from the shipped corpus; jsonl must be readable
    assert not (root / "corpus" / "chunks.parquet").exists()
    lines = (root / "corpus" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(x) for x in lines]
    assert [r["id"] for r in rows] == _IDS
    assert "oath of salt" in rows[0]["text"]

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "run-mcp.sh" in readme and "RAG_EMBED_BASE_URL" in readme
    cfg = json.loads((root / "mcp-config.example.json").read_text())
    assert "test-corpus-rag" in cfg["mcpServers"]


def test_standalone_keyword_search(corpus_dir: Path, tmp_path: Path) -> None:
    """server.py --query works offline (no embedding endpoint) via BM25."""
    archive = build_package(corpus_dir, tmp_path / "p.tar.gz", name="Test Corpus")
    root = _extract(archive, tmp_path / "x")
    r = subprocess.run(
        [sys.executable, str(root / "server.py"),
         "--corpus", str(root / "corpus"),
         "--query", "oath of salt ledger", "--top-k", "2"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert hits and hits[0]["chunk_id"] == "c1"
    assert "oath of salt" in hits[0]["text"]
    assert hits[0]["filename"] == "ledger.txt"
    assert hits[0]["bm25_score"] > 0


def test_standalone_mcp_stdio(corpus_dir: Path, tmp_path: Path) -> None:
    """MCP handshake + tools/list + tools/call over stdio, JSON-RPC 2.0."""
    archive = build_package(corpus_dir, tmp_path / "p.tar.gz", name="Test Corpus")
    root = _extract(archive, tmp_path / "x")
    reqs = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "rag_search", "arguments": {"query": "highmere crowns",
                                                "top_k": 3}}},
    ]
    stdin = "\n".join(json.dumps(m) for m in reqs) + "\n"
    r = subprocess.run(
        [sys.executable, str(root / "server.py"), "--mcp"],
        input=stdin, capture_output=True, text=True, timeout=60, check=False,
        env={**dict(__import__("os").environ), "PYTHONPATH": ""},
    )
    assert r.returncode == 0, r.stderr
    responses = {}
    for line in r.stdout.splitlines():
        if line.strip():
            msg = json.loads(line)
            responses[msg["id"]] = msg

    init = responses[1]["result"]
    assert init["serverInfo"]["name"] == "fts-rag-mcp"
    assert "tools" in init["capabilities"]

    tools = {t["name"] for t in responses[2]["result"]["tools"]}
    assert {"rag_search", "rag_info"} <= tools

    call = responses[3]["result"]
    assert call["isError"] is False
    hits = json.loads(call["content"][0]["text"])
    assert hits and hits[0]["chunk_id"] == "c2"
    assert "nine crowns" in hits[0]["text"]


def test_missing_corpus_files_raises(tmp_path: Path) -> None:
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        build_package(empty, tmp_path / "p.tar.gz")
