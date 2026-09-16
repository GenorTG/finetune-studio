"""MIME-varied RAG ingestion with a fake deterministic embedder (no HF download)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from finetune_studio.data.rag_eval import (
    UNKNOWN_REPLY,
    QAEntry,
    run_rag_evaluation,
)
from finetune_studio.data.rag_portable.schema import EmbeddingModelInfo
from finetune_studio.data.rag_portable.store import PortableRAG

MARKER_HELIOS = "HeliosCEO_MiraChen"
MARKER_MENU = "LunchSpecial_TomatoSoup"


def _hash_embed(texts: list[str] | str, dim: int = 64) -> np.ndarray:
    """Bag-of-token hashing → L2-normalised vectors (deterministic, offline)."""
    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    out = np.zeros((len(items), dim), dtype=np.float32)
    for i, text in enumerate(items):
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            out[i, h % dim] += 1.0
        norm = float(np.linalg.norm(out[i]))
        if norm > 0:
            out[i] /= norm
    return out[0] if single else out


def _fake_get_embedder(name: str = "fake-deterministic", device: str = "cpu"):
    dim = 64

    def encode(texts: list[str] | str) -> np.ndarray:
        return _hash_embed(texts, dim=dim)

    info = EmbeddingModelInfo(
        name=name or "fake-deterministic",
        dim=dim,
        normalize=True,
        distance="cosine",
        cached_at="1970-01-01T00:00:00Z",
    )
    return encode, info


def _write_mime_corpus(root: Path) -> None:
    """Several MIME families that parsers handle without optional binaries."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "helios.txt").write_text(
        f"Helios Industries handbook. The CEO is {MARKER_HELIOS}. "
        "Aurora packs reach 400 Wh/kg.",
        encoding="utf-8",
    )
    (root / "menu.md").write_text(
        f"# Catering\n\nToday's lunch special is {MARKER_MENU} with grilled cheese.\n",
        encoding="utf-8",
    )
    (root / "facts.json").write_text(
        f'{{"org": "Helios Industries", "ceo_token": "{MARKER_HELIOS}"}}\n',
        encoding="utf-8",
    )
    (root / "menu.csv").write_text(
        f"item,note\nlunch,{MARKER_MENU}\n",
        encoding="utf-8",
    )
    (root / "about.html").write_text(
        f"<html><body><h1>Helios</h1><p>CEO token {MARKER_HELIOS}</p></body></html>\n",
        encoding="utf-8",
    )
    (root / "note.yaml").write_text(
        f"org: Helios\nceo: {MARKER_HELIOS}\n",
        encoding="utf-8",
    )


@pytest.fixture()
def patched_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.store.get_embedder",
        _fake_get_embedder,
    )
    # Avoid downloading a cross-encoder during search.
    monkeypatch.setattr(
        "finetune_studio.data.rag_portable.query.get_reranker",
        lambda name="", device="cpu": (None, None),
    )


def test_varied_mime_ingestion_then_rag_eval(
    tmp_path: Path, patched_embedder: None
) -> None:
    src = tmp_path / "sources"
    corpus_dir = tmp_path / "corpus"
    _write_mime_corpus(src)

    rag = PortableRAG(corpus_dir)
    stats = rag.build_from_directory(
        src,
        name="mime-smoke",
        embedder="fake-deterministic",
        chunk_size=80,
        overlap=10,
        extensions=[".txt", ".md", ".json", ".csv", ".html", ".yaml"],
        device="cpu",
    )
    assert stats["documents"] >= 4
    assert stats["chunks"] >= 4

    # Disable rerank in persisted settings so search never touches CE models.
    from finetune_studio.data.rag_portable.io import read_json, write_json
    from finetune_studio.data.rag_portable.schema import Manifest

    manifest = Manifest.from_json(read_json(rag.manifest_path))
    manifest.rag_settings.rerank_enabled = False
    write_json(rag.manifest_path, manifest.to_json())

    engine = rag.load()
    qa = [
        QAEntry(
            id="ceo",
            query="Who is the CEO token for Helios Industries?",
            must_contain=[MARKER_HELIOS],
            expected_source_contains="helios.txt",
        ),
        QAEntry(
            id="lunch",
            query="What is the lunch special token?",
            must_contain=[MARKER_MENU],
            expected_source_contains="menu",
        ),
    ]

    def answer_fn(query: str, context: str) -> str:
        if not context.strip():
            return UNKNOWN_REPLY
        low = context.lower()
        if MARKER_HELIOS.lower() in low and "ceo" in query.lower():
            return f"The CEO token is {MARKER_HELIOS}."
        if MARKER_MENU.lower() in low and "lunch" in query.lower():
            return f"The lunch special is {MARKER_MENU}."
        return UNKNOWN_REPLY

    report = run_rag_evaluation(
        rag_query=engine,
        qa=qa,
        corpus_name="mime-smoke",
        embedding_model="fake-deterministic",
        answer_fn=answer_fn,
        run_no_context=True,
        run_portability=False,
        ks=(1, 3, 5),
    )

    assert report.metadata.ran_retrieval is True
    assert report.metadata.ran_grounding is True
    assert report.metadata.ran_no_context is True
    assert report.recall_at_k[1] == 1.0
    assert report.mrr == 1.0
    assert report.grounding_pass_rate == 1.0
    assert report.fact_coverage_pass_rate == 1.0
    assert report.no_context_pass_rate == 1.0
