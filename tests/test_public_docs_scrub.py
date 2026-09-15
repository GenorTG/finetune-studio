"""Public-facing docs must not leak private host / path / process details.

Scans README.md, docs/index.html (GitHub Pages), and the docs linked from
the README so a future marketing refresh cannot reintroduce hostnames,
home-directory paths, or other owner-setup fingerprints.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]

# Files that are deliberately GitHub-facing (README links + GH Pages).
_PUBLIC_DOCS: tuple[Path, ...] = (
    _ROOT / "README.md",
    _ROOT / "docs" / "index.html",
    _ROOT / "docs" / "INSTALL.md",
    _ROOT / "docs" / "ARCHITECTURE.md",
    _ROOT / "docs" / "DEPENDENCIES.md",
    _ROOT / "docs" / "DEPLOYMENT.md",
    _ROOT / "docs" / "REFACTOR-SPEC.md",
    _ROOT / "docs" / "ATTRIBUTIONS.md",
    _ROOT / "docs" / "LEGAL.md",
)

# Private infra / owner fingerprints that must not appear in public docs.
_FORBIDDEN: tuple[re.Pattern[str], ...] = (
    re.compile(r"fan[-_]?dragon", re.IGNORECASE),
    re.compile(r"genorbox", re.IGNORECASE),
    re.compile(r"stealth[-_]?dragon", re.IGNORECASE),
    re.compile(r"comfyui", re.IGNORECASE),
    re.compile(r"/home/genortg\b"),
    re.compile(r"/home/genorbox"),
    re.compile(r"\bpid\s+\d{4,}\b", re.IGNORECASE),
    re.compile(r"b08426e3"),
    re.compile(r"http://fan-dragon", re.IGNORECASE),
    re.compile(r"/tmp/uvicorn\.log"),
    re.compile(r"70/70"),
)


@pytest.mark.parametrize("path", _PUBLIC_DOCS, ids=lambda p: p.relative_to(_ROOT).as_posix())
def test_public_doc_exists(path: Path) -> None:
    assert path.is_file(), f"missing public doc: {path}"


@pytest.mark.parametrize("path", _PUBLIC_DOCS, ids=lambda p: p.relative_to(_ROOT).as_posix())
def test_public_doc_has_no_private_infra_leaks(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    hits: list[str] = []
    for pat in _FORBIDDEN:
        m = pat.search(text)
        if m is not None:
            hits.append(f"{pat.pattern!r} → {m.group(0)!r}")
    assert not hits, f"{path.relative_to(_ROOT)} leaked private details: {hits}"


def test_readme_documents_honest_export_caveats() -> None:
    """Public README should not over-advertise GGUF/GPTQ as always-on."""
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    assert "llama.cpp" in text
    assert "auto-gptq" in text or "auto_gptq" in text
    assert "fails" in text.lower() or "when" in text.lower()
    # Full HF benchmark names as if shipped wholesale are stale.
    assert "Winogrande" not in text
    assert "TruthfulQA" not in text
    assert re.search(r"\bARC\b", text) is None


def test_pages_index_avoids_stale_benchmark_and_e2e_badges() -> None:
    text = (_ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    assert "E2E_QA-70%2F70" not in text
    assert "70/70" not in text
    assert "Winogrande" not in text
    assert "TruthfulQA" not in text
