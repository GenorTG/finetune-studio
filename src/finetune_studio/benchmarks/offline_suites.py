"""Write / load substantive synthetic offline benchmark fixtures.

Fixtures are labeled as synthetic/offline — not licensed HuggingFace datasets
and not industry MMLU/GSM8K/HellaSwag scores. Smoke fixtures (6 cases) remain
for quick checks; offline fixtures are larger and scored with strict
task-aware matching (exact MC option / normalized numeric final).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from finetune_studio.benchmarks.offline_fixture_data import (
    gsm8k_offline_cases,
    hellaswag_offline_cases,
    mmlu_offline_cases,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Minimum case counts for "substantive" offline suites (not smoke).
MIN_OFFLINE_CASES = 40

_OFFLINE_WRITERS: tuple[dict[str, Any], ...] = (
    {
        "name": "mmlu_offline",
        "filename": "mmlu_offline.v1.json",
        "family": "mmlu",
        "title": "Synthetic knowledge MCQ (MMLU-shaped)",
        "description": (
            "Synthetic offline knowledge suite with MMLU-shaped multiple-choice. "
            "NOT the HuggingFace MMLU dataset and NOT an industry score — "
            "synthetic/offline only; no licensed external data is bundled."
        ),
        "version": 1,
        "builder": mmlu_offline_cases,
    },
    {
        "name": "gsm8k_offline",
        "filename": "gsm8k_offline.v1.json",
        "family": "gsm8k",
        "title": "Synthetic arithmetic (GSM8K-shaped)",
        "description": (
            "Synthetic offline grade-school math suite with GSM8K-shaped items. "
            "NOT the HuggingFace GSM8K dataset and NOT an industry score — "
            "synthetic/offline only; no licensed external data is bundled."
        ),
        "version": 1,
        "builder": gsm8k_offline_cases,
    },
    {
        "name": "hellaswag_offline",
        "filename": "hellaswag_offline.v1.json",
        "family": "hellaswag",
        "title": "Synthetic completion MCQ (HellaSwag-shaped)",
        "description": (
            "Synthetic offline sentence-completion suite with HellaSwag-shaped "
            "items. NOT the HuggingFace HellaSwag dataset and NOT an industry "
            "score — synthetic/offline only; no licensed external data is bundled."
        ),
        "version": 1,
        "builder": hellaswag_offline_cases,
    },
)


def fixtures_dir() -> Path:
    """Directory containing versioned suite JSON fixtures."""
    return _FIXTURES_DIR


def offline_suite_metas() -> list[dict[str, Any]]:
    """Metadata for discoverable offline suites (without builder callables)."""
    out: list[dict[str, Any]] = []
    for meta in _OFFLINE_WRITERS:
        out.append({
            "name": str(meta["name"]),
            "filename": str(meta["filename"]),
            "family": str(meta["family"]),
            "title": str(meta["title"]),
            "description": str(meta["description"]),
            "version": int(meta["version"]),
        })
    return out


def build_suite_document(meta: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the on-disk suite JSON object."""
    return {
        "schema_version": 1,
        "id": meta["name"],
        "version": meta["version"],
        "family": meta["family"],
        "suite_type": "synthetic_offline",
        "title": meta["title"],
        "description": meta["description"],
        "synthetic": True,
        "offline": True,
        "is_industry_benchmark": False,
        "industry_benchmark": False,
        "licensed_external_data": False,
        "scoring": "strict_task_aware",
        "cases": cases,
    }


def write_offline_fixtures(*, force: bool = False) -> list[Path]:
    """Write offline fixture JSON files under ``fixtures/``. Returns paths written."""
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for meta in _OFFLINE_WRITERS:
        path = _FIXTURES_DIR / str(meta["filename"])
        if path.is_file() and not force:
            written.append(path)
            continue
        builder: Callable[[], list[dict[str, Any]]] = meta["builder"]
        cases = builder()
        if len(cases) < MIN_OFFLINE_CASES:
            raise ValueError(
                f"{meta['name']} has {len(cases)} cases; need ≥{MIN_OFFLINE_CASES}"
            )
        doc = build_suite_document(meta, cases)
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def ensure_offline_fixtures() -> list[Path]:
    """Ensure offline fixtures exist on disk (idempotent)."""
    return write_offline_fixtures(force=False)
