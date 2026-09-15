"""Versioned built-in benchmark suite definitions and discovery.

Industry-style smoke suites and larger synthetic offline suites ship as JSON
under ``fixtures/``. They are data-agnostic (no HuggingFace / network), clearly
labeled as synthetic/offline, and selectable alongside project auto-suites and
``data/benchmarks/*.json`` files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SuiteType = Literal["industry_smoke", "industry_offline", "local", "auto"]
SuiteSource = Literal["builtin", "data_benchmarks", "auto_suites"]

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_KNOWN_LOCAL_SUITES: tuple[dict[str, str], ...] = (
    {
        "name": "default",
        "path": "data/benchmarks/default.json",
        "description": "General-purpose benchmark",
    },
    {
        "name": "tool_calling",
        "path": "data/benchmarks/tool_calling.json",
        "description": "Tool-calling accuracy",
    },
    {
        "name": "chris_ai_v21",
        "path": "data/benchmarks/chris_ai_v21.json",
        "description": "Chris AI v21 suite",
    },
)

_BUILTIN_SMOKE: tuple[dict[str, Any], ...] = (
    {
        "name": "mmlu_smoke",
        "filename": "mmlu_smoke.v1.json",
        "family": "mmlu",
        "title": "MMLU-style knowledge",
        "description": (
            "Local smoke suite styled after MMLU multiple-choice knowledge. "
            "Not the HuggingFace MMLU dataset — synthetic/offline only."
        ),
        "version": 1,
    },
    {
        "name": "gsm8k_smoke",
        "filename": "gsm8k_smoke.v1.json",
        "family": "gsm8k",
        "title": "GSM8K-style arithmetic",
        "description": (
            "Local smoke suite styled after GSM8K grade-school math. "
            "Not the HuggingFace GSM8K dataset — synthetic/offline only."
        ),
        "version": 1,
    },
    {
        "name": "hellaswag_smoke",
        "filename": "hellaswag_smoke.v1.json",
        "family": "hellaswag",
        "title": "HellaSwag-style completion",
        "description": (
            "Local smoke suite styled after HellaSwag sentence completion. "
            "Not the HuggingFace HellaSwag dataset — synthetic/offline only."
        ),
        "version": 1,
    },
)


@dataclass(frozen=True)
class SuiteDefinition:
    """Metadata for a discoverable benchmark suite."""

    name: str
    path: str
    title: str
    description: str
    suite_type: SuiteType
    source: SuiteSource
    version: int | None = None
    family: str = ""
    case_count: int | None = None

    def label(self) -> str:
        """Human-readable label for dropdowns and lists."""
        if self.suite_type == "industry_smoke":
            ver = f" v{self.version}" if self.version is not None else ""
            n = self.case_count
            count = f" · {n} cases" if n is not None else ""
            return f"industry · {self.title} (smoke{ver} · synthetic/offline){count}"
        if self.suite_type == "industry_offline":
            ver = f" v{self.version}" if self.version is not None else ""
            n = self.case_count
            count = f" · {n} cases" if n is not None else ""
            return (
                f"industry · {self.title} "
                f"(offline synthetic{ver}){count}"
            )
        if self.suite_type == "auto":
            n = self.case_count or 0
            return f"auto · {self.name} ({n} cases)"
        title = self.title or self.name
        return f"local · {title}"

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API / template discovery entries."""
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "title": self.title,
            "label": self.label(),
            "suite_type": self.suite_type,
            "source": self.source,
            "version": self.version,
            "family": self.family,
            "case_count": self.case_count,
        }


def fixtures_dir() -> Path:
    """Directory containing versioned suite JSON fixtures."""
    return _FIXTURES_DIR


def _case_count_from_file(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        cases = data.get("cases")
        if isinstance(cases, list):
            return len(cases)
    return None


def list_builtin_smoke_suites() -> list[SuiteDefinition]:
    """Return built-in industry smoke suites whose fixture files exist."""
    out: list[SuiteDefinition] = []
    for meta in _BUILTIN_SMOKE:
        path = _FIXTURES_DIR / str(meta["filename"])
        if not path.is_file():
            continue
        out.append(
            SuiteDefinition(
                name=str(meta["name"]),
                path=str(path),
                title=str(meta["title"]),
                description=str(meta["description"]),
                suite_type="industry_smoke",
                source="builtin",
                version=int(meta["version"]),
                family=str(meta["family"]),
                case_count=_case_count_from_file(path),
            )
        )
    return out


def list_builtin_offline_suites() -> list[SuiteDefinition]:
    """Return substantive synthetic offline suites (ensure fixtures on disk)."""
    from finetune_studio.benchmarks.offline_suites import (
        ensure_offline_fixtures,
        offline_suite_metas,
    )

    ensure_offline_fixtures()
    out: list[SuiteDefinition] = []
    for meta in offline_suite_metas():
        path = _FIXTURES_DIR / str(meta["filename"])
        if not path.is_file():
            continue
        out.append(
            SuiteDefinition(
                name=str(meta["name"]),
                path=str(path),
                title=str(meta["title"]),
                description=str(meta["description"]),
                suite_type="industry_offline",
                source="builtin",
                version=int(meta["version"]),
                family=str(meta["family"]),
                case_count=_case_count_from_file(path),
            )
        )
    return out


def list_builtin_industry_suites() -> list[SuiteDefinition]:
    """Smoke + substantive offline built-ins."""
    return list_builtin_smoke_suites() + list_builtin_offline_suites()


def format_auto_suite_label(suite_name: str, case_count: int) -> str:
    """Label for a project auto-generated suite."""
    return f"auto · {suite_name} ({case_count} cases)"


def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    """Offline first (substantive), then smoke, then local, then auto."""
    st = str(entry.get("suite_type") or "")
    order = {
        "industry_offline": 0,
        "industry_smoke": 1,
        "local": 2,
        "auto": 3,
    }.get(st, 9)
    return (order, str(entry.get("label") or entry.get("name") or ""))


def discover_suites(project_id: str | None = None) -> list[dict[str, Any]]:
    """Discover selectable suites: offline, smoke, local JSON, project auto.

    Known / discovered files under ``data/benchmarks/`` are included only when
    the path is present on disk. Built-in fixtures are always listed when
    their package files exist. When ``project_id`` is set, rows from
    ``auto_suites`` for that project are appended.
    """
    found: dict[str, dict[str, Any]] = {}

    for industry in list_builtin_industry_suites():
        found[f"builtin:{industry.name}"] = industry.as_dict()

    for s in _KNOWN_LOCAL_SUITES:
        path = s["path"]
        if Path(path).is_file():
            name = s["name"]
            desc = s.get("description", "")
            found[name] = SuiteDefinition(
                name=name,
                path=path,
                title=desc or name,
                description=desc,
                suite_type="local",
                source="data_benchmarks",
                case_count=_case_count_from_file(Path(path)),
            ).as_dict()

    bench_dir = Path("data/benchmarks")
    if bench_dir.is_dir():
        for f in sorted(bench_dir.glob("*.json")):
            name = f.stem
            key = name
            if key in found:
                continue
            # Skip if already registered under a different key with same path
            if any(e.get("path") == str(f) for e in found.values()):
                continue
            found[key] = SuiteDefinition(
                name=name,
                path=str(f),
                title=name,
                description=f"Discovered: {f.name}",
                suite_type="local",
                source="data_benchmarks",
                case_count=_case_count_from_file(f),
            ).as_dict()

    if project_id:
        from finetune_studio import db

        with db.cursor() as c:
            rows = c.execute(
                "SELECT suite_name, suite_path, case_count FROM auto_suites "
                "WHERE project_id = ? ORDER BY created_at DESC",
                (project_id,),
            ).fetchall()
        for row in rows:
            suite_name = str(row["suite_name"])
            suite_path = str(row["suite_path"])
            case_count = int(row["case_count"] or 0)
            key = f"auto:{suite_name}:{suite_path}"
            found[key] = SuiteDefinition(
                name=suite_name,
                path=suite_path,
                title=suite_name,
                description=format_auto_suite_label(suite_name, case_count),
                suite_type="auto",
                source="auto_suites",
                case_count=case_count,
            ).as_dict()

    return sorted(found.values(), key=_sort_key)


def selectable_paths(project_id: str | None = None) -> set[str]:
    """Paths currently offered by discovery (for selection validation)."""
    return {str(s["path"]) for s in discover_suites(project_id)}


def is_selectable_suite(
    suite_path: str,
    project_id: str | None = None,
) -> bool:
    """True when suite_path is among discovered selectable suites."""
    if not suite_path or not str(suite_path).strip():
        return False
    allowed = selectable_paths(project_id)
    raw = str(suite_path)
    if raw in allowed:
        return True
    try:
        resolved = str(Path(suite_path).resolve())
    except OSError:
        return False
    resolved_allowed = set()
    for p in allowed:
        try:
            resolved_allowed.add(str(Path(p).resolve()))
        except OSError:
            continue
    return resolved in resolved_allowed
