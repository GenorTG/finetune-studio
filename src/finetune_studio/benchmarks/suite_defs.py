"""Versioned built-in benchmark suite definitions and discovery.

Synthetic offline / smoke suites ship as JSON under ``fixtures/``. They are
data-agnostic (no HuggingFace / network), clearly labeled as synthetic — not
industry MMLU/GSM8K/HellaSwag scores — and selectable alongside project
auto-suites and ``data/benchmarks/*.json`` files.

Real suites (``suite_type=real``) use official HuggingFace splits via
``real://…`` virtual paths; they require a datasets cache on first run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SuiteType = Literal[
    "synthetic_smoke",
    "synthetic_offline",
    "real",
    "local",
    "auto",
]
SuiteSource = Literal["builtin", "data_benchmarks", "auto_suites", "huggingface"]

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
        "title": "Synthetic knowledge MCQ (MMLU-shaped)",
        "description": (
            "Built-in synthetic smoke suite with MMLU-shaped multiple-choice "
            "items. NOT the HuggingFace MMLU dataset and NOT an industry score "
            "— synthetic/offline only; no licensed external data."
        ),
        "version": 1,
    },
    {
        "name": "gsm8k_smoke",
        "filename": "gsm8k_smoke.v1.json",
        "family": "gsm8k",
        "title": "Synthetic arithmetic (GSM8K-shaped)",
        "description": (
            "Built-in synthetic smoke suite with GSM8K-shaped grade-school math. "
            "NOT the HuggingFace GSM8K dataset and NOT an industry score — "
            "synthetic/offline only; no licensed external data."
        ),
        "version": 1,
    },
    {
        "name": "hellaswag_smoke",
        "filename": "hellaswag_smoke.v1.json",
        "family": "hellaswag",
        "title": "Synthetic completion MCQ (HellaSwag-shaped)",
        "description": (
            "Built-in synthetic smoke suite with HellaSwag-shaped sentence "
            "completion. NOT the HuggingFace HellaSwag dataset and NOT an "
            "industry score — synthetic/offline only; no licensed external data."
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
    is_industry_benchmark: bool = False
    is_real_benchmark: bool = False
    dataset_id: str = ""
    dataset_config: str | None = None
    split: str = ""
    prompt_protocol: str = ""
    scoring_method: str = ""
    default_sample_limit: int | None = None

    def label(self) -> str:
        """Human-readable label for dropdowns and lists."""
        if self.suite_type == "real":
            n = self.case_count
            count = f" · {n} official cases" if n is not None else ""
            return f"real · {self.title} (HuggingFace · industry){count}"
        if self.suite_type == "synthetic_smoke":
            ver = f" v{self.version}" if self.version is not None else ""
            n = self.case_count
            count = f" · {n} cases" if n is not None else ""
            return (
                f"synthetic · {self.title} "
                f"(smoke{ver} · not industry){count}"
            )
        if self.suite_type == "synthetic_offline":
            ver = f" v{self.version}" if self.version is not None else ""
            n = self.case_count
            count = f" · {n} cases" if n is not None else ""
            return (
                f"synthetic · {self.title} "
                f"(offline{ver} · not industry){count}"
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
            "is_industry_benchmark": self.is_industry_benchmark,
            "is_real_benchmark": self.is_real_benchmark,
            "industry_benchmark": self.is_real_benchmark,
            "dataset_id": self.dataset_id,
            "dataset_config": self.dataset_config,
            "split": self.split,
            "prompt_protocol": self.prompt_protocol,
            "scoring_method": self.scoring_method,
            "default_sample_limit": self.default_sample_limit,
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
    """Return built-in synthetic smoke suites whose fixture files exist."""
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
                suite_type="synthetic_smoke",
                source="builtin",
                version=int(meta["version"]),
                family=str(meta["family"]),
                case_count=_case_count_from_file(path),
                is_industry_benchmark=False,
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
                suite_type="synthetic_offline",
                source="builtin",
                version=int(meta["version"]),
                family=str(meta["family"]),
                case_count=_case_count_from_file(path),
                is_industry_benchmark=False,
            )
        )
    return out


def list_builtin_industry_suites() -> list[SuiteDefinition]:
    """Smoke + substantive offline built-ins (legacy name; all synthetic)."""
    return list_builtin_smoke_suites() + list_builtin_offline_suites()


def list_builtin_real_suites() -> list[SuiteDefinition]:
    """Official HuggingFace MMLU / GSM8K / HellaSwag suites (virtual paths)."""
    from finetune_studio.benchmarks.real_benchmarks import (
        DEFAULT_SAMPLE_LIMIT,
        REAL_SPECS,
        list_real_families,
        real_suite_path,
    )

    out: list[SuiteDefinition] = []
    for family in list_real_families():
        spec = REAL_SPECS[family]
        out.append(
            SuiteDefinition(
                name=spec.name,
                path=real_suite_path(family),
                title=spec.title,
                description=spec.description,
                suite_type="real",
                source="huggingface",
                version=1,
                family=family,
                case_count=spec.official_case_count,
                is_industry_benchmark=True,
                is_real_benchmark=True,
                dataset_id=spec.dataset_id,
                dataset_config=spec.dataset_config,
                split=spec.split,
                prompt_protocol=spec.prompt_protocol,
                scoring_method=spec.scoring_method,
                default_sample_limit=DEFAULT_SAMPLE_LIMIT,
            )
        )
    return out


def format_auto_suite_label(suite_name: str, case_count: int) -> str:
    """Label for a project auto-generated suite."""
    return f"auto · {suite_name} ({case_count} cases)"


def _sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    """Real first, then offline, smoke, local, auto."""
    st = str(entry.get("suite_type") or "")
    order = {
        "real": 0,
        "synthetic_offline": 1,
        "synthetic_smoke": 2,
        # Accept legacy keys if old fixtures linger in-memory.
        "industry_offline": 1,
        "industry_smoke": 2,
        "local": 3,
        "auto": 4,
    }.get(st, 9)
    return (order, str(entry.get("label") or entry.get("name") or ""))


def discover_suites(project_id: str | None = None) -> list[dict[str, Any]]:
    """Discover selectable suites: real HF, offline, smoke, local JSON, auto.

    Known / discovered files under ``data/benchmarks/`` are included only when
    the path is present on disk. Built-in fixtures are always listed when
    their package files exist. Real suites use ``real://`` virtual paths.
    When ``project_id`` is set, rows from ``auto_suites`` for that project
    are appended.
    """
    found: dict[str, dict[str, Any]] = {}

    for real in list_builtin_real_suites():
        found[f"real:{real.name}"] = real.as_dict()

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
    from finetune_studio.benchmarks.real_benchmarks import is_real_suite_path

    raw = str(suite_path)
    if is_real_suite_path(raw):
        return raw in selectable_paths(project_id)
    allowed = selectable_paths(project_id)
    if raw in allowed:
        return True
    try:
        resolved = str(Path(suite_path).resolve())
    except OSError:
        return False
    resolved_allowed = set()
    for p in allowed:
        if is_real_suite_path(p):
            continue
        try:
            resolved_allowed.add(str(Path(p).resolve()))
        except OSError:
            continue
    return resolved in resolved_allowed
