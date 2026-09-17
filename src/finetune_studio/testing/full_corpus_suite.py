"""Project-local full-ingested-corpus suite from approved QA pairs.

Builds ``suites/full-ingested-corpus.json`` under a project's FS root from
every approved pair that carries a ``source_id``. Used by suite discovery so
the Testing dropdown can run the whole ingested corpus without hardcoding a
project id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finetune_studio.data.fs.paths import project_dir
from finetune_studio.data.fs.qa import list_qa_pairs

FULL_CORPUS_SUITE_NAME = "full-ingested-corpus"
FULL_CORPUS_CATEGORY = "full-corpus"
FULL_CORPUS_VERSION = "1.0"


@dataclass(frozen=True)
class FullCorpusCase:
    """One suite case derived from an approved QA pair."""

    name: str
    question: str
    correct_answer: str
    source_id: str
    chunk_idx: int = 0
    keywords: tuple[str, ...] = ()
    category: str = FULL_CORPUS_CATEGORY

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "source_id": self.source_id,
            "chunk_idx": self.chunk_idx,
            "keywords": list(self.keywords),
        }


@dataclass(frozen=True)
class FullCorpusBuildResult:
    """Result of writing (or refusing to write) a full-corpus suite file."""

    path: Path
    case_count: int
    source_count: int
    written: bool


def suite_path_for_project(project_id: str) -> Path:
    """Canonical on-disk path for this project's full-ingested-corpus suite."""
    return project_dir(project_id) / "suites" / f"{FULL_CORPUS_SUITE_NAME}.json"


def _message_value(message: dict[str, Any]) -> str:
    return str(message.get("content") or message.get("value") or "").strip()


def _pair_messages(pair: dict[str, Any]) -> list[dict[str, Any]]:
    messages = pair.get("messages")
    if isinstance(messages, list):
        return [m for m in messages if isinstance(m, dict)]
    conversations = pair.get("conversations")
    if isinstance(conversations, list):
        return [m for m in conversations if isinstance(m, dict)]
    return []


def extract_qa_text(pair: dict[str, Any]) -> tuple[str, str]:
    """Return ``(question, answer)`` from pair fields or message roles."""
    question = str(pair.get("question") or "").strip()
    answer = str(pair.get("answer") or "").strip()
    if question and answer:
        return question, answer
    messages = _pair_messages(pair)
    if not question:
        question = next(
            (
                _message_value(m)
                for m in messages
                if m.get("role", m.get("from")) in ("user", "human")
            ),
            "",
        )
    if not answer:
        answer = next(
            (
                _message_value(m)
                for m in messages
                if m.get("role", m.get("from")) in ("assistant", "gpt")
            ),
            "",
        )
    return question, answer


def case_from_pair(pair: dict[str, Any], *, fallback_name: str = "") -> FullCorpusCase | None:
    """Build a suite case from one pair, or None if it cannot be used."""
    status = pair.get("status", "approved")
    if status != "approved":
        return None
    source_id = str(pair.get("source_id") or "").strip()
    if not source_id:
        return None
    question, answer = extract_qa_text(pair)
    if not question or not answer:
        return None
    name = str(pair.get("id") or fallback_name or "").strip() or source_id
    raw_keywords = pair.get("keywords") or []
    keywords: tuple[str, ...] = ()
    if isinstance(raw_keywords, list):
        keywords = tuple(str(k) for k in raw_keywords)
    return FullCorpusCase(
        name=name,
        question=question,
        correct_answer=answer,
        source_id=source_id,
        chunk_idx=int(pair.get("chunk_idx") or 0),
        keywords=keywords,
    )


def cases_from_approved_pairs(project_id: str) -> list[FullCorpusCase]:
    """Collect typed cases from every usable approved pair in the project."""
    cases: list[FullCorpusCase] = []
    for pair in list_qa_pairs(project_id):
        if not isinstance(pair, dict):
            continue
        case = case_from_pair(pair)
        if case is not None:
            cases.append(case)
    return cases


def cases_from_pairs_directory(pairs_dir: Path) -> list[FullCorpusCase]:
    """Collect cases by reading ``*.json`` pair files under ``pairs_dir``."""
    if not pairs_dir.is_dir():
        return []
    cases: list[FullCorpusCase] = []
    for pair_path in sorted(pairs_dir.glob("*.json")):
        try:
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(pair, dict):
            continue
        case = case_from_pair(pair, fallback_name=pair_path.stem)
        if case is not None:
            cases.append(case)
    return cases


def build_suite_document(cases: list[FullCorpusCase]) -> dict[str, Any]:
    """Serialize cases into the on-disk suite JSON shape."""
    return {
        "name": FULL_CORPUS_SUITE_NAME,
        "version": FULL_CORPUS_VERSION,
        "case_count": len(cases),
        "cases": [c.as_dict() for c in cases],
    }


def _write_cases(path: Path, cases: list[FullCorpusCase]) -> FullCorpusBuildResult:
    source_count = len({c.source_id for c in cases})
    if not cases:
        if path.is_file():
            path.unlink()
        return FullCorpusBuildResult(
            path=path,
            case_count=0,
            source_count=0,
            written=False,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_suite_document(cases), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return FullCorpusBuildResult(
        path=path,
        case_count=len(cases),
        source_count=source_count,
        written=True,
    )


def write_full_corpus_suite(
    project_id: str,
    *,
    output_path: Path | None = None,
) -> FullCorpusBuildResult:
    """Write ``full-ingested-corpus.json`` from approved source-grounded pairs.

    When there are no usable pairs, does not write a file (and removes a stale
    suite file if present) and returns ``written=False``.
    """
    path = output_path or suite_path_for_project(project_id)
    return _write_cases(path, cases_from_approved_pairs(project_id))


def write_full_corpus_suite_from_project_dir(
    project_root: Path,
    output_path: Path,
) -> FullCorpusBuildResult:
    """Write suite JSON from ``project_root/qa/pairs`` (CLI / offline use)."""
    return _write_cases(output_path, cases_from_pairs_directory(project_root / "qa" / "pairs"))


def ensure_full_corpus_suite(project_id: str) -> FullCorpusBuildResult:
    """Regenerate the project-local full-corpus suite from current approved QA."""
    return write_full_corpus_suite(project_id)


def suite_definition_from_build(result: FullCorpusBuildResult) -> Any:
    """Map a successful build into a discoverable ``SuiteDefinition``.

    Returns ``None`` when nothing was written. Import of ``SuiteDefinition`` is
    deferred to avoid a cycle with ``benchmarks.suite_defs``.
    """
    if not result.written or result.case_count <= 0:
        return None
    from finetune_studio.benchmarks.suite_defs import SuiteDefinition

    n = result.case_count
    return SuiteDefinition(
        name=FULL_CORPUS_SUITE_NAME,
        path=str(result.path),
        title=FULL_CORPUS_SUITE_NAME,
        description=(
            f"All approved source-grounded QA pairs for this project ({n} cases)"
        ),
        suite_type="local",
        source="project_qa",
        case_count=n,
        family="project_corpus",
    )


def ensure_full_corpus_suite_definition(project_id: str) -> Any:
    """Ensure the suite file exists and return its discovery definition, or None."""
    return suite_definition_from_build(ensure_full_corpus_suite(project_id))
