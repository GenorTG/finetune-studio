"""scripts/rag_reader_compare.py hands the run route a stored JSON suite, never the raw JSONL quiz (live E2E finding 2026-10-09)."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "rag_reader_compare.py"


@pytest.fixture
def rrc() -> Any:
    spec = importlib.util.spec_from_file_location("rag_reader_compare_script", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_jsonl_quiz_goes_through_the_app_import_and_the_stored_path_is_used(rrc: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quiz = tmp_path / "q.jsonl"
    quiz.write_text('{"id": "e001", "q": "Q?", "expect": ["A"]}\n', encoding="utf-8")
    seen: dict[str, Any] = {}

    class _Resp(io.BytesIO):
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_a: object) -> None:
            return None

    def fake_urlopen(req: Any, timeout: float = 0) -> _Resp:
        seen["url"], seen["body"], seen["type"] = req.full_url, req.data, req.headers["Content-type"]
        return _Resp(json.dumps({"ok": True, "path": "/stored/q.json"}).encode())

    monkeypatch.setattr(rrc.urllib.request, "urlopen", fake_urlopen)
    assert rrc.import_quiz("http://x", "p1", quiz) == "/stored/q.json"
    assert seen["url"] == "http://x/api/benchmarks/projects/p1/suites/import"
    assert seen["type"].startswith("multipart/form-data; boundary=")
    assert b'filename="q.jsonl"' in seen["body"] and b'"id": "e001"' in seen["body"]


def test_a_json_suite_is_used_as_it_is(rrc: Any, tmp_path: Path) -> None:
    suite = tmp_path / "s.json"
    suite.write_text("[]", encoding="utf-8")
    assert rrc.import_quiz("http://unused", "p1", suite) == str(suite)
