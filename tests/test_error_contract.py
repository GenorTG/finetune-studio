"""The API error contract (2026-10-15): a real 4xx/5xx and ``{"detail", "code"}``; never a 2xx carrying a failure."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from finetune_studio.webui import errors
from finetune_studio.webui.errors import ApiError

# Built from parts: tests/test_no_private_hosts.py rejects key-shaped literals in the tree.
FAKE_KEY = "sk-" + "abcdefghijklmnop1234"

SRC = Path(__file__).resolve().parents[1] / "src" / "finetune_studio"


class Body(BaseModel):
    name: str
    count: int


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    errors.install(app)

    @app.get("/api/busy")
    def busy() -> None:
        raise ApiError(409, "a build is already running", code="busy", job_id="j1")

    @app.get("/api/plain")
    def plain() -> None:
        raise HTTPException(status_code=404, detail="no such project")

    @app.post("/api/body")
    def body(b: Body) -> dict:
        return {"ok": True}

    @app.get("/api/crash")
    def crash() -> None:
        raise RuntimeError("secret internals with api_key=" + FAKE_KEY)

    return TestClient(app, raise_server_exceptions=False)


def test_api_error_carries_status_detail_code_and_extras(client: TestClient) -> None:
    r = client.get("/api/busy")
    assert r.status_code == 409 and r.json() == {"detail": "a build is already running", "code": "busy", "job_id": "j1"}


def test_plain_http_exceptions_get_the_same_shape(client: TestClient) -> None:
    r = client.get("/api/plain")
    assert r.status_code == 404 and r.json() == {"detail": "no such project", "code": "not_found"}


def test_validation_errors_are_one_readable_sentence(client: TestClient) -> None:
    r = client.post("/api/body", json={"name": "x", "count": "many"})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "invalid_request" and body["detail"].startswith("count:") and body["errors"]


def test_a_crash_is_a_500_that_names_the_log_and_leaks_nothing(client: TestClient) -> None:
    r = client.get("/api/crash")
    body = r.json()
    assert r.status_code == 500 and body["code"] == "internal" and "logs/web.log" in body["detail"]
    assert "sk-abcdef" not in r.text and "secret internals" not in r.text


def test_browser_404_gets_the_styled_page(client: TestClient) -> None:
    r = client.get("/nope", headers={"Accept": "text/html"})
    assert r.status_code == 404 and "Page not found" in r.text
    assert client.get("/api/nope", headers={"Accept": "text/html"}).json()["code"] == "not_found"


def _json_error_calls() -> list[str]:
    bad = []
    for f in SRC.rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if (isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == "JSONResponse"
                    and n.args and isinstance(n.args[0], ast.Dict)
                    and any(isinstance(k, ast.Constant) and k.value == "error" for k in n.args[0].keys)):
                bad.append(f"{f.relative_to(SRC)}:{n.lineno}")
    return bad


def test_no_error_response_uses_the_old_error_key() -> None:
    """Error bodies say "detail" (FastAPI's own key), so one frontend helper reads every failure the same way."""
    assert _json_error_calls() == []


def _own_returns(fn: ast.AST):
    """Return statements of this function, not of functions nested in it (those feed a JSONResponse(status=...))."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Return):
            yield n
        stack.extend(ast.iter_child_nodes(n))


def test_no_route_returns_ok_false_with_a_2xx() -> None:
    bad = []
    for f in (SRC / "webui" / "routes").glob("*.py"):
        for fn in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not any(
                    isinstance(d, ast.Call) and getattr(d.func, "attr", "") in ("get", "post", "put", "patch", "delete")
                    for d in fn.decorator_list):
                continue
            for n in _own_returns(fn):
                if isinstance(n.value, ast.Dict):
                    for k, v in zip(n.value.keys, n.value.values, strict=True):
                        if (isinstance(k, ast.Constant) and k.value == "ok" and isinstance(v, ast.Constant)
                                and v.value is False):
                            bad.append(f"{f.name}:{n.lineno}")
    assert bad == [], "a failure must be a 4xx/5xx (raise ApiError), not 200 {ok: false}"
