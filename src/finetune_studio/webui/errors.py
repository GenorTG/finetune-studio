"""The API error contract: a real 4xx/5xx status and ``{"detail": "<what happened and what to do>", "code": ...}``.

- ``raise ApiError(409, "a build is already running", code="busy", job_id=...)``: extra keyword fields go into the body.
- ``HTTPException`` (FastAPI's) keeps working and gets the same shape.
- Validation errors: 422 with one readable sentence in ``detail`` (field: problem), the raw list in ``errors``.
- Anything unhandled: 500 ``{"detail": "internal error ... request id <id>", "code": "internal"}`` and the traceback
  in ``logs/web.log`` under that id. Never a 200 with an error inside.
- Browser page loads that 404 get the styled page instead of JSON.

Frontend side: ``static/js/api.js`` (``api.get/post/...``) throws an Error whose message is ``detail``.
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from finetune_studio import logsetup
from finetune_studio.knowledge.errors import KnowledgeError

_log = logging.getLogger(__name__)

_NOT_FOUND_HTML = (
    "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>Page not found · Finetune Studio</title><style>"
    "body{font-family:system-ui,sans-serif;background:#12161a;color:#dde;"
    "display:grid;place-items:center;min-height:100vh;margin:0}"
    "main{text-align:center;padding:24px}a{color:#7fd48a}</style></head>"
    "<body><main><h1>Page not found</h1>"
    "<p>That address does not exist in Finetune Studio.</p>"
    "<p><a href='/'>Back to the dashboard</a></p></main></body></html>"
)

_DEFAULT_CODES = {400: "bad_request", 401: "auth_required", 403: "forbidden", 404: "not_found", 405: "method_not_allowed",
                  409: "conflict", 413: "too_large", 422: "invalid_request", 423: "locked", 429: "throttled",
                  500: "internal", 502: "upstream_failed", 503: "unavailable", 504: "upstream_timeout"}


class ApiError(StarletteHTTPException):
    """An expected failure with a reason the user can act on."""

    def __init__(self, status: int, detail: str, *, code: str | None = None, **extra: Any) -> None:
        super().__init__(status_code=status, detail=detail)
        self.code = code or _DEFAULT_CODES.get(status, "error")
        self.extra = extra


def error_body(status: int, detail: Any, code: str | None = None, **extra: Any) -> dict[str, Any]:
    if isinstance(detail, dict):                      # HTTPException(detail={...}) from older code
        extra = {**{k: v for k, v in detail.items() if k not in ("detail", "error", "code")}, **extra}
        code = code or detail.get("code")
        detail = detail.get("detail") or detail.get("error") or str(detail)
    return {"detail": str(detail), "code": code or _DEFAULT_CODES.get(status, "error"), **extra}


def error_response(status: int, detail: Any, code: str | None = None, *, headers: dict[str, str] | None = None,
                   **extra: Any) -> JSONResponse:
    return JSONResponse(error_body(status, detail, code, **extra), status_code=status, headers=headers)


def _validation_sentence(errors: list[dict[str, Any]]) -> str:
    parts = []
    for e in errors[:5]:
        loc = ".".join(str(x) for x in e.get("loc", ()) if x not in ("body", "query", "path"))
        parts.append(f"{loc}: {e.get('msg', 'invalid')}" if loc else str(e.get("msg", "invalid")))
    more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
    return "; ".join(parts) + more


def install(app: FastAPI) -> None:
    @app.exception_handler(KnowledgeError)
    async def _knowledge(request: Request, exc: KnowledgeError):
        """A source or index request that cannot be served (unknown, not parsed, nothing to index): 4xx + reason."""
        return error_response(exc.status, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        wants_html = "text/html" in request.headers.get("accept", "")
        if exc.status_code == 404 and wants_html and not request.url.path.startswith("/api/"):
            return HTMLResponse(_NOT_FOUND_HTML, status_code=404)
        if isinstance(exc, ApiError):
            return error_response(exc.status_code, exc.detail, exc.code, headers=exc.headers, **exc.extra)
        return error_response(exc.status_code, exc.detail, headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        errs = [{k: v for k, v in e.items() if k in ("loc", "msg", "type")} for e in exc.errors()]
        return error_response(422, _validation_sentence(errs), "invalid_request", errors=errs)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        rid = logsetup.request_id.get()
        _log.exception("unhandled error in %s %s", request.method, request.url.path)
        return error_response(500, f"internal error: {type(exc).__name__}. It is logged in logs/web.log under "
                                   f"request id {rid}.", "internal", request_id=rid)
