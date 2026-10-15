"""One logging setup for every process: ``$FTS_ROOT/logs/<component>.log``, rotating, secrets redacted.

Before 2026-10-15 nothing configured logging: Python's last-resort handler printed WARNING+ to stderr, INFO was
dropped, and the journal kept a few hundred lines, so after a restart nothing was left to read.

- ``setup(component)`` (web, inference, supervisor, training): INFO+ to ``logs/<component>.log`` (10 MB x 5) and
  WARNING+ to stderr (the supervisor ring buffer / journal). Idempotent.
- Every record passes ``redact()`` after formatting, traceback included: ``password=...``, ``"api_key": "..."``,
  ``Authorization: Bearer ...``, cookies, ``?token=`` in URLs and key shapes (``sk-``, ``hf_``, ``ghp_``,
  ``github_pat_``, ``fts_``) become ``***``.
- ``request_id`` (a ContextVar set by the web access log) is stamped on every line logged while serving a request.
"""
from __future__ import annotations

import contextvars
import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
BACKUPS = 5
FORMAT = "%(asctime)s %(levelname)-7s %(component)s %(request_id)s %(name)s: %(message)s"

request_id: contextvars.ContextVar[str] = contextvars.ContextVar("fts_request_id", default="-")

_SECRET_KEYS = (r"password|passwd|passphrase|pass|secret|token|api[_-]?key|apikey|access[_-]?token|refresh[_-]?token"
                r"|hf[_-]?token|authorization|cookie|set-cookie|session|private[_-]?key|client[_-]?secret")
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "key": "value" (JSON, dict reprs with either quote style)
    (re.compile(rf"""(?i)(["']?(?:{_SECRET_KEYS})["']?\s*[:=]\s*)(["'])(.*?)(\2)"""), r"\1\2***\4"),
    # key=value in query strings, form bodies, env dumps, CLI args
    (re.compile(rf"(?i)\b((?:{_SECRET_KEYS})=)([^&\s,;\"']+)"), r"\1***"),
    # header lines: "Authorization: Bearer x", "Cookie: a=b; c=d"
    (re.compile(r"(?i)\b(authorization|cookie|set-cookie|x-api-key)(\s*:\s*)([^\r\n]+)"), r"\1\2***"),
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"), r"\1 ***"),
    # bare key shapes anywhere (a pasted key in an error message)
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
                r"|fts_[A-Za-z0-9_-]{20,})"), "***"),
]


def redact(text: str) -> str:
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class _ContextFilter(logging.Filter):
    def __init__(self, component: str) -> None:
        super().__init__()
        self.component = component

    def filter(self, record: logging.LogRecord) -> bool:
        record.component = getattr(record, "component", self.component)
        record.request_id = request_id.get()
        return True


def log_dir() -> Path:
    root = Path(os.environ.get("FTS_ROOT") or str(Path.home() / ".finetune-studio"))
    return root / "logs"


def _file_handler(path: Path, component: str, level: int) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(path, maxBytes=MAX_BYTES, backupCount=BACKUPS, encoding="utf-8")
    h.setLevel(level)
    h.setFormatter(RedactingFormatter(FORMAT))
    h.addFilter(_ContextFilter(component))
    h._fts_managed = True  # type: ignore[attr-defined]
    return h


def setup(component: str, *, level: int = logging.INFO, stderr_level: int = logging.WARNING) -> Path:
    """Configure the root logger for this process. Returns the log file path.

    ``FTS_LOG_TO_FILES=0`` turns it into a no-op (the test suite sets it: tests must not write into a live log dir).
    """
    root = logging.getLogger()
    path = log_dir() / f"{component}.log"
    if os.environ.get("FTS_LOG_TO_FILES") == "0":
        return path
    if any(getattr(h, "_fts_managed", False) and getattr(h, "baseFilename", "") == str(path) for h in root.handlers):
        return path
    for h in [h for h in root.handlers if getattr(h, "_fts_managed", False)]:
        root.removeHandler(h)
    root.setLevel(min(level, stderr_level))
    root.addHandler(_file_handler(path, component, level))
    err = logging.StreamHandler(sys.stderr)
    err.setLevel(stderr_level)
    err.setFormatter(RedactingFormatter(FORMAT))
    err.addFilter(_ContextFilter(component))
    err._fts_managed = True  # type: ignore[attr-defined]
    root.addHandler(err)
    # Libraries that log their own chatter at INFO stay at WARNING.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "huggingface_hub", "datasets", "PIL", "multipart",
                  "asyncio", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # uvicorn configures its own loggers with their own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").disabled = True     # replaced by the access log below
    return path


def access_logger() -> logging.Logger:
    """``logs/web-access.log``: one line per HTTP request (method, path, status, ms, client, request id)."""
    lg = logging.getLogger("fts.access")
    path = log_dir() / "web-access.log"
    if os.environ.get("FTS_LOG_TO_FILES") == "0":
        return lg
    if not any(getattr(h, "baseFilename", "") == str(path) for h in lg.handlers):
        for h in list(lg.handlers):
            lg.removeHandler(h)
        lg.addHandler(_file_handler(path, "web", logging.INFO))
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg
