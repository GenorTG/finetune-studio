"""Errors of the knowledge layer. Each carries the HTTP status a route answers with (``webui.app`` maps them)."""
from __future__ import annotations


class KnowledgeError(Exception):
    def __init__(self, message: str, status: int = 404) -> None:
        super().__init__(message)
        self.status = status


class SourceError(KnowledgeError):
    """A source that cannot be served: unknown, not parsed yet, parse failed, unreadable type."""


class IndexBuildError(KnowledgeError):
    """An index build that cannot run (no project, nothing parsed yet)."""
