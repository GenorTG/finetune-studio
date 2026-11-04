"""Plain names for API requests in the Activity feed ("Edit a Q&A pair" instead of "Update").

Every mutating API request is recorded as an Activity row (``webui.app.record_activity_operations``). Its message used to
be the HTTP verb alone for most routes, so the feed read "Create, Create, Update, Delete". :func:`describe` names the
request from its route: an exact entry of :data:`NAMES` (the routes the app's own pages call), else a fallback built from
the path's resource words ("Create: usecases › drafts"), never a bare verb.
"""
from __future__ import annotations

import re

#: (method, route) -> name. ``{}`` stands for one id segment (project, run, file, case ...); ``*`` for any one segment.
NAMES: dict[tuple[str, str], str] = {
    # Pairs (data prep)
    ("PATCH", "/api/projects/{}/data-prep/qa/{}"): "Edit a Q&A pair",
    ("POST", "/api/projects/{}/data-prep/qa"): "Add a Q&A pair",
    ("POST", "/api/projects/{}/data-prep/preference"): "Build preference pairs",
    ("POST", "/api/projects/{}/data-prep/study"): "Build a study set",
    ("POST", "/api/data-editor/projects/{}/approve"): "Approve a dataset row",
    ("POST", "/api/providers/helper/use"): "Use an installed model as the helper",
    # Files
    ("POST", "/api/projects/{}/files/{}/purge"): "Delete a file from the trash",
    ("DELETE", "/api/projects/{}/files/{}"): "Move a file to the trash",
    # Datasets, projects, versions
    ("POST", "/api/projects/{}/datasets/upload"): "Upload a dataset",
    ("DELETE", "/api/projects/{}"): "Delete project",
    ("PATCH", "/api/projects/{}"): "Update project",
    ("POST", "/api/projects/{}/versions"): "Save a project version",
    ("POST", "/api/projects/{}/usecases/drafts"): "Draft a use case",
    # Chat, inference, Guide
    ("POST", "/api/inference/chat"): "Chat with the loaded model",
    ("POST", "/api/chat-v2/projects/{}/chat"): "Chat with the loaded model",
    ("POST", "/api/inference/load"): "Load model",
    ("POST", "/api/chat-v2/load"): "Load model",
    ("POST", "/api/providers/*/load"): "Load model",
    ("POST", "/api/inference/memory-estimate"): "Estimate a model's memory",
    ("POST", "/api/providers"): "Add a model provider",
    ("POST", "/api/guide/chat"): "Ask the Guide",
    # Smart Brain
    ("POST", "/api/projects/{}/brain/chat"): "Smart Brain chat",
    ("POST", "/api/projects/{}/brain/search"): "Smart Brain search",
    ("POST", "/api/projects/{}/brain/build"): "Build the Smart Brain index",
    ("POST", "/api/projects/{}/brain/build/stop"): "Stop the Smart Brain build",
    ("POST", "/api/projects/{}/brain/quick"): "Quick-index the Smart Brain",
    ("POST", "/api/projects/{}/brain/rebuild-vectors"): "Rebuild the Smart Brain vectors",
    ("POST", "/api/projects/{}/brain/settings"): "Save Smart Brain settings",
    ("POST", "/api/projects/{}/brain/standalone"): "Build a standalone Smart Brain package",
    ("POST", "/api/projects/{}/brain/import"): "Import a Smart Brain bundle",
    ("DELETE", "/api/projects/{}/brain/documents"): "Remove all Smart Brain sources",
    # Testing, benchmarks
    ("POST", "/api/testing/run-brain-suite"): "Run a Smart Brain test",
    ("POST", "/api/testing/evaluate-training"): "Evaluate a training run",
    ("POST", "/api/testing/projects/{}/runs/{}/judge"): "Judge a test run",
    ("POST", "/api/testing/projects/{}/runs/{}/cancel"): "Stop a test run",
    ("DELETE", "/api/testing/projects/{}/runs/{}"): "Delete a test run",
    ("PUT", "/api/testing/projects/{}/runs/{}/cases/{}/verdict"): "Set a verdict",
    ("POST", "/api/projects/{}/testing/reworded-quiz"): "Build a reworded quiz",
    ("POST", "/api/benchmarks/projects/{}/suites/import"): "Import a benchmark suite",
    ("POST", "/api/benchmarks/projects/{}/runs/{}/run"): "Run a benchmark",
    ("POST", "/api/benchmarks/projects/{}/base/run"): "Benchmark the base model",
    ("DELETE", "/api/benchmarks/projects/{}/benchmarks/{}"): "Delete a benchmark result",
    # Training
    ("POST", "/api/training/stop"): "Stop training",
    ("POST", "/api/training/runs/{}/auto-suites/generate"): "Generate test suites for a run",
    ("DELETE", "/api/projects/{}/runs/{}"): "Delete a training run",
    # Settings, access, system
    ("PUT", "/api/settings/assistant/jobs/*"): "Choose the assistant for a job",
    ("PUT", "/api/system/compute-device"): "Set the compute device",
    ("POST", "/api/system/update"): "Update the app",
    ("POST", "/api/auth/login"): "Sign in",
    ("POST", "/api/auth/logout"): "Sign out",
}

_VERBS = {"POST": "Create", "PUT": "Update", "PATCH": "Update", "DELETE": "Delete"}
# An id segment: hex ids (projects, runs, cases, chunk-qualified sources) and plain numbers.
_ID = re.compile(r"^(?:[0-9a-f]{6,}(?:-[0-9a-f]{6,})*|\d+)$")


def _segments(path: str) -> list[str]:
    return [s for s in path.strip("/").split("/") if s]


def _matches(route: str, segs: list[str]) -> bool:
    want = _segments(route)
    return len(want) == len(segs) and all(
        w == "*" or (w == "{}" and _ID.match(s)) or w == s for w, s in zip(want, segs, strict=True))


def describe(path: str, method: str) -> str:
    """The request's name for the Activity feed; never only the verb."""
    method = method.upper()
    segs = _segments(path)
    for (m, route), name in NAMES.items():
        if m == method and _matches(route, segs):
            return name
    if method == "PUT" and segs[:2] == ["api", "settings"] and len(segs) == 3:
        return f"Save {segs[2]} settings"
    if method == "POST" and len(segs) == 4 and segs[:2] == ["api", "settings"] and segs[3] == "test":
        return f"Test the {segs[2]} connection"
    words = [s for s in segs if s not in ("api", "projects") and not _ID.match(s)]
    verb = _VERBS.get(method, "Operation")
    return f"{verb}: {' › '.join(words)}" if words else verb
