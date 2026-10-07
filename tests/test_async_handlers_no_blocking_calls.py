"""Guard: no ``async def`` route handler may call known long-blocking work inline.

Every call below blocks for seconds to an hour (subprocess, GPU merge, model
unload, tree deletes, archive building). Inline in an ``async def`` it freezes
the whole server — every page, status poll and SSE stream — until it returns.
Run it via ``await asyncio.to_thread(...)`` or as an export job
(``webui/export_jobs.py``). Nested sync ``def``s are ignored: they are the
worker bodies handed to ``to_thread``.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROUTES = Path(__file__).resolve().parents[1] / "src" / "finetune_studio" / "webui" / "routes"

BLOCKING = {
    "subprocess.run", "subprocess.Popen", "subprocess.check_output", "subprocess.call",
    "time.sleep",
    "shutil.rmtree", "shutil.copytree", "shutil.make_archive", "shutil.move",
    "export_trained_run", "ensure_merged_for_export", "merge_adapter_for_run",
    "convert_merged_to_gguf", "quantize_gguf_imatrix", "run_group_subprocess",
    "unload_all_models", "release_rag_models", "snapshot_download", "hf_hub_download",
}

# Known offenders in files owned by another lane (provider routes); fix there, then drop the entry.
KNOWN: set[tuple[str, str]] = set()


class _Finder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: list[str] = []
        self.hits: list[tuple[str, int, str]] = []

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if not self.stack:
            self.generic_visit(node)  # module-level sync def: look for nested async defs
        # inside an async def: a sync worker body, skipped on purpose

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Call(self, node: ast.Call) -> None:
        if self.stack and ast.unparse(node.func) in BLOCKING:
            self.hits.append((self.stack[-1], node.lineno, ast.unparse(node.func)))
        self.generic_visit(node)


def test_no_async_handler_runs_blocking_work_inline() -> None:
    offenders = []
    for path in sorted(ROUTES.glob("*.py")):
        finder = _Finder()
        finder.visit(ast.parse(path.read_text(encoding="utf-8")))
        offenders += [
            f"{path.name}:{line} {fn}() calls {call}"
            for fn, line, call in finder.hits
            if (path.name, fn) not in KNOWN
        ]
    assert not offenders, "blocking call on the event loop:\n" + "\n".join(offenders)


def test_known_offenders_still_exist() -> None:
    """Stale allow-list entries must be removed once the handler is fixed."""
    for fname, fn in KNOWN:
        finder = _Finder()
        finder.visit(ast.parse((ROUTES / fname).read_text(encoding="utf-8")))
        assert any(h[0] == fn for h in finder.hits), f"{fname}:{fn} is clean now; drop it from KNOWN"
