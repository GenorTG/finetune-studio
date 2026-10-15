"""Regression tests for Lane F (webui core/glue + CLI + entrypoints) audit fixes.

Pins: `finetune_studio.cli` resolves to the `cli/` package (never the deleted
`cli.py` shim), the package entry point exposes `main`, and the templates
package/manager modules don't carry a dead duplicate module-docstring
statement that silently does nothing.
"""
from __future__ import annotations

import subprocess
import sys


def test_cli_py_shim_file_removed():
    """src/finetune_studio/cli.py was dead code — unreachable because the
    cli/ package in the same directory always wins package-vs-module import
    resolution. It must stay deleted, not resurrected as a confusing no-op.
    """
    import pathlib

    pkg_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
    assert not (pkg_root / "cli.py").exists()
    assert (pkg_root / "cli" / "__init__.py").exists()


def test_finetune_studio_cli_resolves_to_package():
    import finetune_studio.cli as m

    assert m.__file__.endswith("cli/__init__.py") or m.__file__.endswith(
        "cli\\__init__.py"
    )
    assert hasattr(m, "main")
    assert hasattr(m, "COMMANDS")


def test_entry_point_target_is_importable_and_callable():
    """pyproject.toml's `finetune_studio.cli:main` entry point must resolve
    to a real, callable function (not silently fall through to a dead
    module if the package/module shadowing ever regresses).
    """
    from finetune_studio.cli import main

    assert callable(main)


def test_python_dash_m_finetune_studio_runs_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "finetune_studio", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "finetune-studio" in result.stdout


def test_python_dash_m_finetune_studio_cli_runs_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "finetune_studio.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "finetune-studio" in result.stdout


def test_templates_package_has_no_dead_duplicate_docstring_statement():
    """templates/__init__.py and templates/manager.py each used to carry a
    second bare string literal right after the real module docstring — a
    no-op statement that looked like documentation but was never captured
    anywhere. Pin that each module now has exactly one leading docstring.
    """
    import ast
    import pathlib

    pkg_root = pathlib.Path(__file__).resolve().parents[1] / "src" / "finetune_studio"
    for rel in ("templates/__init__.py", "templates/manager.py"):
        tree = ast.parse((pkg_root / rel).read_text())
        # A dead duplicate docstring shows up as a second top-level `Expr`
        # node containing a `Constant` string immediately after the first.
        string_exprs = [
            node
            for node in tree.body[:2]
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        assert len(string_exprs) <= 1, f"{rel} has a dead duplicate docstring statement"
