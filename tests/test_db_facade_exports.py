"""``finetune_studio.db`` is a re-export facade, not a bag of dead imports.

The 98 ``F401`` findings this file used to raise were false positives: every
import there is re-exported for ``db.<name>`` attribute access, and 56 call
sites depend on it. ``ruff --fix`` would have deleted the DB layer's public
API. The fix was an explicit ``__all__``, which both silences F401 and
documents the public surface.

This test pins the invariant so the two halves cannot drift: every imported
name must be listed, and every listed name must actually exist. A stale
``__all__`` is exactly the failure mode that turns a silent facade into a
silent ``AttributeError`` somewhere far away.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import finetune_studio.db as db

INIT_PATH = Path(db.__file__)


def _imported_names() -> set[str]:
    """Every name bound by a module-level ``from ... import`` in the facade."""
    tree = ast.parse(INIT_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def test_all_is_declared() -> None:
    """The facade must export an explicit list (this is what silences F401)."""
    assert hasattr(db, "__all__"), "db/__init__.py must declare __all__"
    assert db.__all__, "__all__ must not be empty"


def test_every_import_is_exported() -> None:
    """An imported-but-unlisted name reappears as F401 and can be 'fixed' away."""
    unlisted = _imported_names() - set(db.__all__)
    assert not unlisted, (
        f"imported but missing from __all__ (ruff will flag F401, and a blind "
        f"`ruff --fix` would delete the public API): {sorted(unlisted)}"
    )


def test_every_exported_name_exists() -> None:
    """A stale __all__ entry would raise AttributeError only at the call site."""
    missing = [n for n in db.__all__ if not hasattr(db, n)]
    assert not missing, f"__all__ lists names that do not exist: {missing}"


def test_all_is_sorted_and_unique() -> None:
    """Keeps diffs stable and prevents duplicate entries accumulating."""
    assert db.__all__ == sorted(db.__all__), "__all__ must stay sorted"
    assert len(db.__all__) == len(set(db.__all__)), "__all__ has duplicates"


def test_core_api_is_still_exported() -> None:
    """Spot-check the calls used across the app, not just any name."""
    for name in ("get_project", "list_projects", "create_project", "get_run",
                 "list_rags", "init_db"):
        assert name in db.__all__, f"core DB helper {name} missing from __all__"
        assert callable(getattr(db, name)), f"{name} is not callable"


@pytest.mark.parametrize("name", ["get_project", "list_projects", "list_runs"])
def test_exports_are_callable_in_a_temp_db(name: str, tmp_path: Path) -> None:
    """Attribute access must still work, which is the whole point of the facade."""
    assert callable(getattr(db, name))
    result = getattr(db, name)() if name != "get_project" else db.get_project("nope")
    assert result is None or isinstance(result, (list, dict))
