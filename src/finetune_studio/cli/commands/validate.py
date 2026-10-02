"""`fts validate` — validate training data files."""
from __future__ import annotations

import sys


def cmd_validate(args) -> None:
    """Report on every file, then exit 1 if any file is invalid."""
    from finetune_studio.data.validator import validate_file

    any_invalid = False
    for f in args.files:
        report = validate_file(f)
        icon = "✅" if report["valid"] else "❌"
        print(f"{icon} {report['name']}: {report['stats']}")
        for e in report["errors"]:
            print(f"   ERROR: {e}")
        for w in report["warnings"]:
            print(f"   WARN: {w}")
        any_invalid = any_invalid or not report["valid"]
    if any_invalid:
        sys.exit(1)
