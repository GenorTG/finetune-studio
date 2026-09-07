"""`fts validate` — validate training data files."""
from __future__ import annotations


def cmd_validate(args) -> None:
    from finetune_studio.data.validator import validate_file
    for f in args.files:
        report = validate_file(f)
        icon = "✅" if report["valid"] else "❌"
        print(f"{icon} {report['name']}: {report['stats']}")
        for e in report["errors"]:
            print(f"   ERROR: {e}")
        for w in report["warnings"]:
            print(f"   WARN: {w}")
