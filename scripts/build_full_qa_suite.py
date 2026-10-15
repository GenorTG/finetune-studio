"""Build a benchmark suite from every approved source-grounded QA pair.

Thin CLI wrapper around ``finetune_studio.testing.full_corpus_suite``.
Prefer discovery via ``discover_suites(project_id)`` in the WebUI.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from finetune_studio.testing.full_corpus_suite import (
    FULL_CORPUS_SUITE_NAME,
    write_full_corpus_suite_from_project_dir,
)


def build_suite(project_dir: Path, output_path: Path) -> dict[str, int]:
    """Write suite JSON for a project directory; returns case/source counts."""
    result = write_full_corpus_suite_from_project_dir(project_dir, output_path)
    return {"cases": result.case_count, "source_ids": result.source_count}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=f"Build {FULL_CORPUS_SUITE_NAME}.json from approved QA pairs",
    )
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_suite(args.project_dir, args.output_path), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
