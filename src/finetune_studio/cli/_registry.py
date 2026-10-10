"""Registry: maps subcommand name → handler function.

Single responsibility: turn `args.command` into the right handler, and
run the entry point.

Adding a new subcommand:
  1. Add the parser in `_parser.py`
  2. Add `cmd_<name>` in `commands/<name>.py`
  3. Register the handler in `COMMANDS` below.
"""
from __future__ import annotations

import sys

from finetune_studio.cli._parser import build_parser
from finetune_studio.cli.commands.accel import cmd_accel
from finetune_studio.cli.commands.analyze import cmd_analyze
from finetune_studio.cli.commands.augment import cmd_augment
from finetune_studio.cli.commands.benchmark import cmd_benchmark
from finetune_studio.cli.commands.compare import cmd_compare
from finetune_studio.cli.commands.convert import cmd_convert
from finetune_studio.cli.commands.dataset import cmd_dataset
from finetune_studio.cli.commands.files import cmd_files
from finetune_studio.cli.commands.models import cmd_models
from finetune_studio.cli.commands.optimize import cmd_optimize
from finetune_studio.cli.commands.rag import cmd_rag
from finetune_studio.cli.commands.rag_test import cmd_rag_test
from finetune_studio.cli.commands.suite import cmd_suite
from finetune_studio.cli.commands.supervisor import (
    cmd_component_action,
    cmd_doctor,
    cmd_events,
    cmd_logs,
    cmd_status,
    cmd_supervisor,
    cmd_up,
)
from finetune_studio.cli.commands.test import cmd_test
from finetune_studio.cli.commands.train import cmd_train
from finetune_studio.cli.commands.validate import cmd_validate
from finetune_studio.cli.commands.validate_hallucination import (
    cmd_validate_hallucination,
)
from finetune_studio.cli.commands.vram import cmd_vram
from finetune_studio.cli.commands.webui import cmd_webui

COMMANDS = {
    "models": cmd_models,
    "train": cmd_train,
    "test": cmd_test,
    "suite": cmd_suite,
    "validate": cmd_validate,
    "convert": cmd_convert,
    "webui": cmd_webui,
    "rag": cmd_rag,
    "compare": cmd_compare,
    "benchmark": cmd_benchmark,
    "analyze": cmd_analyze,
    "augment": cmd_augment,
    "optimize": cmd_optimize,
    "validate-hallucination": cmd_validate_hallucination,
    "rag-test": cmd_rag_test,
    "vram": cmd_vram,
    "accel": cmd_accel,
    "files": cmd_files,
    "dataset": cmd_dataset,
    "supervisor": cmd_supervisor,
    "status": cmd_status,
    "start": cmd_component_action,
    "stop": cmd_component_action,
    "restart": cmd_component_action,
    "logs": cmd_logs,
    "events": cmd_events,
    "up": cmd_up,
    "doctor": cmd_doctor,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    try:
        COMMANDS[args.command](args)
    except OSError as exc:
        # Missing/unreadable/unwritable user paths are user errors, not crashes.
        print(f"Error: {exc.strerror or exc}: {exc.filename or ''}".rstrip(": "), file=sys.stderr)
        sys.exit(1)
