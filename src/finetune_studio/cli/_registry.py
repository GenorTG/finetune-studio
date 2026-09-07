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
from finetune_studio.cli.commands.analyze import cmd_analyze
from finetune_studio.cli.commands.augment import cmd_augment
from finetune_studio.cli.commands.benchmark import cmd_benchmark
from finetune_studio.cli.commands.compare import cmd_compare
from finetune_studio.cli.commands.convert import cmd_convert
from finetune_studio.cli.commands.models import cmd_models
from finetune_studio.cli.commands.optimize import cmd_optimize
from finetune_studio.cli.commands.rag import cmd_rag
from finetune_studio.cli.commands.rag_test import cmd_rag_test
from finetune_studio.cli.commands.suite import cmd_suite
from finetune_studio.cli.commands.test import cmd_test
from finetune_studio.cli.commands.train import cmd_train
from finetune_studio.cli.commands.validate import cmd_validate
from finetune_studio.cli.commands.validate_hallucination import cmd_validate_hallucination
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
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    COMMANDS[args.command](args)
