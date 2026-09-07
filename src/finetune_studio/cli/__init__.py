"""Finetune Studio CLI — the `fts` command.

LAYOUT
------
cli/
  __init__.py     — public API (re-export `main` for back-compat)
  __main__.py     — `python -m finetune_studio.cli` entry
  _parser.py      — argparse setup: defines all subcommands + their args
  _registry.py    — maps subcommand name → handler function
  _vram_print.py  — pretty-printer used by `vram check`
  commands/       — one file per subcommand handler (cmd_<name>)
"""
from finetune_studio.cli._registry import COMMANDS, main


def cli_main() -> None:
    """Alias for `main()` — used by `python -m finetune_studio.cli`."""
    main()
