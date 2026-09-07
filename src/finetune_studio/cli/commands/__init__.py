"""Per-subcommand handlers. One file per command — each exports `cmd_<name>(args)`.

Adding a new subcommand:
  1. Add the parser in `_parser.py`
  2. Add `cmd_<name>` in a new file in this directory
  3. Register it in `_registry.py`
"""
