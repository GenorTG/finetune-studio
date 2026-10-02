"""Regression: ``generate_fixes`` must only suggest commands the CLI really has."""

from __future__ import annotations

import argparse
import shlex

import pytest

from finetune_studio.cli._parser import build_parser
from finetune_studio.training.data_quality import generate_fixes

ISSUE_TYPES = [
    "format", "duplicates", "balance", "length", "language",
    "system_prompt", "empty", "hallucination_risk",
]


def _registered_subcommands() -> set[str]:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("no subparsers registered")


def _all_fixes() -> list[dict]:
    issues = [{"type": t, "severity": "medium", "message": t} for t in ISSUE_TYPES]
    return generate_fixes({"issues": issues})


def test_every_suggested_fts_command_is_a_registered_subcommand() -> None:
    registered = _registered_subcommands()
    commands = [f["command"] for f in _all_fixes() if f.get("command")]
    assert commands, "expected at least one runnable suggestion"
    for command in commands:
        tokens = shlex.split(command)
        assert tokens[0] == "fts", command
        assert tokens[1] in registered, f"`{command}`: unknown subcommand {tokens[1]!r}"


def test_every_suggested_fts_command_parses_with_real_flags() -> None:
    parser = build_parser()
    for fix in _all_fixes():
        if not fix.get("command"):
            continue
        argv = [
            {"INPUT": "in.jsonl", "OUTPUT": "out.jsonl"}.get(t, t)
            for t in shlex.split(fix["command"])[1:]
        ]
        try:
            parser.parse_args(argv)
        except SystemExit as exc:
            pytest.fail(f"`{fix['command']}` rejected by the real parser (exit {exc.code})")


def test_every_fix_without_a_command_carries_plain_advice() -> None:
    for fix in _all_fixes():
        assert fix.get("command") or fix.get("advice"), fix
