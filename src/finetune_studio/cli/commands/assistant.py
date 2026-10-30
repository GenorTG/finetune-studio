"""Manage the Settings → AI assistant connections and per-job routing from the CLI."""
from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

import requests
from fastapi import HTTPException

from finetune_studio.models import assistant
from finetune_studio.models.manager import get_manager
from finetune_studio.webui.routes import assistant_settings


class AssistantCommandError(Exception):
    """A user-facing assistant-settings CLI failure."""


def _emit(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


def _api_connections() -> list[dict[str, Any]]:
    rows = []
    for row in get_manager().list_providers():
        if row.get("kind") == "openai_compat":
            extra = row.get("extra") or {}
            rows.append({
                "id": row["id"],
                "name": row.get("name") or row["id"],
                "model": row.get("model_id") or "",
                "base_url": row.get("base_url") or "",
                "key_set": bool(row.get("api_key_set")),
                "key_state": row.get("api_key_state") or "unset",
                "preset": str(extra.get("preset") or "custom"),
                "reasoning_effort": str((extra.get("body") or {}).get("reasoning_effort") or ""),
                "concurrency": assistant_settings.connection_concurrency(extra),
            })
    return rows


def _settings() -> dict[str, Any]:
    mgr = get_manager()
    main_id = assistant.assistant_provider_id()
    main_row = mgr.get_provider(main_id)
    jobs = assistant.jobs_state()
    return {
        "main": {"id": main_id, "label": assistant.connection_label(main_row)},
        "connections": _api_connections(),
        "jobs": jobs,
        "vision": next((job for job in jobs if job["id"] == "vision"), {}),
    }


def _show(args: argparse.Namespace) -> None:
    state = _settings()
    if args.json:
        _emit(state, True)
        return
    print(f"Main assistant: {state['main']['label']} [{state['main']['id']}]")
    if state["connections"]:
        print("API connections:")
        for row in state["connections"]:
            key = "key saved" if row["key_set"] else f"key {row['key_state']}"
            print(f"  {row['id']}: {row['name']} · {row['model']} · {key}")
    else:
        print("No API connections.")
    print("Jobs:")
    for row in state["jobs"]:
        print(f"  {row['id']}: {row['provider_label'] or row['setting'] or 'main assistant'}")
        if row.get("instructions"):
            print(f"    instructions: {row['instructions']}")


def _main(args: argparse.Namespace) -> None:
    mgr = get_manager()
    row = mgr.get_provider(args.connection)
    if row is None:
        raise AssistantCommandError(f"unknown connection {args.connection!r}")
    if row.get("kind") == "openai_compat" and not row.get("api_key_set"):
        raise AssistantCommandError(f"{row.get('name') or args.connection} has no saved API key yet")
    old_id = assistant.assistant_provider_id()
    assistant.set_assistant_provider_id(args.connection)
    if old_id != args.connection and (mgr.active() or {}).get("id") == old_id:
        from finetune_studio.models.llama_loader import unload_all_models

        unload_all_models()
    _emit({"ok": True, "main": {"id": args.connection, "label": assistant.connection_label(row)}}, args.json)
    if not args.json:
        print(f"Main assistant: {assistant.connection_label(row)} [{args.connection}]")


def _jobs(args: argparse.Namespace) -> None:
    if args.jobs_action == "list":
        rows = assistant.jobs_state()
        if args.json:
            _emit({"jobs": rows}, True)
            return
        for row in rows:
            print(f"{row['id']:<12} {row['provider_label'] or row['setting'] or 'main assistant'}")
            if row.get("instructions"):
                print(f"  Instructions: {row['instructions']}")
        return

    if args.connection is None and args.instructions is None:
        raise AssistantCommandError("set --connection, --instructions, or both")
    if args.local_model and (args.job_id != "vision" or args.connection != "local-file"):
        raise AssistantCommandError("--local-model is only valid for vision with --connection local-file")
    provider_id = args.connection
    if provider_id == "main":
        provider_id = ""
    try:
        assistant.set_job(args.job_id, provider_id=provider_id, instructions=args.instructions,
                          local_model=args.local_model or "")
    except ValueError as exc:
        raise AssistantCommandError(str(exc)) from exc
    row = next(job for job in assistant.jobs_state() if job["id"] == args.job_id)
    _emit({"ok": True, "job": row}, args.json)
    if not args.json:
        print(f"{row['label']}: {row['provider_label'] or row['setting'] or 'main assistant'}")


def _read_key(args: argparse.Namespace, *, required: bool) -> str:
    if args.key_file:
        path = Path(args.key_file).expanduser()
        info = path.stat()
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise AssistantCommandError(f"key file {path} must be owned by this user and private (chmod 600)")
        return path.read_text(encoding="utf-8").strip()
    if args.key_stdin:
        return sys.stdin.read().strip()
    if getattr(args, "prompt_key", False) or (required and sys.stdin.isatty()):
        return getpass.getpass("API key (input hidden): ").strip()
    if required:
        raise AssistantCommandError("provide a key with --key-file, --key-stdin, or an interactive terminal")
    return ""


def _connection_body(args: argparse.Namespace, *, create: bool = False) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for field in ("name", "base_url", "model", "preset", "reasoning_effort", "concurrency"):
        value = getattr(args, field, None)
        if value is not None:
            body[field] = value
    if hasattr(args, "key_file"):
        key = _read_key(args, required=create)
        if key:
            body["api_key"] = key
    if getattr(args, "clear_key", False):
        body["clear_key"] = True
    return body


def _connections_list(args: argparse.Namespace) -> None:
    rows = _api_connections()
    if args.json:
        _emit({"connections": rows}, True)
        return
    if not rows:
        print("No API connections.")
        return
    print(f"{'ID':<28} {'NAME':<24} {'MODEL':<28} KEY")
    print("-" * 92)
    for row in rows:
        print(f"{row['id']:<28} {row['name'][:24]:<24} {row['model'][:28]:<28} "
              f"{'saved' if row['key_set'] else row['key_state']}")


def _connection_upsert(args: argparse.Namespace, *, create: bool) -> None:
    body = _connection_body(args, create=create)
    try:
        pid = assistant_settings.upsert_connection(None if create else args.connection, body)
    except (HTTPException, ValueError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        raise AssistantCommandError(str(detail)) from exc
    row = next((item for item in _api_connections() if item["id"] == pid), None)
    result = {"ok": True, "connection": row}
    _emit(result, args.json)
    if not args.json:
        print(f"{'Added' if create else 'Updated'} connection {pid}; API key is stored encrypted and is never printed.")


def _delete(args: argparse.Namespace) -> None:
    mgr = get_manager()
    row = mgr.get_provider(args.connection)
    if row is None:
        raise AssistantCommandError(f"unknown connection {args.connection!r}")
    if not args.yes:
        if not sys.stdin.isatty():
            raise AssistantCommandError("deleting a connection without a terminal requires --yes")
        answer = input(f"Delete API connection {row.get('name') or args.connection!r}? [y/N] ")
        if answer.strip().casefold() not in {"y", "yes"}:
            _emit({"ok": False, "cancelled": True}, args.json)
            if not args.json:
                print("Cancelled.")
            return
    try:
        assistant_settings.delete_connection_record(args.connection)
    except HTTPException as exc:
        raise AssistantCommandError(str(exc.detail)) from exc
    _emit({"ok": True, "deleted": args.connection}, args.json)
    if not args.json:
        print(f"Deleted {args.connection}; jobs using it returned to the main assistant.")


def _probe_body(args: argparse.Namespace) -> dict[str, Any]:
    body = _connection_body(args)
    if getattr(args, "connection", None):
        body["id"] = args.connection
    return body


def _test(args: argparse.Namespace) -> None:
    try:
        cfg = assistant_settings.probe_connection_config(_probe_body(args))
    except (HTTPException, ValueError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        raise AssistantCommandError(str(detail)) from exc
    if not cfg.api_key or not cfg.model_id:
        raise AssistantCommandError("enter a model name and API key first")
    from finetune_studio.models.providers import OpenAICompatProvider, ProviderError

    provider = OpenAICompatProvider(cfg)
    started = time.perf_counter()
    try:
        reply = provider.chat([{"role": "user", "content": "Reply with exactly: pong"}],
                              max_tokens=512, temperature=0)
    except ProviderError as exc:
        raise AssistantCommandError(str(exc)) from exc
    finally:
        provider.unload()
    result = {"ok": True, "reply": reply.strip()[:80],
              "latency_ms": int((time.perf_counter() - started) * 1000)}
    _emit(result, args.json)
    if not args.json:
        print(f"Connection answered: {result['reply']} · {result['latency_ms']} ms")


def _models(args: argparse.Namespace) -> None:
    try:
        cfg = assistant_settings.probe_connection_config(_probe_body(args))
    except (HTTPException, ValueError) as exc:
        detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
        raise AssistantCommandError(str(detail)) from exc
    if not cfg.api_key:
        raise AssistantCommandError("enter an API key first")
    try:
        response = requests.get(cfg.base_url + "/models",
                                headers={"Authorization": f"Bearer {cfg.api_key}"}, timeout=20)
    except requests.RequestException as exc:
        raise AssistantCommandError(f"{type(exc).__name__} calling {cfg.base_url}/models") from exc
    if response.status_code >= 400:
        detail = " ".join(response.text.split())[:200]
        raise AssistantCommandError(f"provider answered HTTP {response.status_code}: {detail}")
    try:
        data = response.json().get("data") or []
        models = sorted(str(row["id"]) for row in data if isinstance(row, dict) and row.get("id"))
    except (ValueError, AttributeError) as exc:
        raise AssistantCommandError("provider /models response was not the OpenAI list shape") from exc
    _emit({"models": models}, args.json)
    if not args.json:
        print("\n".join(models) if models else "Provider returned no models.")


def cmd_assistant(args: argparse.Namespace) -> None:
    """Manage AI assistant connections and job assignments."""
    from finetune_studio import db

    db.init_db()
    try:
        if args.assistant_action == "connections":
            actions = {
                "list": _connections_list,
                "add": lambda value: _connection_upsert(value, create=True),
                "update": lambda value: _connection_upsert(value, create=False),
                "delete": _delete,
            }
            actions[args.connections_action](args)
        elif args.assistant_action == "jobs":
            _jobs(args)
        else:
            actions = {"show": _show, "main": _main, "test": _test, "models": _models}
            actions[args.assistant_action](args)
    except AssistantCommandError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
