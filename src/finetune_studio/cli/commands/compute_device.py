"""Show or save the compute device used by the app at its next start."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from finetune_studio.accel import saved_choice
from finetune_studio.webui import compute_device as device_settings


class ComputeDeviceCommandError(Exception):
    """A user-facing compute-device CLI failure."""


def _print_report(report: dict[str, Any]) -> None:
    saved = report["saved"]
    selected = saved["name"] if saved["mode"] == "gpu" else saved["mode"].upper()
    print(f"Saved choice: {selected}")
    if saved["mode"] == "gpu":
        print(f"  device id: {saved['device_id']} · {'available' if saved['available'] else 'not detected'}")

    effective = report["effective"]
    print(f"Effective now: {effective['device']} — {effective['name']} ({effective['runtime']})")
    if report["devices"]:
        print("Detected GPUs:")
        for device in report["devices"]:
            flags = []
            if device["active"]:
                flags.append("active")
            if not device["visible"]:
                flags.append("hidden from this process")
            suffix = f" · {', '.join(flags)}" if flags else ""
            print(f"  {device['id']}: {device['name']} · {device['free_gb']:.1f}/{device['total_gb']:.1f} GiB free{suffix}")
    else:
        print("Detected GPUs: none")

    if report["env_overrides"]:
        print("Environment overrides: " + ", ".join(report["env_overrides"]))
        if report["overridden_by_env"]:
            print("The environment overrides the saved choice; remove that override for the saved choice to apply.")
    if report["restart_required"]:
        print(f"Restart required to apply: {report['restart_command']}")


def cmd_compute_device(args: argparse.Namespace) -> None:
    """Display the compute-device report or persist a validated choice."""
    try:
        devices = device_settings.list_devices()
        if args.compute_device_action == "show":
            report = device_settings.report(saved_choice.load(), devices)
        else:
            choice = device_settings.build_choice(
                device_settings.ComputeDeviceIn(mode=args.mode, device_id=args.device_id), devices,
            )
            saved_choice.save(choice)
            report = device_settings.report(choice, devices)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
