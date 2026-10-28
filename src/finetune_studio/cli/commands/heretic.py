"""``fts heretic`` — the optional Heretic tool (removes a model's refusals).

Heretic is a separate program (AGPL-3.0) that Finetune Studio installs into
``$FTS_ROOT/heretic`` and drives headlessly; it is never vendored into this
codebase. This command installs, updates and inspects it — the same three
actions the Settings page offers, for scripts and a terminal.

Exit status: 0 when the tool is installed and can run unattended, 1 otherwise
(so a script can gate an export on ``fts heretic status``).
"""
from __future__ import annotations

import json
import sys


def _print_status(d: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(d, indent=2))
        return
    lines = [
        f"Heretic     : {'installed' if d['installed'] else 'not installed'}",
        f"Version     : {d['version'] or '—'}{(' @ ' + d['ref'][:7]) if d['ref'] else ''}",
        f"Pinned      : {d['pinned']}",
        f"Environment : {d['python'] or '—'}",
        "Unattended  : " + ("yes" if d["installed"] and not d["missing_flags"]
                            else ("missing " + ", ".join(d["missing_flags"]) if d["missing_flags"] else "n/a")),
    ]
    if d["hint"]:
        lines.append(f"Note        : {d['hint']}")
    print("\n".join(lines))


def cmd_heretic(args) -> None:
    from finetune_studio.training import heretic_tool

    action = getattr(args, "heretic_action", "status") or "status"
    as_json = bool(getattr(args, "json", False))

    if action == "status":
        d = heretic_tool.probe().as_dict()
        _print_status(d, as_json=as_json)
        if not d["usable"]:
            sys.exit(1)
        return

    if action == "check":
        status = heretic_tool.probe()
        upstream = heretic_tool.check_upstream()
        payload = {
            **status.as_dict(),
            **upstream.as_dict(),
            "update_available": heretic_tool.update_available(status, upstream),
        }
        if as_json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"Installed  : {status.version or '—'}"
                  f"{(' @ ' + status.ref[:7]) if status.ref else ''}")
            print(f"Newest tag : {upstream.latest_release or '—'} ({upstream.latest_release_at[:10]})")
            print(f"Newest code: {upstream.head_commit[:7] or '—'} ({upstream.head_at[:10]})"
                  f" {upstream.head_message}")
            if upstream.error:
                print(f"Note       : {upstream.error}")
            print("Update     : " + ("available — fts heretic install" if payload["update_available"]
                                    else "already current"))
        if upstream.error:
            sys.exit(1)
        return

    if action in ("install", "update"):
        ref = (getattr(args, "ref", "") or "").strip() or heretic_tool.PINNED_REF
        print(f"Installing Heretic ({ref[:7]}) into {heretic_tool.root()} …")
        try:
            status = heretic_tool.install(ref, on_line=lambda line: print(f"  {line}"))
        except heretic_tool.HereticError as e:
            print(f"Failed: {e}", file=sys.stderr)
            sys.exit(1)
        _print_status(status.as_dict(), as_json=as_json)
        return

    if action == "uninstall":
        if not getattr(args, "yes", False):
            print("Refusing to remove the Heretic environment without --yes.", file=sys.stderr)
            sys.exit(2)
        heretic_tool.uninstall()
        print(f"Removed {heretic_tool.root()}")
        return

    print(f"Unknown action: {action}", file=sys.stderr)
    sys.exit(2)


def add_parser(sub) -> None:
    """Attach ``fts heretic …`` to the root parser."""
    p = sub.add_parser(
        "heretic",
        help="Optional Heretic tool: removes a model's refusals (install, update, status)",
        description=(
            "Heretic is an optional external program (AGPL-3.0) that removes a model's "
            "refusal behaviour by searching abliteration parameters. It is installed into "
            "$FTS_ROOT/heretic and reuses this app's PyTorch; it is not part of this codebase."
        ),
    )
    hs = p.add_subparsers(dest="heretic_action")
    p_st = hs.add_parser("status", help="Installed? Can it run unattended? (exit 1 when it cannot)")
    p_st.add_argument("--json", action="store_true")
    p_chk = hs.add_parser("check", help="Compare the installed version with Heretic's GitHub upstream")
    p_chk.add_argument("--json", action="store_true")
    p_in = hs.add_parser("install", help="Install (or re-install) it; reuses this app's PyTorch")
    p_in.add_argument("--ref", default="", help="Git commit or tag (default: the pinned headless build)")
    p_in.add_argument("--json", action="store_true")
    p_up = hs.add_parser("update", help="Same as install: fetch the newest pinned build and rebuild the environment")
    p_up.add_argument("--ref", default="")
    p_up.add_argument("--json", action="store_true")
    p_un = hs.add_parser("uninstall", help="Delete the checkout and its environment (never this app's venv)")
    p_un.add_argument("--yes", action="store_true", help="Actually remove it")
