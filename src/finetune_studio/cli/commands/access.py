"""`fts auth ...` and `fts server ...`: who may reach this instance. Works with the app running or stopped.

Passwords are read from a prompt or a file, never from the command line (it lands in shell history and `ps`).
"""
from __future__ import annotations

import getpass
import sys
import time
from pathlib import Path

from finetune_studio.access import server, sessions, store


def _read_password(args) -> str:
    if getattr(args, "password_file", None):
        text = Path(args.password_file).read_text(encoding="utf-8")
        return text.rstrip("\r\n")
    if not sys.stdin.isatty():
        return sys.stdin.readline().rstrip("\r\n")
    first = getpass.getpass("New password: ")
    if getpass.getpass("Repeat it: ") != first:
        print("The two entries differ; nothing changed.", file=sys.stderr)
        sys.exit(2)
    return first


def _when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "never"


def cmd_auth(args) -> None:
    action = args.auth_action
    if action == "status":
        try:
            cfg = store.load()
        except RuntimeError as e:
            print(f"auth.json: {e}", file=sys.stderr)
            sys.exit(1)
        addr = server.load()
        print(f"password:   {'on' if cfg.enabled else 'off'} (changed {_when(cfg.password_changed_at)})")
        print(f"reachable:  {server.describe(addr)}, port {addr.port} (fts server show)")
        print(f"sessions:   {sessions.count()} signed-in browser(s)")
        print(f"API tokens: {len(cfg.tokens)}")
        for t in cfg.tokens:
            print(f"  {t.id}  {t.name:<24} created {_when(t.created_at)}  last used {_when(t.last_used_at)}")
        if not addr.loopback_only and not cfg.enabled:
            print("WARNING: reachable from the network without a password. Run: fts auth set-password", file=sys.stderr)
        return
    if action == "set-password":
        try:
            store.set_password(_read_password(args))
        except ValueError as e:
            print(f"Not set: {e}", file=sys.stderr)
            sys.exit(2)
        signed_out = sessions.revoke_all()
        print(f"Password set. Sign-in is now required{f'; {signed_out} browser(s) signed out' if signed_out else ''}.")
        return
    if action == "disable":
        if getattr(args, "force", False):
            store.reset_damaged()
        else:
            try:
                store.disable()
            except RuntimeError as e:
                print(f"{e}\nUse --force to replace the file.", file=sys.stderr)
                sys.exit(1)
        sessions.revoke_all()
        print("Password removed: no sign-in is asked.")
        if not server.load().loopback_only:
            print("WARNING: this instance is reachable from the network; anyone there can now use it.", file=sys.stderr)
        return
    if action == "token":
        if args.token_action == "create":
            tok, plain = store.create_token(args.name)
            print(f"API token {tok.id} ({tok.name}). Shown once, copy it now:\n{plain}")
            print("Use it as:  Authorization: Bearer <token>")
        elif args.token_action == "revoke":
            if not store.revoke_token(args.id):
                print(f"No API token {args.id!r}.", file=sys.stderr)
                sys.exit(1)
            print(f"Revoked {args.id}.")
        else:
            for t in store.load().tokens:
                print(f"{t.id}  {t.name:<24} created {_when(t.created_at)}  last used {_when(t.last_used_at)}")
        return
    if action in ("apply-config", "setup"):
        from finetune_studio.access import install_config
        try:
            if action == "setup":
                done = install_config.interactive()
                want_service = False
            else:
                cfg = install_config.parse(Path(args.file))
                done, want_service = install_config.apply(cfg), cfg.install_service
        except (ValueError, OSError) as e:
            print(f"Access settings not applied: {e}", file=sys.stderr)
            sys.exit(2)
        for line in done or ["access settings unchanged"]:
            print(f"[access] {line}")
        if want_service:
            print("INSTALL_SERVICE=1")              # read by install.sh
        return
    if action == "sessions":
        print(f"Signed out {sessions.revoke_all()} browser session(s).")
        return
    raise SystemExit(f"unknown auth action {action!r}")


def _onoff(v: bool) -> str:
    return "on " if v else "off"


def cmd_server(args) -> None:
    from finetune_studio.access import netinfo, tailscale
    current = server.load()
    if args.server_action == "show":
        # First line "<address>:<port> (<who>)": install-service.sh reads the address and port from it.
        print(f"{current.probe_host()}:{current.port} ({server.describe(current)})")
        lan = ", ".join(f"{a.ip} {a.interface}" for a in netinfo.lan_addresses()) or "no LAN address found"
        tailnet = tailscale.detect()
        ts = (f"{tailnet.ipv4}{' · ' + tailnet.dns_name if tailnet.dns_name else ''}" if tailnet
              else f"not detected{f', last known {current.tailscale_ip}' if current.tailscale_ip else ''}")
        print("  this computer   on  (always)")
        print(f"  local network   {_onoff(current.lan)} ({lan})")
        print(f"  Tailscale       {_onoff(current.tailscale)} ({ts})")
        print(f"  custom          {', '.join(current.custom) or 'none'}")
        print(f"  listens on      {', '.join(current.binds())}  port {current.port}")
        if current.allowed_hosts:
            print(f"  extra host names {', '.join(current.allowed_hosts)}")
        if tailnet and not current.tailscale and not current.lan:
            print("Tailscale detected: fts server set --tailscale on")
        return
    tailnet = tailscale.detect() if args.tailscale == "on" else None
    if args.tailscale == "on" and tailnet is None and not current.tailscale_ip:
        print("Not saved: Tailscale is not detected on this computer (not installed, logged out or down).",
              file=sys.stderr)
        sys.exit(2)

    def switch(v: str | None) -> bool | None:
        return None if v is None else v == "on"

    def csv(v: str | None) -> list[str] | None:
        return None if v is None else [x for x in v.split(",") if x.strip()]

    try:
        base = current
        if args.host:                                   # the pre-switches single address
            legacy = server.from_host(args.host, args.port or current.port)
            base = server.change(current, lan=legacy.lan, tailscale=legacy.tailscale, custom=legacy.custom)
        addr = server.change(base, port=args.port, lan=switch(args.lan), tailscale=switch(args.tailscale),
                             custom=csv(args.custom), allowed_hosts=csv(args.allowed_hosts), tailnet=tailnet)
    except ValueError as e:
        print(f"Not saved: {e}", file=sys.stderr)
        sys.exit(2)
    if not addr.loopback_only and not store.load().enabled and not args.yes:
        print(f"Reachable from {server.describe(addr)} and no password is set.\n"
              "Set one first (fts auth set-password) or repeat with --yes.", file=sys.stderr)
        sys.exit(2)
    server.save(addr)
    print(f"Saved: {server.describe(addr)}, port {addr.port} (listens on {', '.join(addr.binds())}). "
          "Switching something off applies at once; on, or a new port, after: fts restart web")
