"""`fts jobs [list|stop ID|logs ID]`: long jobs (training, export, ...) running as their own processes."""
from __future__ import annotations

import sys
import time

from finetune_studio.jobs import launch, store


def _age(ts: float | None) -> str:
    if not ts:
        return "-"
    s = int(time.time() - ts)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def cmd_jobs(args) -> None:
    action = getattr(args, "jobs_action", None) or "list"
    if action == "list":
        jobs = launch.list_jobs(running_only=not args.all)
        if not jobs:
            print("No running jobs." if not args.all else "No jobs.")
            return
        print(f"{'JOB':<38}{'KIND':<11}{'STATE':<10}{'AGE':<9}TITLE")
        for j in jobs:
            state = "running" if j["running"] else (j.get("exit") or {}).get("status", "?")
            print(f"{j['id']:<38}{j['kind']:<11}{state:<10}{_age(j.get('created_at')):<9}{j.get('title') or ''}")
        return
    try:
        if action == "stop":
            d = launch.stop(args.id)
            print(f"Stop requested for {args.id}" if d["running"] else f"{args.id} is not running")
        elif action == "logs":
            path = store.job_dir(args.id)
            out = path / "output.log"
            if out.exists():                     # started without a supervisor
                print("\n".join(out.read_text(encoding="utf-8", errors="replace").splitlines()[-args.lines:]))
            else:
                from finetune_studio.supervisor.client import SupervisorClient
                print("\n".join(SupervisorClient().job_logs(args.id, args.lines)))
            print(f"(full log: logs/job-{store.read_spec(args.id).kind}.log under $FTS_ROOT)")
    except (FileNotFoundError, ValueError, launch.JobRejected) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
