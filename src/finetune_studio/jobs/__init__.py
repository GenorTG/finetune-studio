"""Long jobs (training, export, mining, index builds, test runs) as their own processes, owned by the supervisor.

Before 2026-10-15 these ran inside the web process (threads or ``daemon=True`` children): a web restart or crash
killed them mid-run. Now:

- the web app (or the CLI) writes a job spec and asks ``jobs.launch.start`` to run it; with a supervisor the
  supervisor spawns ``python -m finetune_studio.jobs.runner --id <id>`` as its own child (survives web restarts,
  GPU lease: one GPU job at a time, the inference worker's model is unloaded first); without one (``fts webui``)
  the runner is started detached in its own session, which also outlives the web process;
- the runner dispatches to the kind's ``run(ctx)``, which persists durable results itself (DB rows) and publishes
  a live state dict (``ctx.publish``) that pages read through ``jobs.store.read_state``;
- stop = SIGTERM to the job's process group -> ``ctx.stop`` is set -> the kind finishes cooperatively;
  SIGKILL after the job's grace period;
- every job leaves ``exit.json`` (written by the runner, or by the supervisor/launcher when the process died hard).

Layout: ``$FTS_ROOT/run/jobs/<id>/{spec.json, state.json, exit.json, pid, data*}``.
"""
from __future__ import annotations
