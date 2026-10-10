"""Supervisor: the small process that owns the app's components.

It spawns child processes (today the web UI; next the inference worker, trainers, gateway),
keeps a desired-state table and a process table, probes health, restarts with backoff,
records an event log, and serves a control API on a unix socket. It never imports torch,
llama.cpp or the web app: a crash in any child must not be able to take it down.
Architecture and decisions: ``docs/TOPOLOGY.md`` (local dev doc).
"""
