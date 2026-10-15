"""Inference as a separate process (supervisor P2).

The model lives in ONE worker process (``inference/worker.py``) that serves a small HTTP API on a
unix socket; the web UI and the CLI talk to it through :class:`RemoteEngine`, which has the same
interface as ``testing.inference.InferenceEngine``. A CUDA/llama.cpp abort now kills the worker, not
the web server: the proxy raises :class:`EngineCrashed`, the supervisor restarts an EMPTY worker and
the model stays unloaded until someone loads it again (Genor 2026-10-10: report, never auto-reload).

``get_engine()`` is the one factory: a running worker is used (shared), else one is started for this
process (standalone web server, or a CLI command) and stopped with it.
"""
from __future__ import annotations

from finetune_studio.inference.remote import (
    EngineCrashed,
    RemoteEngine,
    WorkerUnavailable,
)


def get_engine() -> RemoteEngine:
    return RemoteEngine()


def describe_worker(engine: RemoteEngine) -> str:
    """One line for CLI output: which process holds the model."""
    if not hasattr(engine, "worker_info"):
        return "an in-process engine"
    info = engine.worker_info()
    where = {"supervised": "the service's inference worker", "shared": "a running inference worker",
             "private": "a private inference worker for this command"}.get(info["transport"], "no inference worker")
    return f"{where} (pid {info['pid']})" if info.get("pid") else where


__all__ = ["EngineCrashed", "RemoteEngine", "WorkerUnavailable", "describe_worker", "get_engine"]
