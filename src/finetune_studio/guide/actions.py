"""Actions the Guide can prepare and run for the user.

Each action tool checks its arguments against the real project (the files, datasets, suites and presets exist), then
returns a ``ui_event`` of type ``action`` carrying the exact request the page's own button sends: method, URL and body.
The browser runs it under the user's own session, so an action goes through the same endpoint, checks and limits as a
click; the server never fires it by itself.

How it runs depends on the Guide's action mode (Settings → AI assistant / the Guide panel; ``settings.guide_action_mode``):

- ``ask`` (default): the panel shows a card with **Run** / **Skip**; nothing happens until the user presses Run.
- ``auto`` (hands-free): the panel runs the action at once and hands the result back to the Guide, which may continue
  with the next step (a few actions per message at most).

Nothing here deletes, approves pairs, publishes or exports a model: those stay on the pages.
"""
from __future__ import annotations

import uuid
from typing import Any

ACTION_TOOLS = ("start_mining", "build_dataset", "build_smart_brain", "start_training", "build_reworded_quiz",
                "run_test", "pin_version")
MAX_AUTO_ACTIONS = 4        # hands-free: actions in a row before the Guide stops and reports (browser-enforced)


def _obj(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties or {}, "required": required or []}


CATALOG: list[dict[str, Any]] = [
    {
        "name": "start_mining",
        "description": "Action: queue mining jobs (the assistant writes Q&A pairs for review) for parsed files. Omit source_ids for every parsed file. Runs after the user confirms, or at once in hands-free mode.",
        "parameters": _obj({"source_ids": {"type": "array", "items": {"type": "string"}},
                            "kind": {"type": "string", "enum": ["mine", "fill"], "description": "mine = new pairs; fill = only the facts still missing"}}),
    },
    {
        "name": "build_dataset",
        "description": "Action: build and register the training dataset from the approved pairs (refused while a file is incomplete unless force).",
        "parameters": _obj({"force": {"type": "boolean"}}),
    },
    {
        "name": "build_smart_brain",
        "description": "Action: build the Smart Brain (retrieval index) from every readable file of the project.",
        "parameters": _obj(),
    },
    {
        "name": "start_training",
        "description": "Action: prepare a training run for confirmation. Omit base_model to use the project's selected base; if overriding, pass its exact local path or Hub ID, not a display-name fragment. Use tier for the Training page's adaptive effort preset (smoke, balanced, precision, overkill); the UI label 'Baseline' means balanced. The legacy preset argument still accepts fixed backend preset IDs.",
        "parameters": _obj({"dataset_id": {"type": "string"}, "base_model": {"type": "string"},
                            "tier": {"type": "string", "enum": ["smoke", "balanced", "precision", "overkill"]},
                            "preset": {"type": "string", "description": "Legacy fixed preset ID, or 'baseline' as an alias for tier=balanced."},
                            "num_epochs": {"type": "integer", "minimum": 1, "maximum": 100}}),
    },
    {
        "name": "build_reworded_quiz",
        "description": "Action: build a reworded held-out quiz from a dataset (default: the newest).",
        "parameters": _obj({"dataset_id": {"type": "string"}}),
    },
    {
        "name": "run_test",
        "description": "Action: run a test suite (name from list_suites) on a model: 'latest' = the newest exported model (default), 'base' = the untrained base.",
        "parameters": _obj({"suite": {"type": "string"}, "model": {"type": "string", "enum": ["latest", "base"]}},
                           ["suite"]),
    },
    {
        "name": "pin_version",
        "description": "Action: pin a version of the project (datasets, runs, Smart Brain build), building on the newest version.",
        "parameters": _obj({"label": {"type": "string"}}),
    },
]

LIST_SUITES_TOOL = {
    "name": "list_suites",
    "description": "Test suites available in this project (name, title, cases): pass a name to run_test.",
    "parameters": _obj(),
}


class ActionError(ValueError):
    """The action cannot be prepared (message goes back to the model)."""


def _mode() -> str:
    from finetune_studio.webui.routes.settings import get_guide_action_mode

    return get_guide_action_mode()


def _project(pid: str) -> dict[str, Any]:
    from finetune_studio import db

    project = db.get_project(pid) if pid else None
    if not project:
        raise ActionError("no active project; open a project first")
    return project


def _dataset(pid: str, dataset_id: str) -> dict[str, Any]:
    from finetune_studio import db

    datasets = db.list_datasets(pid)
    if not datasets:
        raise ActionError("this project has no dataset yet: build one first (build_dataset)")
    if not dataset_id:
        return max(datasets, key=lambda d: float(d.get("created_at") or 0))
    for d in datasets:
        if d["id"] == dataset_id:
            return d
    raise ActionError(f"unknown dataset {dataset_id!r}; datasets: {[d['id'] + ' ' + d['name'] for d in datasets]}")


def _same_model_ref(requested: str, configured: str) -> bool:
    """Match a model display/repo name to the project's configured local model path."""
    left = requested.strip().replace("\\", "/").rstrip("/")
    right = configured.strip().replace("\\", "/").rstrip("/")
    if left.casefold() == right.casefold():
        return True
    requested_leaf = left.rsplit("/", 1)[-1]
    configured_leaf = right.rsplit("/", 1)[-1]
    if "__" in configured_leaf:  # HF-cache directories encode Org/Repo as Org__Repo.
        configured_repo = configured_leaf.replace("__", "/", 1)
        if left.count("/") == 1 and left.casefold() == configured_repo.casefold():
            return True
        configured_leaf = configured_leaf.split("__", 1)[-1]
    # The Guide sometimes emits just the displayed repo leaf. Resolve that only
    # against the project's selected model; do not turn it into a Hub lookup.
    return "/" not in left and requested_leaf.casefold() == configured_leaf.casefold()


def list_suites(pid: str) -> dict[str, Any]:
    from finetune_studio.benchmarks.suite_defs import discover_suites

    _project(pid)
    return {"suites": [{"name": s.get("name"), "title": s.get("title"), "cases": s.get("case_count")}
                       for s in discover_suites(pid)]}


def _suite(pid: str, name: str) -> dict[str, Any]:
    from finetune_studio.benchmarks.suite_defs import discover_suites

    suites = discover_suites(pid)
    for s in suites:
        if name in (s.get("name"), s.get("path")):
            return s
    raise ActionError(f"unknown suite {name!r}; call list_suites")


def _prepare(pid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """{label, detail, method, url, body} for one action; ``ActionError`` with the reason when it cannot run."""
    from finetune_studio import db

    project = _project(pid)
    base = f"/api/projects/{pid}"
    if name == "start_mining":
        from finetune_studio.knowledge.sources import list_sources as sources

        ready = {s.id: s for s in sources(pid) if s.ready}
        if not ready:
            raise ActionError("no parsed file to mine: upload files and wait for parsing (Files page)")
        ids = [str(i) for i in (args.get("source_ids") or [])] or list(ready)
        unknown = [i for i in ids if i not in ready]
        if unknown:
            raise ActionError(f"not parsed or unknown file id(s) {unknown[:5]}; parsed: "
                              f"{[(s.id, s.filename) for s in ready.values()][:20]}")
        kind = str(args.get("kind") or "mine")
        if kind not in ("mine", "fill"):
            raise ActionError("kind must be mine or fill")
        names = ", ".join(ready[i].filename for i in ids[:5]) + (f" and {len(ids) - 5} more" if len(ids) > 5 else "")
        return {"label": f"{'Mine' if kind == 'mine' else 'Fill missing facts in'} {len(ids)} file(s)",
                "detail": f"{names}. Runs on the assistant's mining job; pairs land pending review.",
                "method": "POST", "url": f"{base}/mining/jobs", "body": {"source_ids": ids, "kind": kind}}
    if name == "build_dataset":
        force = bool(args.get("force"))
        return {"label": "Build the training dataset" + (" (force)" if force else ""),
                "detail": "Exports the approved pairs and registers the dataset."
                          + (" Incomplete files are included as they are." if force else ""),
                "method": "GET", "url": f"{base}/data-prep/export" + ("?force=true" if force else ""), "body": None}
    if name == "build_smart_brain":
        return {"label": "Build the Smart Brain", "detail": "Parses every readable file not parsed yet, then builds "
                "the retrieval index (a background build).", "method": "POST", "url": f"{base}/brain/quick",
                "body": {}}
    if name == "start_training":
        import finetune_studio.webui.app  # noqa: F401 - loads the route modules in the app's own order (no cycle)
        from finetune_studio.training.preset_advisor import propose
        from finetune_studio.webui.routes.training import TRAINING_PRESETS

        ds = _dataset(pid, str(args.get("dataset_id") or ""))
        project_model = str(project.get("base_model") or "").strip()
        requested_model = str(args.get("base_model") or "").strip()
        model = (project_model if requested_model and project_model
                 and _same_model_ref(requested_model, project_model) else requested_model or project_model)
        if not model:
            raise ActionError("no base model: pass base_model, or set the project's base model (project Settings)")
        tier = str(args.get("tier") or "").strip().lower()
        preset = str(args.get("preset") or "").strip().lower()
        if tier == "baseline":
            tier = "balanced"
        if not tier and preset in {"baseline", "smoke", "balanced", "precision", "overkill"}:
            tier = "balanced" if preset == "baseline" else preset

        if tier:
            try:
                advice = propose(tier=tier, base_model_ref=model, dataset_path=ds.get("data_path"))
            except ValueError as exc:
                raise ActionError(str(exc)) from exc
            epochs = advice.num_epochs
            if args.get("num_epochs") not in (None, ""):
                epochs = int(args["num_epochs"])
                if not 1 <= epochs <= 100:
                    raise ActionError("num_epochs must be 1-100")
            steps = advice.pair_count * epochs // max(advice.effective_batch, 1)
            body = {
                "project_id": pid, "dataset_id": ds["id"], "model_path": model,
                "training_mode": "sft", "output_dir": "output", "num_epochs": epochs,
                "lora_rank": advice.lora_rank, "lora_alpha": advice.lora_alpha,
                "learning_rate": advice.learning_rate, "batch_size": advice.batch_size,
                "gradient_accumulation_steps": advice.gradient_accumulation_steps,
                "max_seq_length": 2048, "warmup_steps": advice.warmup_steps,
                "merge_on_save": True, "unsloth": False, "save_checkpoints": True,
                "system_prompt_mode": "bake",
            }
            warning = f" Warnings: {'; '.join(advice.warnings)}" if advice.warnings else ""
            return {
                "label": f"Train {model.rstrip('/').rsplit('/', 1)[-1]} on {ds['name']}",
                "detail": (f"Adaptive {tier} tier (the page's Baseline is balanced): {epochs} epochs, "
                           f"rank {advice.lora_rank}, alpha {advice.lora_alpha}, learning rate "
                           f"{advice.learning_rate}, batch size {advice.batch_size}, gradient accumulation "
                           f"{advice.gradient_accumulation_steps}, warmup {advice.warmup_steps} steps, "
                           f"max sequence length 2048; about {steps} optimizer steps.{warning} "
                           "The final model is merged when training finishes. "
                           "Uses the GPU until it ends; Training page shows progress."),
                "method": "POST", "url": "/api/training/start", "body": body,
            }

        preset = preset or "standard"
        if preset not in TRAINING_PRESETS:
            raise ActionError(f"unknown preset {preset!r}; presets: {sorted(TRAINING_PRESETS)}")
        overrides: dict[str, Any] = {}
        if args.get("num_epochs") not in (None, ""):
            epochs = int(args["num_epochs"])
            if not 1 <= epochs <= 100:
                raise ActionError("num_epochs must be 1-100")
            overrides["num_epochs"] = epochs
        return {"label": f"Train {model.rstrip('/').rsplit('/', 1)[-1]} on {ds['name']}",
                "detail": f"Preset {preset}" + (f", {overrides['num_epochs']} epochs" if overrides else "")
                          + ". Uses the GPU until it ends; Training page shows progress.",
                "method": "POST", "url": "/api/training/start",
                "body": {"project_id": pid, "dataset_id": ds["id"], "model_path": model, "preset_id": preset,
                         "overrides": overrides, "training_mode": "sft"}}
    if name == "build_reworded_quiz":
        ds = _dataset(pid, str(args.get("dataset_id") or ""))
        return {"label": f"Build a reworded quiz from {ds['name']}",
                "detail": "The assistant rewords the held-out questions (the quiz job).",
                "method": "POST", "url": f"{base}/testing/reworded-quiz", "body": {"dataset_id": ds["id"]}}
    if name == "run_test":
        suite = _suite(pid, str(args.get("suite") or ""))
        model = str(args.get("model") or "latest")
        if model not in ("latest", "base"):
            raise ActionError("model must be latest or base")
        body: dict[str, Any] = {"project_id": pid, "suite_path": suite["path"], "max_tokens": 512, "temperature": 0.3}
        if model == "base":
            body["model_path"] = "__base__"
        return {"label": f"Run {suite.get('title') or suite.get('name')} on the {'untrained base' if model == 'base' else 'latest exported model'}",
                "detail": f"{suite.get('case_count') or '?'} questions; answers are saved, then judged if auto-judge is on.",
                "method": "POST", "url": "/api/testing/run-suite", "body": body}
    if name == "pin_version":
        versions = db.list_versions(pid)
        body = {"label": str(args.get("label") or "")[:80] or "pinned-by-guide"}
        if versions:
            body["parent_version_id"] = versions[0]["id"]
        return {"label": f"Pin version {body['label']}",
                "detail": ("Builds on v" + str(versions[0]["version_number"])) if versions else "The first version.",
                "method": "POST", "url": f"{base}/versions", "body": body}
    raise ActionError(f"unknown action {name!r}")


def prepare_action(pid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """The tool result for an action tool: what the model is told, plus the ``ui_event`` the panel runs."""
    try:
        spec = _prepare(pid, name, args if isinstance(args, dict) else {})
    except ActionError as exc:
        return {"error": str(exc)}
    mode = _mode()
    event = {"type": "action", "id": uuid.uuid4().hex[:12], "action": name, "mode": mode, **spec}
    note = ("The user sees a card with Run / Skip; nothing has happened yet." if mode == "ask"
            else "Hands-free mode: the panel runs it now.")
    return {"ok": True, "effect": "action", "status": "awaiting_confirmation" if mode == "ask" else "running",
            "action": name, "label": spec["label"], "mode": mode,
            "note": note + " The outcome comes back as the next message (ACTION_RESULT / ACTION_SKIPPED). Do not call "
                           "this action again in this reply; end your reply with one short sentence.",
            "ui_event": event}
