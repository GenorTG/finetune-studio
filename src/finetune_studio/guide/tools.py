"""The guide's tool catalog and dispatcher.

WHAT THIS FILE DOES
===================
One place that defines every tool the guide may call and runs it:

* **Knowledge**: ``app_help`` (BM25 over the KB), ``get_app_guide``, ``explain_setting``.
* **Read-only state**: ``project_overview``, ``inspect_project_readiness``, ``list_datasets``,
  ``list_runs``, ``system_status``, ``recommend_training``, ``dataset_health``,
  ``list_sources``, ``read_source``, ``list_qa_pairs``.
* **UI guidance** (visible effect, validated against ``registry``): ``navigate``, ``highlight``,
  ``suggest_settings``. They return a ``ui_event`` that the SSE loop forwards to the browser.
* **The only mutation**: ``create_qa_pairs`` (review-pending rows). Approve / export / train /
  delete stay user-clicked.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from finetune_studio.data.fs import qa as qa_fs
from finetune_studio.data.fs.paths import project_dir
from finetune_studio.guide import registry, settings_help, state
from finetune_studio.guide.kb import table_of_contents
from finetune_studio.guide.search import search_kb

log = logging.getLogger(__name__)

_PAGE_PATH_RES = [
    (re.compile("^" + re.escape(p.url).replace(r"\{pid\}", r"[^/]+") + "$"), key)
    for key, p in registry.PAGES.items()
]
_PROJECT_PATH_RE = re.compile(r"^/projects/([^/]+)(?:/|$)")
_NON_PROJECT_SEGMENTS = {"", "new"}


@dataclass(frozen=True)
class ToolContext:
    """Where the user is: active project (from the URL) and current page key."""

    pid: str | None = None
    page: str | None = None

    @classmethod
    def from_path(cls, path: str | None, pid: str | None = None) -> ToolContext:
        """Derive page key (and project id when not given) from a browser path."""
        clean = (path or "").split("?")[0].split("#")[0].rstrip("/") or "/"
        page = next((key for rx, key in _PAGE_PATH_RES if rx.match(clean)), None)
        match = _PROJECT_PATH_RE.match(clean)
        derived = match.group(1) if match and match.group(1) not in _NON_PROJECT_SEGMENTS else None
        return cls(pid=pid or derived, page=page)


def _obj(properties: dict[str, Any] | None = None, required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": properties or {}, "required": required or []}


_PAGE_KEYS = sorted(registry.PAGES)
_CONTROL_KEYS = sorted(registry.CONTROLS)

TOOLS_CATALOG: list[dict[str, Any]] = [
    {
        "name": "app_help",
        "description": "Search the app knowledge base (every page, control, workflow, route and common mistake). Use FIRST for any 'how do I / what is / where is / which' question about the app; cite the returned entry and page.",
        "parameters": _obj({"query": {"type": "string"}}, ["query"]),
    },
    {
        "name": "get_app_guide",
        "description": "Supported workflow order, page routes, training routes and dataset formats in one short overview, plus the KB table of contents.",
        "parameters": _obj(),
    },
    {
        "name": "project_overview",
        "description": "This project's state: sources, Q&A review counts, datasets, RAG corpora, runs, production run, next step. With no active project, lists projects.",
        "parameters": _obj(),
    },
    {
        "name": "inspect_project_readiness",
        "description": "Authoritative readiness counts with a `summary` line. Use when the user asks what is done or what to do next; quote the summary verbatim.",
        "parameters": _obj(),
    },
    {
        "name": "list_datasets",
        "description": "Registered training datasets of this project (id, name, rows, size).",
        "parameters": _obj(),
    },
    {
        "name": "list_runs",
        "description": "Recent training runs (status, base model, route, loss, steps, production flag).",
        "parameters": _obj(),
    },
    {
        "name": "system_status",
        "description": "GPU and VRAM, accelerator, RAM, loaded model and its placement, helper model state, training state, free disk.",
        "parameters": _obj(),
    },
    {
        "name": "recommend_training",
        "description": "Settings the preset advisor proposes for this project's dataset and base model, with the step arithmetic, too-few-steps flag, alternatives and the right route for the user's goal.",
        "parameters": _obj({
            "goal": {"type": "string", "description": "what the user wants, e.g. 'memorize product facts', 'prefer concise answers', 'tool calling'"},
            "tier": {"type": "string", "enum": ["smoke", "balanced", "precision", "overkill"]},
            "base_model": {"type": "string"},
            "dataset_id": {"type": "string"},
        }),
    },
    {
        "name": "dataset_health",
        "description": "Will a dataset train? Usable rows, held-out rows, duplicates, conflicting or short answers for the chosen route, plus the Q&A pair audit.",
        "parameters": _obj({
            "dataset_id": {"type": "string"},
            "training_mode": {"type": "string", "enum": list(state._ROUTES)},
        }),
    },
    {
        "name": "explain_setting",
        "description": "Meaning, guidance and pitfall for one setting (epochs, lora_rank, learning_rate, batch_size, qa_per_chunk, chunk_size, n_ctx, …).",
        "parameters": _obj({"name": {"type": "string"}}, ["name"]),
    },
    {
        "name": "navigate",
        "description": "Open an app page in the user's browser (allow-listed page keys only). The conversation stays open.",
        "parameters": _obj({"page": {"type": "string", "enum": _PAGE_KEYS}}, ["page"]),
    },
    {
        "name": "highlight",
        "description": "Pulse one control on screen so the user sees where to click (allow-listed control ids). Opens its page first if needed. Does not click anything.",
        "parameters": _obj({"control_id": {"type": "string", "enum": _CONTROL_KEYS}}, ["control_id"]),
    },
    {
        "name": "suggest_settings",
        "description": "Pre-fill form fields on a page with recommended values, visibly and editable. Never submits or starts anything; the user reviews and presses the button.",
        "parameters": _obj({
            "page": {"type": "string", "enum": sorted({p for p, _ in registry.FIELDS})},
            "settings": {"type": "object", "description": "field name → value; allowed fields are listed per page in the error if one is rejected"},
        }, ["page", "settings"]),
    },
    {
        "name": "list_sources",
        "description": "List all parsed source files in this project. Returns each source's id, filename, status, and chunk count.",
        "parameters": _obj(),
    },
    {
        "name": "read_source",
        "description": "Read the parsed text content of a source file by id. Use this AFTER list_sources to see what the file contains before generating Q&A pairs.",
        "parameters": _obj({"source_id": {"type": "string"}}, ["source_id"]),
    },
    {
        "name": "list_qa_pairs",
        "description": "List existing Q&A pairs in this project, optionally filtered by source_id or status (pending/approved/rejected).",
        "parameters": _obj({
            "source_id": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "approved", "rejected"]},
        }),
    },
    {
        "name": "create_qa_pairs",
        "description": "Create new Q&A pairs for a source. Each pair needs a source_id, question, answer, and 1-based parsed chunk_idx. Pairs land in pending status for review. Only when the user asked for generation.",
        "parameters": _obj({
            "source_id": {"type": "string"},
            "pairs": {
                "type": "array",
                "items": _obj({
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "chunk_idx": {"type": "integer", "minimum": 1},
                }, ["question", "answer"]),
            },
        }, ["source_id", "pairs"]),
    },
]
TOOL_NAMES = frozenset(t["name"] for t in TOOLS_CATALOG)


def app_guide() -> dict[str, Any]:
    """Short orientation; ``app_help`` has the detail. Shape kept for existing callers."""
    return {
        "workflow": [
            {"step": 1, "page": "/projects", "action": "Create or choose a project."},
            {"step": 2, "page": "/projects/{pid}/data", "action": "Upload files; inspect parse status and parsed text."},
            {"step": 3, "page": "/projects/{pid}/data-prep", "action": "Generate/review Q&A, then export a dataset."},
            {"step": 4, "page": "/projects/{pid}/training", "action": "Choose SFT, tool-calling SFT, DPO, continued pretraining, or reasoning distillation to match the dataset you actually have."},
            {"step": 5, "page": "/projects/{pid}/testing", "action": "Run the generated project quiz and held-out evaluation before selecting a production run."},
            {"step": 6, "page": "/projects/{pid}/export", "action": "Export a GGUF for local inference when needed."},
        ],
        "routes": {
            "sft": "Use for teaching task format, style, or selected facts from curated prompt/answer examples.",
            "dpo": "Use for preference/alignment behavior when each prompt has a human-reviewed preferred and rejected answer. JSONL rows need prompt, chosen, rejected; standard strings or conversational role/content lists are accepted.",
            "tool_sft": "Use for agent behavior when traces include assistant tool_calls, tool replies, and a tools JSON-schema list; the model must have a compatible chat template.",
            "continued_pretraining": "Use for domain adaptation on raw text JSONL rows with a non-empty text field. It is not instruction tuning and should be evaluated on held-out domain text and downstream tasks.",
            "reasoning_distillation": "Use for reviewed teacher traces and final answers in conversational messages JSONL. Validate the final task outcome separately; a plausible rationale is not proof of correctness.",
            "rag": "Use when knowledge changes often, must retain source citations, or should be fetched rather than memorized. Build/index a corpus on the RAG page.",
            "not_supported_yet": ["ORPO", "KTO", "in-app preference comparison authoring", "automatic teacher-trace generation"],
        },
        "quality_checks": [
            "Keep train and evaluation examples separate; avoid benchmark contamination.",
            "Review synthetic answers against source text; generated data is not ground truth by default.",
            "For DPO, compare answers to the same prompt and ensure the preference reflects the intended behavior, not just response length.",
            "For tool SFT, inspect tool names/arguments/results and confirm tool schemas match the runtime implementation.",
            "For continued pretraining and reasoning distillation, hold out clean evaluation data and compare against the untuned base model.",
            "Use RAG instead of weight updates for frequently changing or citation-critical facts.",
        ],
        "navigation_note": "The guide can open pages, highlight controls and pre-fill form fields with allow-listed tools; it never presses Start, Approve, Export or Delete for the user.",
        "kb": table_of_contents(),
    }


def _ui_navigate(ctx: ToolContext, args: dict) -> dict[str, Any]:
    key = str(args.get("page") or "")
    url, err = registry.resolve_page_url(key, ctx.pid)
    if err:
        return {"error": err}
    page = registry.PAGES[key]
    return {
        "ok": True, "effect": "navigate", "page": key, "url": url, "title": page.title,
        "ui_event": {"type": "navigate", "page": key, "url": url, "title": page.title},
    }


def _ui_highlight(ctx: ToolContext, args: dict) -> dict[str, Any]:
    cid = str(args.get("control_id") or "")
    control = registry.CONTROLS.get(cid)
    if control is None:
        return {"error": f"unknown control {cid!r}", "allowed": _CONTROL_KEYS}
    url, err = registry.resolve_page_url(control.page, ctx.pid)
    if err:
        return {"error": err}
    return {
        "ok": True, "effect": "highlight", "control": cid, "label": control.label, "page": control.page,
        "url": url, "opens_page_first": ctx.page != control.page,
        "ui_event": {"type": "highlight", "control": cid, "selector": control.selector,
                     "label": control.label, "page": control.page, "url": url},
    }


def _ui_suggest(ctx: ToolContext, args: dict) -> dict[str, Any]:
    page = str(args.get("page") or "")
    settings = args.get("settings")
    if not isinstance(settings, dict) or not settings:
        return {"error": "settings must be a non-empty object of field → value"}
    url, err = registry.resolve_page_url(page, ctx.pid)
    if err:
        return {"error": err}
    allowed = sorted(name for (p, name) in registry.FIELDS if p == page)
    if not allowed:
        return {"error": f"page {page!r} has no prefillable fields",
                "pages_with_fields": sorted({p for p, _ in registry.FIELDS})}
    applied: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    fields: list[dict[str, Any]] = []
    for name, raw in settings.items():
        spec = registry.FIELDS.get((page, str(name)))
        if spec is None:
            rejected[str(name)] = f"not a prefillable field on {page!r}; allowed: {allowed}"
            continue
        value, problem = registry.coerce_field_value(spec, raw)
        if problem:
            rejected[str(name)] = problem
            continue
        applied[spec.name] = value
        fields.append({"name": spec.name, "selector": spec.selector, "scope": spec.scope,
                       "kind": spec.kind, "value": value, "label": spec.label})
    if not applied:
        return {"ok": False, "error": "no field could be pre-filled", "rejected": rejected, "allowed": allowed}
    out: dict[str, Any] = {
        "ok": True, "effect": "prefill", "page": page, "url": url, "applied": applied,
        "note": "Fields are pre-filled and editable; nothing was submitted. The user reviews and presses the button.",
        "ui_event": {"type": "prefill", "page": page, "url": url, "fields": fields},
    }
    if rejected:
        out["rejected"] = rejected
    return out


def _list_sources(pid: str) -> dict[str, Any]:
    sources_dir = project_dir(pid) / "qa" / "sources"
    if not sources_dir.exists():
        return {"sources": []}
    out = []
    for p in sorted(sources_dir.glob("*.json")):
        try:
            src = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        out.append({
            "id": src.get("id", p.stem),
            "filename": src.get("filename", "?"),
            "status": src.get("status", "ready"),
            "doc_count": src.get("doc_count", 1),
            "chunk_count": src.get("chunk_count", 0),
        })
    return {"sources": out}


def _read_source(pid: str, args: dict) -> dict[str, Any]:
    sid = args.get("source_id", "")
    if not sid:
        return {"error": "source_id required"}
    src = qa_fs.read_qa_source(pid, sid)
    if not src:
        return {"error": f"source {sid} not found"}
    # Prefer in-manifest text (legacy path); the data-prep runner stores parsed text at
    # files/<sha256[:12]>/parsed.txt, so fall back to disk when the manifest has none.
    text = src.get("text") or src.get("parsed_text") or ""
    if not text:
        from finetune_studio.data.fs.paths import file_dir
        parsed_path = file_dir(pid, src.get("sha256") or sid) / "parsed.txt"
        if parsed_path.exists():
            try:
                text = parsed_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
    return {
        "id": src.get("id", sid),
        "filename": src.get("filename", "?"),
        "text": text[:8000],
        "truncated": len(text) > 8000,
    }


def _list_pairs(pid: str, args: dict) -> dict[str, Any]:
    pairs = qa_fs.list_qa_pairs(pid, source_id=args.get("source_id"), status=args.get("status"))
    return {
        "pairs": [
            {
                "id": p.get("id"),
                "source_id": p.get("source_id"),
                "question": p.get("question", "")[:200],
                "answer": p.get("answer", "")[:300],
                "status": p.get("status", "pending"),
            }
            for p in pairs[:200]
        ],
        "count": len(pairs),
    }


def _create_pairs(pid: str, args: dict) -> dict[str, Any]:
    sid = args.get("source_id", "")
    pairs = args.get("pairs") or []
    if not sid or not pairs:
        return {"error": "source_id and pairs required"}
    written = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        q = (pair.get("question") or "").strip()
        a = (pair.get("answer") or "").strip()
        if not q or not a:
            continue
        qa_fs.write_qa_pair(pid, {
            "id": f"qa_{int(time.time() * 1000)}_{written}",
            "source_id": sid,
            "question": q,
            "answer": a,
            "chunk_idx": max(1, int(pair.get("chunk_idx") or 1)),
            "status": "pending",
            "created_at": time.time(),
            "created_via": "data-prep-chat",
        })
        written += 1
    return {"written": written, "source_id": sid}


def _need_project(ctx: ToolContext) -> dict[str, Any] | None:
    if not ctx.pid:
        return {"error": "no active project; open a project first (navigate to 'projects')"}
    return None


_PROJECT_TOOLS = {"list_sources", "read_source", "list_qa_pairs", "create_qa_pairs", "inspect_project_readiness"}


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool; always returns a JSON-serializable dict (``error`` key on failure)."""
    args = args if isinstance(args, dict) else {}
    try:
        if name in _PROJECT_TOOLS and (missing := _need_project(ctx)):
            return missing
        pid = ctx.pid or ""
        if name == "app_help":
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query required"}
            hits = search_kb(query)
            return {"query": query, "results": hits} if hits else {
                "query": query, "results": [],
                "note": "Nothing in the app knowledge base matches; say you do not know rather than guessing.",
                "kb": table_of_contents(),
            }
        if name == "get_app_guide":
            return app_guide()
        if name == "project_overview":
            return state.project_overview(ctx.pid)
        if name == "inspect_project_readiness":
            return state.readiness(pid)
        if name == "list_datasets":
            return state.list_datasets(ctx.pid)
        if name == "list_runs":
            return state.list_runs(ctx.pid)
        if name == "system_status":
            return state.system_status()
        if name == "recommend_training":
            return state.recommend_training(
                ctx.pid, goal=str(args.get("goal") or ""), tier=str(args.get("tier") or ""),
                base_model=str(args.get("base_model") or ""), dataset_id=str(args.get("dataset_id") or ""),
            )
        if name == "dataset_health":
            return state.dataset_health(
                ctx.pid, dataset_id=str(args.get("dataset_id") or "") or None,
                training_mode=str(args.get("training_mode") or "") or None,
            )
        if name == "explain_setting":
            return settings_help.explain_setting(str(args.get("name") or ""))
        if name == "navigate":
            return _ui_navigate(ctx, args)
        if name == "highlight":
            return _ui_highlight(ctx, args)
        if name == "suggest_settings":
            return _ui_suggest(ctx, args)
        if name == "list_sources":
            return _list_sources(pid)
        if name == "read_source":
            return _read_source(pid, args)
        if name == "list_qa_pairs":
            return _list_pairs(pid, args)
        if name == "create_qa_pairs":
            return _create_pairs(pid, args)
        return {"error": f"unknown tool: {name}"}
    except Exception as e:
        log.exception("tool %s failed", name)
        return {"error": f"tool {name} failed: {e}"}
