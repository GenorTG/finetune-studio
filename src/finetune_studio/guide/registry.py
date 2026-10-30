"""Allow-lists the guide may act on: pages, highlightable controls, prefillable fields.

WHAT THIS FILE DOES
===================
The guide's UI tools (``navigate`` / ``highlight`` / ``suggest_settings``) never
accept a free-form URL, selector or field. Every target the model can name is
a row in this file, and ``tests/test_guide_registry.py`` checks each selector
against the real template, so a renamed element fails a test instead of the
guide silently pointing at nothing.

Selector grammar (kept tiny so it is statically checkable): ``#element-id`` or
``[name="field-name"]``. The browser side resolves a field's ``scope`` (a form
selector) first, then the selector inside it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Page:
    key: str
    title: str
    url: str          # ``{pid}`` is replaced by the active project id
    template: str     # file under webui/templates that renders it
    project: bool     # needs an active project
    purpose: str


@dataclass(frozen=True)
class Control:
    key: str
    page: str
    selector: str
    label: str


@dataclass(frozen=True)
class Field:
    page: str
    name: str
    selector: str
    kind: str                      # int | float | text | bool | choice
    label: str
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    scope: str = ""
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _pages() -> dict[str, Page]:
    rows = [
        ("dashboard", "Dashboard", "/", "index.html", False, "System overview: projects, GPU, loaded model."),
        ("projects", "Projects", "/projects", "projects.html", False, "Create, import, search and delete projects."),
        ("models_library", "Model library", "/models/explore", "hf_models.html", False, "Search and download base models from Hugging Face."),
        ("my_models", "My models", "/models", "models_index.html", False, "Models already on disk: downloaded bases and trained exports."),
        ("inference", "Inference", "/inference", "inference.html", False, "Load any model, set context and GPU layers, chat."),
        ("settings", "Settings", "/settings", "settings.html", False, "Updates, access (password, tokens, network address), AI assistant (connections, jobs, test judge), compute device."),
        ("overview", "Project overview", "/projects/{pid}", "project.html", True, "Project home: recent runs, models, files, activity."),
        ("wizard", "Quick work", "/projects/{pid}/wizard", "project_wizard.html", True, "One page that chains files, pairs, dataset, training, test."),
        ("files", "Files", "/projects/{pid}/data", "project_data.html", True, "Upload and organise documents; check each one parsed."),
        ("pairs", "Question-answer pairs", "/projects/{pid}/data-prep", "data_prep.html", True, "Generate, review and export Q&A pairs as the training dataset."),
        ("brain", "Smart Brain", "/projects/{pid}/brain", "brain.html", True, "Answer from the project's files without training: build the index, test a search, chat with sources."),
        ("training", "Training", "/projects/{pid}/training", "project_training.html", True, "Configure and start a fine-tune; live progress; past runs."),
        ("testing", "Testing", "/projects/{pid}/testing", "project_testing.html", True, "Run a test on your own questions (saved answers), then judge them with an AI judge or by hand."),
        ("compare", "Compare", "/projects/{pid}/compare", "project_compare.html", True, "Ask several models the same quiz, read the saved answers side by side, then judge them (AI judge or you)."),
        ("benchmarks", "Benchmarks", "/projects/{pid}/benchmarks", "benchmarks.html", True, "Public and synthetic benchmark suites; compare runs."),
        ("export", "Export", "/projects/{pid}/export", "export_models.html", True, "Export GGUF quants / merged / abliterated; browse trained exports."),
        ("chat", "Chat", "/projects/{pid}/chat", "chat_v2.html", True, "Test chat with the project's model, or Agent (guide) mode."),
        ("project_settings", "Project settings", "/projects/{pid}/settings", "project_settings.html", True, "Project settings and the service log tail."),
    ]
    return {r[0]: Page(*r) for r in rows}


PAGES: dict[str, Page] = _pages()


def _controls() -> dict[str, Control]:
    rows = [
        # projects
        ("projects.new_button", "projects", "#new-project-btn", "＋ New project"),
        ("projects.search", "projects", "#proj-search", "Project search box"),
        # files
        ("files.upload_toggle", "files", "#fb-upload-toggle", "Upload toggle"),
        ("files.upload_input", "files", "#fb-upload-input", "Choose files"),
        ("files.table", "files", "#fb-files-table", "Uploaded files table"),
        ("files.bulk_reparse", "files", "#fb-bulk-reparse", "Re-parse selected"),
        # pairs
        ("pairs.mining_card", "pairs", "#dp-mining", "Mine pairs, file by file card"),
        ("pairs.mine_selected", "pairs", "#dp-mine-selected", "Mine selected"),
        ("pairs.fill_selected", "pairs", "#dp-fill-selected", "Fill gaps on selected"),
        ("pairs.files_table", "pairs", "#dp-files", "Files table (status, data points, pairs)"),
        ("pairs.file_panel", "pairs", "#dp-file-panel", "A file's checklist and rounds"),
        ("pairs.review_card", "pairs", "#dp-review-card", "Training / Q&A output card"),
        ("pairs.filter_pending", "pairs", "#dp-filter-pending", "Pending filter"),
        ("pairs.approve_all_pending", "pairs", "#dp-approve-all-pending", "Approve all pending"),
        ("pairs.dedupe", "pairs", "#dp-dedupe", "Remove near-duplicates"),
        ("pairs.audit_card", "pairs", "#dp-audit-card", "Data audit card"),
        ("pairs.audit_run", "pairs", "#dp-audit-run", "Run the data audit"),
        ("pairs.grounded", "pairs", "#dp-grounded", "Include retrieved context"),
        ("pairs.export_approved", "pairs", "#dp-export-approved", "Export approved → training"),
        ("pairs.open_training", "pairs", "#dp-open-training", "Start training with this dataset"),
        # Smart Brain
        ("brain.quick_index", "brain", "#quick-index-btn", "⚡ Quick index"),
        ("brain.embedder", "brain", "#b-embedder", "Embedder"),
        ("brain.build", "brain", "#build-btn", "Build"),
        ("brain.build_stop", "brain", "#build-stop-btn", "Stop build"),
        ("brain.import_panel", "brain", "#brain-import-panel", "Import an index from another studio"),
        ("brain.import_file", "brain", "#bundle-import-file", "Bundle file to import"),
        ("brain.reembed", "brain", "#reembed-btn", "Re-embed only"),
        ("brain.search_box", "brain", "#q-text", "Search test box"),
        ("brain.reranker", "brain", "#s-reranker", "Reranker model"),
        ("brain.rerank_toggle", "brain", "#s-rerank-enabled", "Rerank toggle"),
        ("brain.hybrid_toggle", "brain", "#s-hybrid-enabled", "Hybrid toggle"),
        ("brain.export_layers", "brain", "#mcp-layers", "Standalone export: search layers"),
        ("brain.chat_follow_up", "brain", "#chat-follow-up", "Grounded chat: follow-up search"),
        ("brain.export_semantic", "brain", "#mcp-sem", "Export: meaning search (vectors)"),
        ("brain.export_keyword", "brain", "#mcp-kw", "Export: word search (BM25)"),
        ("brain.export_reranker", "brain", "#mcp-rerank", "Export: reranker"),
        ("brain.export_answer", "brain", "#mcp-answer", "Export: answer generation (LLM)"),
        ("brain.export_pipeline", "brain", "#mcp-pipeline", "Export: pipeline summary"),
        # training
        ("training.usecases", "training", "#uc-card", "Start from a use case"),
        ("training.usecase_new", "training", "#uc-new", "New use case form"),
        ("training.usecase_template", "training", "#uc-template", "Use-case template"),
        ("training.usecase_file", "training", "#uc-file", "Use-case examples file"),
        ("training.usecase_build", "training", "#uc-build-btn", "Check examples and build"),
        ("training.usecase_report", "training", "#uc-report", "Use-case evaluation and results"),
        ("training.usecase_draft", "training", "#uc-draft", "Draft examples with the AI assistant"),
        ("training.usecase_draft_file", "training", "#uc-draft-file", "Raw texts to draft examples from"),
        ("training.usecase_draft_btn", "training", "#uc-draft-btn", "Draft with the assistant"),
        ("training.usecase_draft_review", "training", "#uc-draft-review", "Review drafts (approve, fix, reject)"),
        ("training.route", "training", "#training-mode-help", "Training route help line"),
        ("training.base_model", "training", "#train-base-model", "Base model"),
        ("training.dataset_select", "training", "#dataset-select", "Project dataset picker"),
        ("training.dataset_upload", "training", "#dataset-upload", "Upload my own dataset"),
        ("training.dataset_health", "training", "#dataset-health", "Dataset check"),
        ("training.preset", "training", "#training-preset", "Training preset"),
        ("training.advisory", "training", "#preset-advisory", "Preset advisory"),
        ("training.epochs", "training", '[name="num_epochs"]', "Epochs"),
        ("training.lora_rank", "training", '[name="lora_rank"]', "LoRA rank"),
        ("training.lora_alpha", "training", '[name="lora_alpha"]', "LoRA alpha"),
        ("training.learning_rate", "training", '[name="learning_rate"]', "Learning rate"),
        ("training.batch_size", "training", '[name="batch_size"]', "Batch size"),
        ("training.max_seq", "training", '[name="max_seq_length"]', "Max sequence"),
        ("training.eval_steps", "training", "#eval-steps", "Eval every N steps"),
        ("training.early_stopping", "training", "#early-stopping-check", "Early stopping"),
        ("training.merge_on_save", "training", "#merge-on-save-check", "Also save merged model"),
        ("training.start", "training", "#start-btn", "Start training"),
        ("training.live_status", "training", "#train-status", "Live status"),
        ("training.past_runs", "training", "#past-runs-table", "Past runs"),
        # testing
        ("testing.mode", "testing", '[name="t-mode"]', "What to test (project quiz, RAG-grounded, dataset check)"),
        ("testing.model", "testing", "#t-model", "Model to test"),
        ("testing.suite", "testing", "#t-suite", "Question suite"),
        ("testing.run", "testing", "#t-run-btn", "Run test"),
        ("testing.eval_kind", "testing", "#t-eval-kind", "Evaluation type"),
        ("testing.auto_judge", "testing", "#t-auto-judge", "Judge automatically after each run"),
        ("testing.judge_provider", "testing", "#t-judge-provider", "Judge model"),
        ("testing.runs", "testing", "#t-runs", "Saved test runs"),
        ("testing.review", "testing", "#rv-list", "Cases of the open run (review list)"),
        # compare
        ("compare.models", "compare", "#cp-models", "Models to compare"),
        ("compare.suite", "compare", "#cp-suite", "Question suite"),
        ("compare.run", "compare", "#cp-run-btn", "Run comparison"),
        ("compare.judge", "compare", "#cp-judge-btn", "Judge all answers"),
        ("compare.cases", "compare", "#cp-cases", "Answers side by side"),
        # benchmarks
        ("benchmarks.run_latest", "benchmarks", "#bench-run-latest", "Run a new benchmark"),
        ("benchmarks.base_row", "benchmarks", "#bench-base-row", "Base-model benchmark row"),
        ("benchmarks.compare", "benchmarks", "#cmp-run-btn", "Run comparison"),
        # export
        ("export.runs", "export", "#export-runs", "Training runs"),
        ("export.gguf_quants", "export", "#gguf-quants", "GGUF quantizations"),
        ("export.gguf_imatrix", "export", "#gguf-imatrix", "Importance matrix (imatrix)"),
        ("export.start", "export", "#export-btn", "Export selected"),
        ("export.trained_exports", "export", "#trained-exports-table", "Trained exports"),
        # wizard
        ("wizard.run_all", "wizard", "#wiz-run-all", "Run everything at once"),
        ("wizard.train_step", "wizard", "#wiz-step-train", "Step 4 · Train"),
        ("wizard.epochs", "wizard", "#wiz-epochs", "Epochs override"),
        ("wizard.version_parent", "wizard", "#wiz-version-parent", "Builds on version"),
        ("wizard.versions", "wizard", "#wiz-versions-table", "Saved versions"),
        # chat
        ("chat.message_box", "chat", "#chat-input", "Chat message box"),
        # models / inference / settings
        ("models_library.search", "models_library", "#hf-q", "Model search"),
        ("my_models.table", "my_models", "#models-index-table", "Local models table"),
        ("my_models.rescan", "my_models", "#models-rescan", "Rescan folders"),
        ("inference.model_select", "inference", "#model-select", "Model picker"),
        ("inference.context", "inference", "#n-ctx", "Context length"),
        ("inference.gpu_layers", "inference", "#n-gpu-layers", "GPU offload layers"),
        ("inference.load", "inference", "#load-btn", "Load model"),
        ("settings.compute", "settings", "#compute-select", "Compute device"),
        ("settings.updates", "settings", "#btn-update-check", "Check for updates"),
        ("settings.judge_auto", "settings", "#judge-auto", "Judge automatically after a test run"),
        ("guide.panel", "overview", "#guide-panel", "The Guide panel"),
        ("activity.drawer", "models_library", "#activity", "Activity drawer: running jobs; Cancel download stops a model download"),
        ("guide.mode", "overview", "#guide-mode", "Guide actions: ask before each, or hands-free"),
        ("settings.assistant", "settings", "#assistant-card", "AI assistant: connections, main assistant, jobs"),
        ("settings.assistant_main", "settings", "#assistant-main", "Main assistant"),
        ("settings.connections", "settings", "#connections-table", "Connections (local model and API keys)"),
        ("settings.connection_add", "settings", "#btn-conn-add", "Add API connection"),
        ("settings.assistant_jobs", "settings", "#assistant-jobs", "Assistant jobs and their instructions"),
        ("settings.access", "settings", "#access-card", "Access & network: password, address, API tokens"),
        ("settings.heretic", "settings", "#heretic-card", "Heretic (optional tool): install, check upstream, remove"),
    ]
    return {r[0]: Control(*r) for r in rows}


CONTROLS: dict[str, Control] = _controls()

_TRAIN_FORM = "#train-form"


def _fields() -> dict[tuple[str, str], Field]:
    f: list[Field] = [
        Field("training", "training_mode", '[name="training_mode"]', "choice", "Training route",
              choices=("sft", "dpo", "tool_sft", "continued_pretraining", "reasoning_distillation"),
              scope=_TRAIN_FORM, note="Selecting DPO also sets conservative DPO defaults on the page."),
        Field("training", "preset", "#training-preset", "choice", "Training preset (by effort)",
              choices=("smoke", "balanced", "precision", "overkill"),
              note="The page then asks the preset advisor and fills epochs/rank/alpha/LR/batch/warmup itself."),
        Field("training", "num_epochs", '[name="num_epochs"]', "int", "Epochs", minimum=1, maximum=100, scope=_TRAIN_FORM),
        Field("training", "lora_rank", '[name="lora_rank"]', "int", "LoRA rank", minimum=4, maximum=512, scope=_TRAIN_FORM),
        Field("training", "lora_alpha", '[name="lora_alpha"]', "int", "LoRA alpha", minimum=1, maximum=1024, scope=_TRAIN_FORM),
        Field("training", "learning_rate", '[name="learning_rate"]', "text", "Learning rate", scope=_TRAIN_FORM,
              note="Scientific notation, e.g. 2e-4; must be between 1e-7 and 1e-2."),
        Field("training", "batch_size", '[name="batch_size"]', "int", "Batch size", minimum=1, maximum=64, scope=_TRAIN_FORM),
        Field("training", "gradient_accumulation_steps", '[name="gradient_accumulation_steps"]', "int",
              "Gradient accumulation steps", minimum=1, maximum=32, scope=_TRAIN_FORM),
        Field("training", "max_seq_length", '[name="max_seq_length"]', "int", "Max sequence", minimum=128, maximum=32768, scope=_TRAIN_FORM),
        Field("training", "warmup_steps", '[name="warmup_steps"]', "int", "Warmup steps", minimum=0, maximum=5000, scope=_TRAIN_FORM),
        Field("training", "eval_steps", '[name="eval_steps"]', "int", "Eval every N steps", minimum=0, maximum=10000, scope=_TRAIN_FORM),
        Field("training", "early_stopping", '[name="early_stopping"]', "bool", "Early stopping", scope=_TRAIN_FORM),
        Field("training", "merge_on_save", '[name="merge_on_save"]', "bool", "Also save merged model", scope=_TRAIN_FORM),
        Field("training", "system_prompt_mode", '[name="system_prompt_mode"]', "choice", "System prompt mode",
              choices=("bake", "runtime", "none"), scope=_TRAIN_FORM),
        Field("pairs", "grounded", "#dp-grounded", "bool", "Include retrieved context"),
        Field("pairs", "grounded_pct", "#dp-grounded-pct", "int", "Percent of rows with retrieved context", minimum=1, maximum=100),
        Field("brain", "embedder", "#b-embedder", "choice", "Embedder",
              choices=("intfloat/multilingual-e5-large", "BAAI/bge-large-en-v1.5", "Qwen/Qwen3-Embedding-0.6B",
                       "sentence-transformers/all-MiniLM-L6-v2")),
        Field("brain", "fact_cards", "#b-fact-cards", "bool", "Fact cards (index approved pairs)"),
        Field("brain", "reranker", "#s-reranker", "choice", "Reranker model",
              choices=("BAAI/bge-reranker-v2-m3", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
        Field("brain", "rerank_enabled", "#s-rerank-enabled", "bool", "Rerank"),
        Field("brain", "hybrid_enabled", "#s-hybrid-enabled", "bool", "Hybrid retrieval"),
        Field("brain", "rerank_top_n", "#s-rerank-top-n", "int", "Rerank top-N", minimum=1, maximum=200),
        Field("testing", "eval_kind", "#t-eval-kind", "choice", "Evaluation type", choices=("heldout", "training_leakage")),
        Field("wizard", "epochs", "#wiz-epochs", "int", "Epochs override", minimum=1, maximum=100),
        Field("inference", "n_ctx", "#n-ctx", "int", "Context length", minimum=0, maximum=1048576),
    ]
    return {(x.page, x.name): x for x in f}


FIELDS: dict[tuple[str, str], Field] = _fields()

_ID_RE = re.compile(r"#([A-Za-z][\w-]*)")
_NAME_RE = re.compile(r'\[name="([\w-]+)"\]')


def selector_tokens(selector: str) -> list[tuple[str, str]]:
    """``[("id", "x")]`` / ``[("name", "y")]`` pairs used to verify a selector in its template."""
    out = [("id", m) for m in _ID_RE.findall(selector)]
    out += [("name", m) for m in _NAME_RE.findall(selector)]
    return out


def resolve_page_url(page_key: str, pid: str | None) -> tuple[str | None, str | None]:
    """``(url, error)`` for an allow-listed page key under the active project."""
    page = PAGES.get(page_key)
    if page is None:
        return None, f"unknown page {page_key!r}; allowed: {', '.join(sorted(PAGES))}"
    if page.project:
        if not pid:
            return None, f"page {page_key!r} needs a project; navigate to 'projects' and open one first"
        return page.url.replace("{pid}", pid), None
    return page.url, None


def coerce_field_value(spec: Field, value: Any) -> tuple[Any, str | None]:
    """Validate/coerce one suggested value against its field spec; ``(value, error)``."""
    if spec.kind == "bool":
        if isinstance(value, bool):
            return value, None
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True, None
        if text in ("0", "false", "no", "off"):
            return False, None
        return None, f"{spec.name}: expected true/false"
    if spec.kind == "choice":
        text = str(value).strip()
        if text not in spec.choices:
            return None, f"{spec.name}: {text!r} not in {list(spec.choices)}"
        return text, None
    if spec.kind in ("int", "float"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None, f"{spec.name}: expected a number"
        if spec.kind == "int":
            if num != int(num):
                return None, f"{spec.name}: expected a whole number"
            num = int(num)
        if spec.minimum is not None and num < spec.minimum:
            return None, f"{spec.name}: {num} is below the minimum {spec.minimum:g}"
        if spec.maximum is not None and num > spec.maximum:
            return None, f"{spec.name}: {num} is above the maximum {spec.maximum:g}"
        return num, None
    text = str(value).strip()
    if spec.name == "learning_rate":
        try:
            lr = float(text)
        except ValueError:
            return None, "learning_rate: expected a number like 2e-4"
        if not 1e-7 <= lr <= 1e-2:
            return None, "learning_rate: must be between 1e-7 and 1e-2"
    if not text or len(text) > 64:
        return None, f"{spec.name}: empty or too long"
    return text, None
