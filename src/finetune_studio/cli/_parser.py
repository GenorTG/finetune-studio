"""argparse setup — single place that defines every subcommand + every flag.

WHY ONE BIG PARSER FILE
=======================
argparse works by *side effect*: subparsers are attached to a parent
parser with `add_subparsers(...)`. Once a parser is built, you can't
easily split it across files. So all the `add_parser(...)` + `add_argument(...)`
calls live here in one place, organised by subcommand. The *handlers*
(`cmd_*` functions) live in `commands/` and are wired up by `_registry.py`.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune-studio",
        description="Finetune Studio — model training & RAG CLI",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── models ──
    p_models = sub.add_parser("models", help="List discovered models")
    p_models.add_argument("--dirs", nargs="*", help="Extra directories to scan")
    p_models.add_argument("--json", action="store_true", help="Output as JSON")

    # ── projects (list, inspect, create and delete local projects) ──
    p_projects = sub.add_parser("projects", help="List, inspect, create or delete projects")
    projects_sub = p_projects.add_subparsers(dest="projects_action", required=True)
    p_projects_list = projects_sub.add_parser("list", help="List projects")
    p_projects_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_projects_show = projects_sub.add_parser("show", help="Show a project by id or name")
    p_projects_show.add_argument("project", help="Project id or exact name")
    p_projects_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_projects_create = projects_sub.add_parser("create", help="Create a project")
    p_projects_create.add_argument("--name", required=True, help="Project name")
    p_projects_create.add_argument("--description", default="", help="Optional project description")
    p_projects_create.add_argument("--base-model", default="", help="Optional base model path or id")
    p_projects_create.add_argument("--system-prompt", default="", help="Optional project system prompt")
    p_projects_create.add_argument("--allow-duplicate", action="store_true",
                                   help="Allow another project with the same name")
    p_projects_create.add_argument("--json", action="store_true", help="Output as JSON")
    p_projects_delete = projects_sub.add_parser("delete", help="Delete a project and all of its stored data")
    p_projects_delete.add_argument("project", help="Project id or name")
    p_projects_delete.add_argument("--yes", action="store_true", help="Confirm deletion without prompting")
    p_projects_delete.add_argument("--json", action="store_true", help="Output as JSON")

    # ── train ──
    p_train = sub.add_parser("train", help="Start training")
    p_train.add_argument("model", help="Path to base model")
    p_train.add_argument("data", help="Path to training data (JSONL)")
    p_train.add_argument("-o", "--output", default="output", help="Output directory")
    p_train.add_argument("--lr", type=float, default=8e-5, help="Learning rate")
    p_train.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    p_train.add_argument("--batch", type=int, default=2, help="Batch size")
    p_train.add_argument("--lora-rank", type=int, default=64, help="LoRA rank")
    p_train.add_argument("--max-seq", type=int, default=2048, help="Max sequence length")
    p_train.add_argument("--system-prompt", default="", help="System prompt")
    p_train.add_argument("--no-unsloth", action="store_true", help="Use standard transformers")

    # ── test ──
    p_test = sub.add_parser("test", help="Test a model interactively")
    p_test.add_argument("model", help="Path to model")
    p_test.add_argument("--max-tokens", type=int, default=512)
    p_test.add_argument("--temperature", type=float, default=0.7)

    # ── suite ──
    p_suite = sub.add_parser("suite", help="Run a test suite and record the raw answers (judging is a separate step)")
    p_suite.add_argument("model", help="Path to model")
    p_suite.add_argument("suite", help="Path to test suite JSON")
    p_suite.add_argument("--max-tokens", type=int, default=512)
    p_suite.add_argument("--json", action="store_true", help="Output as JSON")
    p_suite.add_argument("--out", help="Write the transcripts (one JSON object per line) to this file")
    p_suite.add_argument(
        "--judge", nargs="?", const="", default=None, metavar="PROVIDER_ID",
        help="Afterwards let an AI judge read the answers (provider row id; default: the AI assistant's Test judge job)",
    )

    # ── validate ──
    p_val = sub.add_parser("validate", help="Validate training data")
    p_val.add_argument("files", nargs="+", help="Files to validate")

    # ── convert ──
    p_conv = sub.add_parser("convert", help="Convert data formats")
    p_conv.add_argument("source", help="Source file")
    p_conv.add_argument("target_format", choices=["jsonl", "json", "csv"])
    p_conv.add_argument("-o", "--output", help="Output path")
    p_conv.add_argument("--system-prompt", default="")

    # ── webui ──
    p_web = sub.add_parser("webui", help="Start the WebUI server")
    p_web.add_argument("--host", default=None, help="Listen address (default: Settings -> Security, $FTS_ROOT/server.json; 127.0.0.1 = this computer only)")
    p_web.add_argument("--port", type=int, default=None, help="Port (default: $FTS_ROOT/server.json, else 7860)")
    p_web.add_argument("--reload", action="store_true")

    # ── supervisor ──
    p_sup = sub.add_parser("supervisor", help="Run the supervisor (what the systemd unit runs)")
    p_sup.add_argument("--host", default=None, help="Listen address (default: Settings -> Security, $FTS_ROOT/server.json; 127.0.0.1 = this computer only)")
    p_sup.add_argument("--port", type=int, default=None, help="Port (default: $FTS_ROOT/server.json, else 7860)")
    p_status = sub.add_parser("status", help="Component table from the supervisor (exit 1 degraded, 3 supervisor down)")
    p_status.add_argument("--json", action="store_true")
    for verb in ("start", "stop", "restart"):
        p_act = sub.add_parser(verb, help=f"{verb.capitalize()} a supervised component")
        p_act.add_argument("name", nargs="?", default="web", help="Component name (default: web)")
    p_logs = sub.add_parser("logs", help="Recent output of a supervised component")
    p_logs.add_argument("name", nargs="?", default="web")
    p_logs.add_argument("-n", "--lines", type=int, default=100)
    p_events = sub.add_parser("events", help="Supervisor event log (spawns, exits, health, restarts)")
    p_events.add_argument("-n", "--lines", type=int, default=50)
    p_events.add_argument("-f", "--follow", action="store_true")
    p_up = sub.add_parser("up", help="Start the supervisor: systemd when the unit is installed, else detached manual")
    mode = p_up.add_mutually_exclusive_group()
    mode.add_argument("--manual", action="store_true", help="Detached supervisor process, no systemd")
    mode.add_argument("--systemd", action="store_true", help="Start the installed systemd user unit")
    p_up.add_argument("--host", default=None, help="Listen address (default: Settings -> Security, $FTS_ROOT/server.json; 127.0.0.1 = this computer only)")
    p_up.add_argument("--port", type=int, default=None, help="Port (default: $FTS_ROOT/server.json, else 7860)")
    sub.add_parser("down", help="Stop the supervisor and everything it runs (works for both start modes)")
    p_svc = sub.add_parser("service", help="Install or remove the optional systemd user unit")
    p_svc.add_argument("action", choices=["install", "uninstall", "status", "restart"])
    p_svc.add_argument("--no-start", action="store_true", help="install: write and enable the unit but do not start it")
    # ── jobs: training/export/... as their own processes ──
    p_jobs = sub.add_parser("jobs", help="Long jobs (training, export, ...) that run as their own processes")
    jobs_sub = p_jobs.add_subparsers(dest="jobs_action")
    p_jl = jobs_sub.add_parser("list", help="Running jobs (--all: also finished ones from the last week)")
    p_jl.add_argument("--all", action="store_true")
    p_js = jobs_sub.add_parser("stop", help="Stop a job cooperatively (training saves what it has)")
    p_js.add_argument("id")
    p_jg = jobs_sub.add_parser("logs", help="Recent output of a job")
    p_jg.add_argument("id")
    p_jg.add_argument("-n", "--lines", type=int, default=100)
    p_jobs.set_defaults(all=False)

    # ── access: password, API tokens, sessions, listen address ──
    p_auth = sub.add_parser("auth", help="Password, API tokens and browser sessions (who may use the web UI)")
    auth_sub = p_auth.add_subparsers(dest="auth_action", required=True)
    auth_sub.add_parser("status", help="Password on/off, listen address, sessions, tokens")
    p_setpw = auth_sub.add_parser("set-password", help="Turn sign-in on / change the password (prompt, stdin or file)")
    p_setpw.add_argument("--password-file", help="Read the password from this file (for automated installs)")
    p_dis = auth_sub.add_parser("disable", help="Remove the password (no sign-in asked)")
    p_dis.add_argument("--force", action="store_true", help="Also replace an unreadable auth.json")
    p_tok = auth_sub.add_parser("token", help="API tokens for scripts: Authorization: Bearer <token>")
    tok_sub = p_tok.add_subparsers(dest="token_action", required=True)
    tok_sub.add_parser("list")
    p_tok_c = tok_sub.add_parser("create")
    p_tok_c.add_argument("name")
    p_tok_r = tok_sub.add_parser("revoke")
    p_tok_r.add_argument("id")
    auth_sub.add_parser("sessions", help="Sign out every browser")
    p_apply = auth_sub.add_parser("apply-config", help="Apply an install config file (FTS_BIND_HOST, FTS_AUTH, ...)")
    p_apply.add_argument("file")
    auth_sub.add_parser("setup", help="Ask who may open the web UI (what install.sh asks on a terminal)")
    p_srv = sub.add_parser("server", help="Listen address of the web UI ($FTS_ROOT/server.json)")
    srv_sub = p_srv.add_subparsers(dest="server_action", required=True)
    srv_sub.add_parser("show")
    p_srv_set = srv_sub.add_parser("set", help="Who can reach the web UI (switches; this computer always can)")
    p_srv_set.add_argument("--lan", choices=("on", "off"), help="Devices on your local network / Wi-Fi")
    p_srv_set.add_argument("--tailscale", choices=("on", "off"), help="Your Tailscale devices")
    p_srv_set.add_argument("--custom", help="Comma list of other addresses of this computer to listen on "
                                            "(e.g. 198.51.100.2, :: or 0.0.0.0 = every network); '' clears it")
    p_srv_set.add_argument("--port", type=int)
    p_srv_set.add_argument("--allowed-hosts", help="Comma list of extra host names browsers may use (reverse proxy)")
    p_srv_set.add_argument("--host", help="Old single-address form: 127.0.0.1 = this computer, 0.0.0.0 = LAN + "
                                          "Tailscale, other = a custom address")
    p_srv_set.add_argument("--yes", action="store_true", help="Allow a network address without a password")

    p_doc = sub.add_parser("doctor", help="Diagnose the install with the app down (supervisor, port, unit, GPU choice)")
    p_doc.add_argument("--port", type=int, default=None, help="default: $FTS_ROOT/server.json, else 7860")

    # ── brain (Smart Brain index: build / search / ask) ──
    from finetune_studio.cli.commands.brain import add_parser as add_brain_parser
    add_brain_parser(sub)

    # ── heretic (optional external tool: removes a model's refusals) ──
    from finetune_studio.cli.commands.heretic import add_parser as add_heretic_parser
    add_heretic_parser(sub)

    # ── compare ──
    p_cmp = sub.add_parser("compare", help="Ask several models the same questions and show the answers side by side (judging is a separate step)")
    p_cmp.add_argument("--models", nargs="+", required=True, help="Model paths (name=path format)")
    p_cmp.add_argument("suite", help="Test suite JSON file")
    p_cmp.add_argument("--max-tokens", type=int, default=512)
    p_cmp.add_argument("--temperature", type=float, default=0.3)
    p_cmp.add_argument("--json", action="store_true", help="Output as JSON")
    p_cmp.add_argument("--report", help="Save the answers (and verdicts, with --judge) as JSON to this file")
    p_cmp.add_argument(
        "--judge", nargs="?", const="", default=None, metavar="PROVIDER_ID",
        help="Afterwards let an AI judge read every answer (provider row id; default: the AI assistant's Test judge job)",
    )

    # ── benchmark ──
    p_bench = sub.add_parser("benchmark", help="Run industry-standard benchmarks")
    p_bench.add_argument("model", help="Path to model (GGUF or safetensors)")
    p_bench.add_argument(
        "--suite",
        default="all",
        help="Real suite: all, mmlu, gsm8k, hellaswag, ifeval (or comma-separated)",
    )
    p_bench.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Bounded samples per suite (ignored with --full-run)",
    )
    p_bench.add_argument(
        "--full-run",
        action="store_true",
        help="Evaluate the entire official split (can be very slow)",
    )
    p_bench.add_argument("--json", action="store_true", help="Output as JSON")
    p_bench.add_argument("--report", help="Save report to file")

    # ── analyze (data quality) ──
    p_analyze = sub.add_parser("analyze", help="Analyze training data quality")
    p_analyze.add_argument("data", help="Path to training data (JSONL)")
    p_analyze.add_argument("--json", action="store_true")

    # ── validate-hallucination ──
    p_val_h = sub.add_parser("validate-hallucination", help="Count training answers with dates, figures, URLs or e-mails to check by hand (a pattern scan, not a fact check)")
    p_val_h.add_argument("data", help="Path to training data (JSONL)")
    p_val_h.add_argument("--json", action="store_true")

    # ── accel ──
    p_accel = sub.add_parser("accel", help="Show the compute accelerator (GPU first); exit 1 if a GPU host runs on CPU")
    p_accel.add_argument("--json", action="store_true")

    # ── vram ──
    p_vram = sub.add_parser("vram", help="GPU VRAM profiling & recommendations")
    vram_sub = p_vram.add_subparsers(dest="vram_command")

    p_vram_report = vram_sub.add_parser("report", help="Generate VRAM report for current GPU")
    p_vram_report.add_argument("--output", help="Save report to file")
    p_vram_report.add_argument("--vram", type=float, help="Override available VRAM (GB)")

    p_vram_check = vram_sub.add_parser("check", help="Check if a model fits in VRAM")
    p_vram_check.add_argument("model", help="Model name or size (e.g. 'qwen2.5-7b' or '7b')")
    p_vram_check.add_argument("--method", default="qlora", choices=["qlora", "lora", "full_ft"])
    p_vram_check.add_argument("--batch", type=int, default=2)
    p_vram_check.add_argument("--seq", type=int, default=2048)
    p_vram_check.add_argument("--lora-rank", type=int, default=64)
    p_vram_check.add_argument("--vram", type=float, help="Override available VRAM (GB)")
    p_vram_check.add_argument("--json", action="store_true")

    p_vram_profile = vram_sub.add_parser("profile", help="Profile actual VRAM usage (requires model)")
    p_vram_profile.add_argument("model", help="HuggingFace model ID or local path")
    p_vram_profile.add_argument("--method", default="qlora", choices=["qlora", "lora"])
    p_vram_profile.add_argument("--batch", type=int, default=1)
    p_vram_profile.add_argument("--seq", type=int, default=512)
    p_vram_profile.add_argument("--lora-rank", type=int, default=16)
    p_vram_profile.add_argument("--steps", type=int, default=5)
    p_vram_profile.add_argument("--json", action="store_true")

    # ── files (project file library) ──
    p_files = sub.add_parser("files", help="Upload, inspect, parse or clean up project files")
    files_sub = p_files.add_subparsers(dest="files_command")
    p_files_upload = files_sub.add_parser("upload", help="Upload one or more files into a project")
    p_files_upload.add_argument("paths", nargs="+", help="Local file paths to upload")
    p_files_upload.add_argument("--project", required=True, help="Project id or name")
    p_files_upload.add_argument("--json", action="store_true", help="Output the upload report as JSON")
    p_files_list = files_sub.add_parser("list", help="List files in a project")
    p_files_list.add_argument("--project", required=True, help="Project id or name")
    p_files_list.add_argument("--search", help="Filter by file name")
    p_files_list.add_argument("--mime", help="Filter by MIME type prefix, e.g. text/")
    p_files_list.add_argument("--include-deleted", action="store_true", help="Include files in trash")
    p_files_list.add_argument("--json", action="store_true", help="Output as JSON")
    p_files_show = files_sub.add_parser("show", help="Show one file by id or name")
    p_files_show.add_argument("file", help="File id or exact name")
    p_files_show.add_argument("--project", required=True, help="Project id or name")
    p_files_show.add_argument("--include-deleted", action="store_true", help="Allow a file in trash")
    p_files_show.add_argument("--json", action="store_true", help="Output as JSON")
    p_files_parse = files_sub.add_parser("parse", help="Parse a file into readable text and passages")
    p_files_parse.add_argument("file", help="File id or exact name")
    p_files_parse.add_argument("--project", required=True, help="Project id or name")
    p_files_parse.add_argument("--reparse", action="store_true",
                               help="Discard any hand-edited parsed text and parse the original file again")
    p_files_parse.add_argument("--json", action="store_true", help="Output as JSON")
    p_files_trash = files_sub.add_parser("trash", help="List or purge trashed files")
    p_files_trash.add_argument("--list", action="store_true",
                                help="List trashed files instead of purging")
    p_files_trash.add_argument("--older-than-days", type=int, default=7,
                                help="Purge threshold in days (default: 7). Pass 0 with --all to skip age filter.")
    p_files_trash.add_argument("--all", action="store_true",
                                help="Purge everything in trash regardless of age")
    p_files_trash.add_argument("--project-id", help="Restrict to a single project")
    p_files_trash.add_argument("--dry-run", action="store_true",
                                help="Show what would be purged, do not delete")

    # ── mining jobs (persistent per-file helper work) ──
    p_mining = sub.add_parser("mining", help="Queue and inspect persistent per-file Mining jobs")
    mining_sub = p_mining.add_subparsers(dest="mining_action", required=True)
    p_mining_start = mining_sub.add_parser("start", help="Mine files or fill gaps in their Q&A coverage")
    p_mining_start.add_argument("files", nargs="+", help="Parsed file ids or names")
    p_mining_start.add_argument("--project", required=True, help="Project id or name")
    p_mining_start.add_argument("--kind", choices=("mine", "fill"), default="mine",
                                 help="mine builds the inventory; fill revisits open data points")
    p_mining_start.add_argument("--json", action="store_true", help="Output queued jobs as JSON")
    p_mining_list = mining_sub.add_parser("list", help="List Mining jobs in a project")
    p_mining_list.add_argument("--project", required=True, help="Project id or name")
    p_mining_list.add_argument("--source-id", help="Filter by parsed source id")
    p_mining_list.add_argument("--active", action="store_true", help="Show only queued and running jobs")
    p_mining_list.add_argument("--limit", type=int, default=100, help="Maximum jobs to show (default: 100)")
    p_mining_list.add_argument("--json", action="store_true", help="Output jobs and helper settings as JSON")
    for action, help_text in (("status", "Show one job's progress"), ("cancel", "Cancel or stop a Mining job")):
        p_mining_job = mining_sub.add_parser(action, help=help_text)
        p_mining_job.add_argument("job_id", help="Mining job id")
        p_mining_job.add_argument("--project", required=True, help="Owning project id or name")
        p_mining_job.add_argument("--json", action="store_true", help="Output as JSON")

    # ── qa review ──
    p_qa = sub.add_parser("qa", help="Review mined Q&A pairs and inspect their status")
    qa_sub = p_qa.add_subparsers(dest="qa_action", required=True)
    p_qa_list = qa_sub.add_parser("list", help="List the review queue")
    p_qa_list.add_argument("--project", required=True, help="Project id or name")
    p_qa_list.add_argument("--status", choices=("pending", "approved", "rejected"), help="Filter by review status")
    p_qa_list.add_argument("--source", help="Filter by source id or parsed file name")
    p_qa_list.add_argument("--search", default="", help="Search question and answer text")
    p_qa_list.add_argument("--offset", type=int, default=0, help="Skip this many matching pairs")
    p_qa_list.add_argument("--limit", type=int, default=200, help="Maximum pairs to show (default: 200)")
    p_qa_list.add_argument("--json", action="store_true", help="Output the queue page and counts as JSON")
    for action, help_text in (("approve", "Approve Q&A pairs for export"), ("reject", "Reject Q&A pairs")):
        p_qa_review = qa_sub.add_parser(action, help=help_text)
        p_qa_review.add_argument("ids", nargs="+", help="Q&A pair ids")
        p_qa_review.add_argument("--project", required=True, help="Project id or name")
        p_qa_review.add_argument("--note", help="Optional reviewer note")
        if action == "reject":
            p_qa_review.add_argument("--yes", action="store_true",
                                     help="Confirm rejection that would leave mined facts uncovered")
        p_qa_review.add_argument("--json", action="store_true", help="Output counts and updated pairs as JSON")

    # ── versions (project snapshots and lineage) ──
    p_versions = sub.add_parser("versions", help="Pin and inspect project snapshots and their lineage")
    versions_sub = p_versions.add_subparsers(dest="versions_action", required=True)
    p_versions_list = versions_sub.add_parser("list", help="List pinned project versions")
    p_versions_list.add_argument("--project", required=True, help="Project id or name")
    p_versions_list.add_argument("--json", action="store_true", help="Output versions and manifests as JSON")
    p_versions_pin = versions_sub.add_parser("pin", help="Pin current datasets, runs and Smart Brain build")
    p_versions_pin.add_argument("--project", required=True, help="Project id or name")
    p_versions_pin.add_argument("--label", default="", help="Optional version label")
    p_versions_pin.add_argument("--notes", default="", help="Optional snapshot notes")
    pin_parent = p_versions_pin.add_mutually_exclusive_group()
    pin_parent.add_argument("--parent", help="Build on this version id (default: latest version)")
    pin_parent.add_argument("--root", action="store_true", help="Start a new lineage instead of using the latest")
    p_versions_pin.add_argument("--json", action="store_true", help="Output the pinned version as JSON")
    for action, help_text in (("show", "Show one version and its manifest"),
                              ("lineage", "Show the root-to-version history"),
                              ("delete", "Delete a version record only")):
        p_version = versions_sub.add_parser(action, help=help_text)
        p_version.add_argument("version_id", help="Version id from `fts versions list`")
        p_version.add_argument("--project", required=True, help="Owning project id or name")
        if action == "delete":
            p_version.add_argument("--yes", action="store_true", help="Confirm deleting only the version record")
        p_version.add_argument("--json", action="store_true", help="Output the result as JSON")

    # ── dataset (build a training dataset from a project's Q&A pairs) ──
    p_dataset = sub.add_parser("dataset", help="Build training datasets from a project")
    dataset_sub = p_dataset.add_subparsers(dest="dataset_command")
    p_ds_build = dataset_sub.add_parser(
        "build", help="Coverage-fill, export and register a project's dataset (same as the WebUI export)")
    p_ds_build.add_argument("--project", required=True, help="Project id or name")
    p_ds_build.add_argument("--name", help="Custom dataset name (default: '<project> · <fmt> · <N> rows')")
    p_ds_build.add_argument("--fmt", default="sharegpt", choices=["sharegpt", "alpaca", "openai"],
                            help="Row format (default: sharegpt)")
    p_ds_build.add_argument("--only", default="approved", choices=["approved", "pending", "rejected", "all"],
                            help="Which pairs to export by review status (default: approved)")
    grounding_group = p_ds_build.add_mutually_exclusive_group()
    grounding_group.add_argument("--grounded-share", type=float, default=None, metavar="0-1",
                                 help="Fraction of rows rewritten to carry the RAG-chat prompt + CONTEXT from "
                                      "the pair's own source chunk (default: auto = 0.4 when the project has "
                                      "a built Smart Brain index, else off)")
    grounding_group.add_argument("--no-grounding", action="store_true",
                                 help="Plain rows only (same as --grounded-share 0)")
    p_ds_build.add_argument("--distractors", type=int, default=0, metavar="0-2",
                            help="Extra other-file chunks added to each grounded row's CONTEXT (default: 0)")
    p_ds_build.add_argument("--seed", type=int, default=42,
                            help="Seed for which rows get grounded (default: 42)")
    p_ds_build.add_argument("--force", action="store_true",
                            help="Export even if some files are not complete (open or unreviewed data points; blocked by default)")
    p_ds_build.add_argument("--out", metavar="PATH",
                            help="Also write the JSONL here (must be inside the project's directories)")
    p_ds_build.add_argument("--json", action="store_true", help="Print the summary as JSON")

    p_ds_pref = dataset_sub.add_parser(
        "build-preference",
        help="Author DPO preference pairs (chosen vs rejected) from approved Q&A and register the dataset")
    p_ds_pref.add_argument("--project", required=True, help="Project id or name")
    p_ds_pref.add_argument("--kinds", default="hallucination,abstain", metavar="LIST",
                           help="Comma-separated pair kinds: hallucination, abstain (default: both)")
    p_ds_pref.add_argument("--max-pairs", type=int, default=100, metavar="N",
                           help="Upper bound on pairs written, split evenly across kinds (default: 100)")
    p_ds_pref.add_argument("--seed", type=int, default=42, help="Seed for candidate order and the split preview")
    p_ds_pref.add_argument("--json", action="store_true", help="Print the summary as JSON")

    p_ds_study = dataset_sub.add_parser(
        "build-study",
        help="Write each approved fact several ways (reworded questions, restated answers, a passage) as an SFT dataset")
    p_ds_study.add_argument("--project", required=True, help="Project id or name")
    p_ds_study.add_argument("--reworded", type=int, default=2, metavar="N", help="Reworded questions per pair (0-5, default 2)")
    p_ds_study.add_argument("--restated", type=int, default=1, metavar="N", help="Restated answers per pair (0-5, default 1)")
    p_ds_study.add_argument("--no-passage", action="store_true", help="Skip the short passage per pair")
    p_ds_study.add_argument("--json", action="store_true", help="Print the summary as JSON")

    return parser
