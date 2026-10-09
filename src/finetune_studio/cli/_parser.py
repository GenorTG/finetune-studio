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
        help="Afterwards let an AI judge read the answers (provider row id; default: the configured judge / helper seat)",
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
    p_web.add_argument("--host", default="0.0.0.0")
    p_web.add_argument("--port", type=int, default=7860)
    p_web.add_argument("--reload", action="store_true")

    # ── rag ──
    p_rag = sub.add_parser("rag", help="RAG operations")
    rag_sub = p_rag.add_subparsers(dest="rag_command")

    p_rag_ingest = rag_sub.add_parser("ingest", help="Ingest documents into RAG store")
    p_rag_ingest.add_argument("path", help="File or directory to ingest")
    p_rag_ingest.add_argument("--chunk-size", type=int, default=512)
    p_rag_ingest.add_argument("--overlap", type=int, default=50)
    p_rag_ingest.add_argument("--store", default="data/rag_store", help="Store path")

    p_rag_query = rag_sub.add_parser("query", help="Query RAG store")
    p_rag_query.add_argument("question", help="Question to ask")
    p_rag_query.add_argument("--top-k", type=int, default=5)
    p_rag_query.add_argument("--store", default="data/rag_store")
    p_rag_query.add_argument("--json", action="store_true")

    p_rag_list = rag_sub.add_parser("list", help="List indexed documents")
    p_rag_list.add_argument("--store", default="data/rag_store")
    p_rag_list.add_argument("--json", action="store_true")

    p_rag_remove = rag_sub.add_parser("remove", help="Remove document from RAG store")
    p_rag_remove.add_argument("document_id", help="Document ID to remove")
    p_rag_remove.add_argument("--store", default="data/rag_store")

    p_rag_stats = rag_sub.add_parser("stats", help="RAG store statistics")
    p_rag_stats.add_argument("--store", default="data/rag_store")

    p_rag_clear = rag_sub.add_parser("clear", help="Clear RAG store")
    p_rag_clear.add_argument("--store", default="data/rag_store")
    p_rag_clear.add_argument("--confirm", action="store_true")

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
        help="Afterwards let an AI judge read every answer (provider row id; default: the configured judge / helper seat)",
    )

    # ── benchmark ──
    p_bench = sub.add_parser("benchmark", help="Run industry-standard benchmarks")
    p_bench.add_argument("model", help="Path to model (GGUF or safetensors)")
    p_bench.add_argument(
        "--suite",
        default="all",
        help="Real suite: all, mmlu, gsm8k, hellaswag (or comma-separated)",
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

    # ── augment (data augmentation) ──
    p_aug = sub.add_parser("augment", help="Augment training data to fix weaknesses")
    p_aug.add_argument("data", help="Path to training data (JSONL)")
    p_aug.add_argument("--output", required=True, help="Output augmented data")
    p_aug.add_argument("--type", default="all", help="Comma-separated augmentation types: knowledge, refusal, language, hallucination, persona, all")

    # ── optimize (config recommendation) ──
    p_opt = sub.add_parser("optimize", help="Get training config recommendations")
    p_opt.add_argument("data", help="Path to training data (JSONL)")
    p_opt.add_argument("--lr", type=float, help="Current learning rate")
    p_opt.add_argument("--epochs", type=int, help="Current epochs")
    p_opt.add_argument("--lora-rank", type=int, help="Current LoRA rank")
    p_opt.add_argument("--json", action="store_true")

    # ── validate-hallucination ──
    p_val_h = sub.add_parser("validate-hallucination", help="Check training data for hallucination risks")
    p_val_h.add_argument("data", help="Path to training data (JSONL)")
    p_val_h.add_argument("--json", action="store_true")

    # ── rag-test (RAG-enhanced inference) ──
    p_rag_test = sub.add_parser("rag-test", help="Test model with RAG context")
    p_rag_test.add_argument("model", help="Path to model")
    p_rag_test.add_argument("question", help="Question to ask")
    p_rag_test.add_argument("--store", default="data/rag_store")
    p_rag_test.add_argument("--top-k", type=int, default=5)
    p_rag_test.add_argument("--max-tokens", type=int, default=512)
    p_rag_test.add_argument("--system-prompt", default="")
    p_rag_test.add_argument("--json", action="store_true")

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

    # ── files (project file library housekeeping) ──
    p_files = sub.add_parser("files", help="Manage the per-project file library (trash, etc.)")
    files_sub = p_files.add_subparsers(dest="files_command")
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
                                      "a built RAG corpus, else off)")
    grounding_group.add_argument("--no-rag-grounding", action="store_true",
                                 help="Plain rows only (same as --grounded-share 0)")
    p_ds_build.add_argument("--distractors", type=int, default=0, metavar="0-2",
                            help="Extra other-file chunks added to each grounded row's CONTEXT (default: 0)")
    p_ds_build.add_argument("--seed", type=int, default=42,
                            help="Seed for which rows get grounded (default: 42)")
    p_ds_build.add_argument("--force", action="store_true",
                            help="Export even if some chunks have no usable Q&A (blocked by default)")
    p_ds_build.add_argument("--include-unreviewed-fill", action="store_true",
                            help="Also approve and export the extractive pairs the coverage pass makes for chunks "
                                 "without an approved pair. They are unreviewed text; off by default (they stay pending)")
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

    return parser
