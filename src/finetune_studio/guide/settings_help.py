"""Plain-language help for every prefillable setting, backing ``explain_setting``.

Text mirrors the page tooltips and the preset advisor's rules (training/preset_advisor.py);
``tests/test_guide_state_tools.py`` fails when a registry field has no entry here.
"""
from __future__ import annotations

from typing import Any

from finetune_studio.guide.registry import FIELDS, Field

SETTING_HELP: dict[str, dict[str, str]] = {
    "training_mode": {
        "meaning": "Which training route to use: sft, dpo, tool_sft, continued_pretraining or reasoning_distillation.",
        "guidance": "Pick from the data shape. Q&A pairs from documents → sft. Reviewed chosen/rejected answers → dpo. Tool traces with schemas → tool_sft. Raw domain text → continued_pretraining. Reviewed teacher traces → reasoning_distillation.",
        "pitfall": "DPO sets learning rate 1e-6, 1 epoch and no warmup on the page; SFT-style values would be far too aggressive for it.",
        "kb": "training-routes",
    },
    "preset": {
        "meaning": "Effort tier that fills epochs, rank, alpha, learning rate, batch and accumulation via the preset advisor.",
        "guidance": "Smoke proves the plumbing (minutes, no recall). Baseline (balanced) is the recommended start. Precision is the 95%-strict-recall evidence class for factual datasets. Overkill only after Precision fails on a verified dataset.",
        "pitfall": "The advisor's optimizer-step count is an upper bound (it ignores the validation split).",
        "kb": "training-settings",
    },
    "num_epochs": {
        "meaning": "How many passes over the whole training set.",
        "guidance": "Think in optimizer steps: steps = pairs × epochs ÷ (batch size × gradient accumulation). Tier floors: Baseline 400, Precision 700, Overkill 1000. Small datasets need more epochs, big ones fewer (the advisor scales by √(500 ÷ pairs)).",
        "pitfall": "Raising epochs on a tiny dataset reaches the step floor by memorising, not learning; add data instead.",
        "kb": "training-settings",
    },
    "lora_rank": {
        "meaning": "Size of the trainable LoRA adapter. Higher = more capacity, slower, bigger adapter.",
        "guidance": "64 is a good default; the advisor halves it for bases ≤1B and uses 128 for Precision, 256 for Overkill. Lower (8–16) only when memory-constrained.",
        "pitfall": "High rank on a small dataset adds capacity you cannot fill.",
        "kb": "training-settings",
    },
    "lora_alpha": {
        "meaning": "How strongly the LoRA adapter's changes are applied (scale = alpha / rank).",
        "guidance": "Twice the rank is the usual choice and what the advisor proposes; keep the ratio when you change the rank.",
        "pitfall": "Raising the rank without the alpha weakens every update (the scale falls).",
        "kb": "training-settings",
    },
    "learning_rate": {
        "meaning": "Step size of each weight update.",
        "guidance": "For LoRA 1e-4 to 2e-4 is typical; the advisor uses 2e-4 up to 7B, 1e-4 up to 15B, 8e-5 above. DPO uses 1e-6.",
        "pitfall": "Too high overshoots and the loss spikes; too low and the run never learns in the planned steps.",
        "kb": "training-settings",
    },
    "batch_size": {
        "meaning": "Examples processed per step on the GPU.",
        "guidance": "1–4 is typical on consumer GPUs. Larger batches are more stable but use more VRAM. Effective batch = batch size × gradient accumulation.",
        "pitfall": "An out-of-memory failure is fixed by lowering batch size or max sequence first.",
        "kb": "training-settings",
    },
    "gradient_accumulation_steps": {
        "meaning": "Number of steps whose gradients are summed before one weight update.",
        "guidance": "Raises the effective batch without more VRAM. The form default is 1; presets use 1 (Smoke) or 4.",
        "pitfall": "More accumulation means fewer optimizer steps for the same data.",
        "kb": "training-settings",
    },
    "max_seq_length": {
        "meaning": "Maximum tokens per training example (prompt + response); longer ones are truncated.",
        "guidance": "2048 suits Q&A pairs; raise only if your examples are long. Cost grows with length.",
        "pitfall": "A large value on a small GPU is a common cause of out-of-memory.",
        "kb": "training",
    },
    "warmup_steps": {
        "meaning": "Steps over which the learning rate ramps up from zero.",
        "guidance": "The advisor uses about 8% of the run (1–100). The form default is 100; training caps any value at 10% of the run's steps (noted in the step log). DPO uses 0.",
        "pitfall": "Without the cap a fixed 100 would swallow most of a short run.",
        "kb": "training-settings",
    },
    "eval_steps": {
        "meaning": "Measure loss on the held-out split every N steps; 0 disables evaluation.",
        "guidance": "Small datasets need a small value or no evaluation happens before training ends. Required for early stopping.",
        "pitfall": "Early stopping with eval every 0 steps never triggers.",
        "kb": "training",
    },
    "early_stopping": {
        "meaning": "Stop when validation loss has not improved for 3 evaluations and keep the best checkpoint.",
        "guidance": "Useful on larger datasets. On tiny datasets make sure it did not end the run before the planned steps.",
        "pitfall": "Needs 'Eval every N steps' above 0.",
        "kb": "training",
    },
    "merge_on_save": {
        "meaning": "After training, merge the LoRA adapter into the base model as a standalone model.",
        "guidance": "Ticked by default; Testing's 'auto (latest merged)' uses it. Untick to keep only the adapter and merge at export time.",
        "pitfall": "The merged model is roughly the base model's size on disk.",
        "kb": "training",
    },
    "system_prompt_mode": {
        "meaning": "How the project's system prompt is handled: bake (trained in), runtime (saved as system_prompt.txt), none.",
        "guidance": "Bake makes the model always behave that way.",
        "pitfall": "Baked prompts cannot be changed without retraining.",
        "kb": "training",
    },
    "api_concurrency": {
        "meaning": "How many calls an API connection serves at once (Settings → AI assistant → the connection's parallel calls, default 8).",
        "guidance": "8 mines a 19-file corpus in minutes; raise it if the provider allows more parallel requests.",
        "pitfall": "Too high and the provider answers 429 (rate limit): calls are retried, but a mining round can still fail.",
        "kb": "helper-model",
    },
    "grounded": {
        "meaning": "Add the pair's own source passage (retrieved context) to a share of exported rows.",
        "guidance": "Only matters if you will chat with the trained model over this project's RAG corpus; build RAG first.",
        "pitfall": "Grounded rows teach answering from context, not memorisation; never score them asked bare.",
        "kb": "dataset-quality",
    },
    "grounded_pct": {
        "meaning": "Percent of exported rows that carry retrieved context (default 40).",
        "guidance": "40% is the default when a RAG corpus exists.",
        "pitfall": "100% teaches the model to depend on context it will not always have.",
        "kb": "dataset-quality",
    },
    "embedder": {
        "meaning": "Embedding model that turns chunks into vectors for retrieval.",
        "guidance": "multilingual-e5-large (default) for mixed languages; bge-large-en for English-only; all-MiniLM-L6-v2 is small and weaker.",
        "pitfall": "Changing it requires a rebuild; the first build downloads it (~2.2 GB).",
        "kb": "brain",
    },
    "fact_cards": {
        "meaning": "Also index every approved Q&A pair as its own short card next to the passages of its file.",
        "guidance": "Off by default; turn it on once many pairs are reviewed and check the Smart Brain on the test page.",
        "pitfall": "Only approved pairs go in; approving or editing pairs later shows 'rebuild needed'.",
        "kb": "brain",
    },
    "rerank_enabled": {
        "meaning": "Re-score retrieved candidates with a cross-encoder.",
        "guidance": "More accurate, slower; disable for fast lookup-only use.",
        "pitfall": "Adds latency on every query.",
        "kb": "brain",
    },
    "hybrid_enabled": {
        "meaning": "Combine embedding search with BM25 keyword search, then rerank.",
        "guidance": "Better recall: embeddings catch paraphrases, BM25 catches exact names and codes.",
        "pitfall": "None significant; leave on unless debugging retrieval.",
        "kb": "brain",
    },
    "rerank_top_n": {
        "meaning": "How many candidates the reranker reads (default 50).",
        "guidance": "30–100 is typical.",
        "pitfall": "Higher is slower for small gains.",
        "kb": "brain",
    },
    "eval_kind": {
        "meaning": "Held-out validation (the 10% slice the trainer never saw) or the full training set (memorisation check).",
        "guidance": "Held-out is the honest generalisation number.",
        "pitfall": "The full-training-set score is recall, not quality.",
        "kb": "testing",
    },
    "epochs": {
        "meaning": "Quick work's epochs override; empty uses the preset's own epoch count.",
        "guidance": "Same field as the Training page's num_epochs; see optimizer-step rule.",
        "pitfall": "Leaving a stale override from an earlier run.",
        "kb": "wizard",
    },
    "n_ctx": {
        "meaning": "Context length (KV cache size) the model loads with; 0 = auto: the model's native window.",
        "guidance": "Leave it at 0. Auto lowers the window only when the whole model cannot fit on the GPU (never below 32768); an explicit value is never lowered, GPU layers step down instead.",
        "pitfall": "A huge explicit context on a nearly full GPU pushes layers to the CPU and slows generation.",
        "kb": "inference",
    },
}

_ALIASES = {
    "lr": "learning_rate", "rank": "lora_rank", "alpha": "lora_alpha", "lora": "lora_rank", "epochs": "num_epochs",
    "epoch": "num_epochs", "batch": "batch_size", "accumulation": "gradient_accumulation_steps",
    "grad_accum": "gradient_accumulation_steps", "sequence": "max_seq_length", "max_seq": "max_seq_length",
    "max_sequence": "max_seq_length", "route": "training_mode", "mode": "training_mode",
    "tier": "preset", "warmup": "warmup_steps", "merge": "merge_on_save", "context": "n_ctx",
    "context_length": "n_ctx", "facts": "fact_cards",
    "early_stop": "early_stopping", "eval": "eval_steps",
}


def _key(name: str) -> str:
    return "_".join(str(name).strip().lower().replace("-", " ").split())


def explain_setting(name: str) -> dict[str, Any]:
    """Meaning, guidance and pitfall for one setting; unknown names list what is known."""
    key = _key(name)
    key = _ALIASES.get(key, key)
    help_row = SETTING_HELP.get(key)
    if help_row is None:
        return {"error": f"unknown setting {name!r}", "known": sorted(SETTING_HELP)}
    pages = sorted(page for (page, field_name) in FIELDS if field_name == key)
    spec: Field | None = FIELDS.get((pages[0], key)) if pages else None
    out: dict[str, Any] = {"setting": key, "label": spec.label if spec else key, "pages": pages, **help_row}
    if spec is not None:
        if spec.choices:
            out["allowed"] = list(spec.choices)
        if spec.minimum is not None or spec.maximum is not None:
            out["range"] = [spec.minimum, spec.maximum]
        if spec.note:
            out["note"] = spec.note
    return out
