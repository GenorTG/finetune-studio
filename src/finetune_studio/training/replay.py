"""Replay: keep general chat ability while an adapter learns facts.

Fact SFT costs instruction following: in Thinking Machines' company-docs run IFEval fell 85 -> 45 and a 70/30 mix with
chat data kept most of it (On-Policy Distillation, 2025-10-27); on Korvane a 3-epoch fact adapter dropped IFEval from
83.3 to 70.0 % (paired p = 0.004, RESULTS §22). Replay mixes in general prompts answered by THE BASE MODEL ITSELF,
before the adapter exists: the model rehearses its own behaviour, so the mix teaches no foreign style and needs no
second model. It works for any chat model the trainer can load.

Prompts come from databricks/databricks-dolly-15k (CC BY-SA 3.0, downloaded into the benchmark HF cache at run time,
never shipped). The answers are cached per (base model, system prompt, prompt set): later runs on the same base reuse
them, so only the first run pays for generation. Answers that hit the token limit, come back empty or carry a thinking
trace are dropped, never truncated into training rows.
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

PROMPT_DATASET = "databricks/databricks-dolly-15k"
PROMPT_SET_VERSION = 1
MAX_REPLAY_ROWS = 4000
DEFAULT_MAX_NEW_TOKENS = 384
_CONTEXT_CATEGORIES = frozenset({"closed_qa", "information_extraction", "summarization"})
_THINK = re.compile(r"<think>|</think>", re.IGNORECASE)

ProgressFn = Callable[[int, int], None]


def replay_row_count(train_rows: int, fraction: float, cap: int = MAX_REPLAY_ROWS) -> int:
    """Replay rows so they make up ``fraction`` of the final mix (0.3 -> 30 % replay), capped."""
    if not 0 <= fraction < 1:
        raise ValueError(f"replay_fraction must be in [0, 1), got {fraction}")
    if fraction == 0 or train_rows <= 0:
        return 0
    return min(cap, round(train_rows * fraction / (1 - fraction)))


def load_prompts(n: int, *, seed: int = 0, cache_dir: str = "data/benchmarks/hf_cache",
                 loader: Callable[..., Any] | None = None) -> list[str]:
    """``n`` seeded Dolly instructions (with their context where the task needs it)."""
    if loader is None:
        from datasets import (
            load_dataset as loader,
        )
    rows = loader(PROMPT_DATASET, split="train", cache_dir=cache_dir)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    out: list[str] = []
    for i in order[: n * 2]:  # headroom for the rows skipped below
        row = rows[i]
        instruction = str(row.get("instruction") or "").strip()
        context = str(row.get("context") or "").strip()
        if not instruction:
            continue
        if row.get("category") in _CONTEXT_CATEGORIES:
            if not context:
                continue
            instruction = f"{instruction}\n\n{context}"
        out.append(instruction)
        if len(out) == n:
            break
    return out


def _cache_path(model_path: str, system_prompt: str, seed: int) -> Path:
    from finetune_studio.data.fs.paths import root

    key = json.dumps([str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
                      system_prompt, seed, PROMPT_SET_VERSION])
    return root() / "replay" / f"{hashlib.sha256(key.encode()).hexdigest()[:20]}.jsonl"


def generate_answers(model: Any, tokenizer: Any, prompts: list[str], *, system_prompt: str = "",
                     max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, batch_size: int = 8,
                     progress: ProgressFn | None = None) -> list[tuple[str, str]]:
    """Greedy answers of ``model`` (the untrained base) to ``prompts``; dropped ones are left out."""
    import torch

    from finetune_studio.training.formatting import (
        chat_stop_token_ids,
        render_chat_text,
        with_system_prompt,
    )

    stop = chat_stop_token_ids(tokenizer, getattr(model, "generation_config", None))
    side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    was_training = model.training
    model.eval()
    out: list[tuple[str, str]] = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            texts = [render_chat_text(tokenizer, with_system_prompt([{"role": "user", "content": p}], system_prompt),
                                      generation=True, enable_thinking=False) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop or None,
                                     pad_token_id=tokenizer.pad_token_id, use_cache=True)
            for prompt, ids in zip(batch, gen[:, enc["input_ids"].shape[1]:], strict=True):
                ids = ids.tolist()
                ended = any(t in stop for t in ids)
                answer = tokenizer.decode([t for t in ids if t not in stop and t != tokenizer.pad_token_id],
                                          skip_special_tokens=True).strip()
                if ended and answer and not _THINK.search(answer):
                    out.append((prompt, answer))
            if progress:
                progress(min(start + batch_size, len(prompts)), len(prompts))
    finally:
        tokenizer.padding_side = side
        if was_training:
            model.train()
    return out


def replay_rows(model: Any, tokenizer: Any, *, model_path: str, n: int, system_prompt: str = "", seed: int = 0,
                progress: ProgressFn | None = None, prompt_loader: Callable[..., Any] | None = None,
                batch_size: int = 16) -> list[dict[str, Any]]:
    """``n`` replay rows (``{"messages": [...]}``), from the cache when it already holds enough for this base."""
    if n <= 0:
        return []
    cache = _cache_path(model_path, system_prompt, seed)
    cached: list[dict[str, Any]] = []
    if cache.is_file():
        cached = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines() if line.strip()]
    tried: set[str] = set()
    for _round in range(3):  # answers that hit the token limit are dropped: top up in up to 3 rounds
        missing = n - len(cached)
        if missing <= 0:
            break
        done = {r["messages"][-2]["content"] for r in cached}
        want = int(missing * 1.5) + 8
        prompts = load_prompts(len(done) + len(tried) + want, seed=seed, loader=prompt_loader)
        todo = [p for p in prompts if p not in done and p not in tried][:want]
        if not todo:
            break
        tried.update(todo)
        fresh = generate_answers(model, tokenizer, todo, system_prompt=system_prompt, progress=progress,
                                 batch_size=batch_size)
        cached += [{"messages": [*([{"role": "system", "content": system_prompt}] if system_prompt else []),
                                 {"role": "user", "content": p}, {"role": "assistant", "content": a}]}
                   for p, a in fresh]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cached), encoding="utf-8")
        log.info("replay: %d of %d answers kept, cache %s holds %d", len(fresh), len(todo), cache, len(cached))
    return cached[:n]
