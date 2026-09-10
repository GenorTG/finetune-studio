"""Training engine — the main training loop.

WHAT THIS FILE DOES
==================
The core of the training pipeline. Loads a base model, configures
LoRA (Low-Rank Adaptation), runs supervised fine-tuning (SFT), and
saves the trained adapter.

KEY CONCEPTS
============
- SFT (Supervised Fine-Tuning): training on input-output pairs to
  teach the model a specific behavior.
- LoRA (Low-Rank Adaptation): a parameter-efficient training method.
  Instead of updating all model weights, we add small "adapter"
  matrices and only train those. Much faster, much less memory.
- Unsloth: a library that optimizes LoRA training for speed.
  2-5x faster than vanilla HuggingFace + PEFT.
- Training loop: forward pass → compute loss → backward pass → update
  weights. Repeat for each batch.
- Checkpointing: save the model periodically so we can resume if
  training crashes.
- Progress notification: callbacks to update the UI as training progresses.
"""

import os
import shutil
import threading
import time
from dataclasses import dataclass, field


def _dir_size(path: str) -> int:
    """Sum of file sizes under `path`, in bytes. Missing dir → 0."""
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _human_size(n: int) -> str:
    """1.4 GB / 235 MB / 12 KB style."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} PB"


@dataclass
class TrainingConfig:
    model_path: str = ""
    output_dir: str = "output"
    lora_rank: int = 64
    lora_alpha: int = 128
    learning_rate: float = 8e-5
    num_epochs: int = 4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_steps: int = 20
    weight_decay: float = 0.005
    save_steps: int = 100
    logging_steps: int = 10
    bf16: bool = True
    unsloth: bool = True
    merge_on_save: bool = False  # also save a full standalone model at <output_dir>/merged/
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

@dataclass
class TrainingState:
    status: str = "idle"
    current_step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    epoch: float = 0.0
    elapsed: float = 0.0
    eta: float = 0.0
    message: str = ""
    error: str = ""
    log_lines: list = field(default_factory=list)

class TrainingEngine:
    def __init__(self):
        self.state = TrainingState()
        self.config = TrainingConfig()
        self.current_run_id: str | None = None  # Project-Run id when active
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list = []

    def on_update(self, callback):
        self._callbacks.append(callback)

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb(self.state)
            except Exception:  # noqa: BLE001, S110
                pass

    def start(self, config, training_data, system_prompt=""):
        if self.state.status in ("training", "loading"):
            raise RuntimeError("Training already in progress")
        self.config = config
        self._stop_event.clear()
        self.state = TrainingState(status="loading")
        self._notify()
        self._thread = threading.Thread(
            target=self._train, args=(training_data, system_prompt), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.state.message = "Stopping..."
        self._notify()

    def _train(self, training_data, system_prompt):
        try:
            from finetune_studio.training.data import format_for_sft, split_data
            self.state.status = "loading"
            self.state.message = "Loading model..."
            self._notify()
            formatted = format_for_sft(training_data, system_prompt)
            train_data, _val_data = split_data(formatted)
            self.state.message = f"Training on {len(train_data)} examples..."
            self._notify()
            if self.config.unsloth:
                try:
                    self._train_unsloth(train_data)
                except ImportError:
                    # unsloth not installed — fall back to standard transformers
                    self._train_standard(train_data)
            else:
                self._train_standard(train_data)
            self.state.status = "done"
            self.state.message = "Training complete!"
            self._notify()
        except Exception as e:  # noqa: BLE001
            self.state.status = "error"
            self.state.error = str(e)
            self.state.message = f"Error: {e}"
            self._notify()

    def _train_unsloth(self, train_data):
        import sys
        from datasets import Dataset
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
        cfg = self.config
        self.state.message = "Loading model with Unsloth..."
        self._notify()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_path, max_seq_length=cfg.max_seq_length,
            dtype=None, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.lora_rank, target_modules=cfg.lora_target_modules,
            lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
        )

        # Fix PicklingError: unsloth monkey-patches SFTTrainer/SFTConfig but
        # pickle looks up the original class by module path. Re-patch sys.modules
        # so pickle finds the patched classes.
        import trl.trainer.sft_trainer as _sft_trainer_mod
        import trl.trainer.sft_config as _sft_config_mod
        sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_trainer_mod.SFTTrainer
        sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_config_mod.SFTConfig

        # Import SFTTrainer AFTER patching sys.modules
        from trl import SFTTrainer

        # Monkey-patch Trainer._save to avoid PicklingError when saving
        # training args. Unsloth's class replacement chain breaks pickle.
        import json as _json
        from transformers.trainer import TRAINING_ARGS_NAME
        def _patched_save(self_trainer, output_dir, _internal_call=False):
            import os as _os
            # Save model weights only (skip torch.save(self.args) which pickle-breaks)
            if hasattr(self_trainer.model, 'save_pretrained'):
                self_trainer.model.save_pretrained(output_dir)
            if hasattr(self_trainer, 'tokenizer') and self_trainer.tokenizer is not None:
                self_trainer.tokenizer.save_pretrained(output_dir)
            # Save training_args as JSON (avoids pickle entirely)
            args_path = _os.path.join(output_dir, "training_args.json")
            with open(args_path, "w") as f:
                _json.dump(self_trainer.args.to_dict(), f, indent=2, default=str)
        SFTTrainer._save = _patched_save

        cfg = self.config
        self.state.message = "Loading model with Unsloth..."
        self._notify()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_path, max_seq_length=cfg.max_seq_length,
            dtype=None, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=cfg.lora_rank, target_modules=cfg.lora_target_modules,
            lora_alpha=cfg.lora_alpha, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
        )
        def format_chat(example):
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            return {"text": text}
        dataset = Dataset.from_list(train_data).map(format_chat, remove_columns=list(train_data[0].keys()))
        steps_per_epoch = len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps)
        total = steps_per_epoch * cfg.num_epochs
        self.state.total_steps = total
        # Use save_safetensors=False as safety net for pickle compat
        args = TrainingArguments(
            output_dir=cfg.output_dir, num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate, warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay, logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps, fp16=not cfg.bf16, bf16=cfg.bf16,
            optim="adamw_torch", seed=3407, report_to="none",
        )
        start_time = time.time()
        engine = self
        from transformers import TrainerCallback
        class ProgressCallback(TrainerCallback):
            def on_log(self2, args, state, control, logs=None, **kwargs):
                if logs:
                    engine.state.current_step = state.global_step
                    engine.state.loss = round(logs.get("loss", 0), 4)
                    engine.state.learning_rate = round(logs.get("learning_rate", 0), 8)
                    engine.state.epoch = round(state.epoch or 0, 2)
                    engine.state.elapsed = round(time.time() - start_time, 1)
                    if state.global_step > 0:
                        rate = engine.state.elapsed / state.global_step
                        engine.state.eta = round(rate * (total - state.global_step), 1)
                    engine.state.log_lines.append(
                        f"Step {state.global_step}/{total} | loss={engine.state.loss} | lr={engine.state.learning_rate}"
                    )
                    engine._notify()
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=args, callbacks=[ProgressCallback()],
        )
        self.state.status = "training"
        self._notify()
        trainer.train()
        self.state.status = "saving"
        self.state.message = "Saving model..."
        self._notify()
        os.makedirs(cfg.output_dir, exist_ok=True)
        adapter_dir = os.path.join(cfg.output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        # Persist chat template alongside adapter so the merged model has it.
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                with open(os.path.join(adapter_dir, "chat_template.jinja"), "w") as f:
                    f.write(tokenizer.chat_template)
            except Exception:  # noqa: BLE001
                pass
        if cfg.merge_on_save:
            self._do_merge(model, tokenizer, cfg.output_dir)
        self.state.status = "done"
        self.state.message = "Training complete!"
        self._notify()

    def _train_standard(self, train_data):
        import sys
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        cfg = self.config
        self.state.message = "Loading model..."
        self._notify()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
        lora_config = LoraConfig(
            r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
            target_modules=cfg.lora_target_modules, lora_dropout=0,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        def format_chat(example):
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            return {"text": text}
        dataset = Dataset.from_list(train_data).map(format_chat, remove_columns=list(train_data[0].keys()))
        steps_per_epoch = len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps)
        total = steps_per_epoch * cfg.num_epochs
        self.state.total_steps = total
        # Fix PicklingError: re-patch sys.modules after any unsloth/trl patches
        import trl.trainer.sft_trainer as _sft_trainer_mod
        import trl.trainer.sft_config as _sft_config_mod
        sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_trainer_mod.SFTTrainer
        sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_config_mod.SFTConfig

        from trl import SFTTrainer
        args = TrainingArguments(
            output_dir=cfg.output_dir, num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate, warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay, logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps, fp16=not cfg.bf16, bf16=cfg.bf16,
            optim="adamw_torch", seed=3407, report_to="none",
        )
        start_time = time.time()
        engine = self
        from transformers import TrainerCallback
        class ProgressCallback(TrainerCallback):
            def on_log(self2, args, state, control, logs=None, **kwargs):
                if logs:
                    engine.state.current_step = state.global_step
                    engine.state.loss = round(logs.get("loss", 0), 4)
                    engine.state.learning_rate = round(logs.get("learning_rate", 0), 8)
                    engine.state.epoch = round(state.epoch or 0, 2)
                    engine.state.elapsed = round(time.time() - start_time, 1)
                    if state.global_step > 0:
                        rate = engine.state.elapsed / state.global_step
                        engine.state.eta = round(rate * (total - state.global_step), 1)
                    engine._notify()
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=args, callbacks=[ProgressCallback()],
        )
        self.state.status = "training"
        self._notify()
        trainer.train()
        self.state.status = "saving"
        self._notify()
        os.makedirs(cfg.output_dir, exist_ok=True)
        adapter_dir = os.path.join(cfg.output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                with open(os.path.join(adapter_dir, "chat_template.jinja"), "w") as f:
                    f.write(tokenizer.chat_template)
            except Exception:  # noqa: BLE001
                pass
        if cfg.merge_on_save:
            self._do_merge(model, tokenizer, cfg.output_dir)
        self.state.status = "done"
        self.state.message = "Training complete!"
        self._notify()

    def _do_merge(self, model, tokenizer, output_dir: str) -> dict:
        """Merge the in-memory PEFT adapter into the base model and save to
        `<output_dir>/merged/`. Returns {merged_path, size_bytes, size_human,
        skipped}.

        Sets engine state messages so the UI shows a 'Merging…' step. Idempotent
        by default: if `<output_dir>/merged/` already has files, we skip and
        report the existing size. Test environments can opt out via
        `FTS_SKIP_MERGE=1`.
        """
        merged_dir = os.path.join(output_dir, "merged")
        if os.path.isdir(merged_dir) and os.listdir(merged_dir):
            size = _dir_size(merged_dir)
            self.state.message = "Merged model already exists; skipping."
            self._notify()
            return {"merged_path": merged_dir, "size_bytes": size,
                    "size_human": _human_size(size), "skipped": True}
        os.makedirs(merged_dir, exist_ok=True)
        if os.environ.get("FTS_SKIP_MERGE") == "1":
            self.state.message = "FTS_SKIP_MERGE=1 — skipping merge."
            self._notify()
            return {"merged_path": merged_dir, "size_bytes": 0,
                    "size_human": "0 B", "skipped": True}
        self.state.message = "Merging adapter into full model..."
        self._notify()
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        # Copy chat template if it lives in the adapter dir (Unsloth case).
        src = os.path.join(output_dir, "adapter", "chat_template.jinja")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(merged_dir, "chat_template.jinja"))
        size = _dir_size(merged_dir)
        self.state.message = f"Saved merged model ({_human_size(size)})."
        self._notify()
        return {"merged_path": merged_dir, "size_bytes": size,
                "size_human": _human_size(size), "skipped": False}


def merge_adapter_for_run(run: dict, force: bool = False) -> dict:
    """Merge a persisted run's adapter on disk into a standalone model.

    Loads `run.base_model` + `<run.output_path>/adapter/`, merges via
    PEFT's `merge_and_unload()`, saves to `<run.output_path>/merged/`.

    Idempotent: if `<output_path>/merged/` already has files and `force`
    is False, the existing merge is returned without reloading the model.

    Returns: {merged_path, size_bytes, size_human, skipped, run}.
    Raises ValueError for the obvious pre-conditions (no base_model,
    no output_path, no adapter dir on disk).
    """
    output_path = (run.get("output_path") or "").strip()
    base_model = (run.get("base_model") or "").strip()
    if not output_path:
        raise ValueError("run has no output_path yet; nothing to merge")
    if not base_model:
        raise ValueError("run has no base_model; cannot load base for merging")
    adapter_dir = os.path.join(output_path, "adapter")
    if not os.path.isdir(adapter_dir):
        raise ValueError(f"adapter dir not found on disk: {adapter_dir}")
    merged_dir = os.path.join(output_path, "merged")
    if os.path.isdir(merged_dir) and os.listdir(merged_dir) and not force:
        size = _dir_size(merged_dir)
        return {"merged_path": merged_dir, "size_bytes": size,
                "size_human": _human_size(size), "skipped": True, "run": run}
    if os.environ.get("FTS_SKIP_MERGE") == "1":
        os.makedirs(merged_dir, exist_ok=True)
        # Touch a marker so the caller can tell the merge ran.
        with open(os.path.join(merged_dir, "SKIPPED_BY_TEST"), "w") as f:
            f.write("FTS_SKIP_MERGE=1\n")
        return {"merged_path": merged_dir, "size_bytes": 0,
                "size_human": "0 B", "skipped": True, "run": run}
    os.makedirs(merged_dir, exist_ok=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base, adapter_dir)
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        src = os.path.join(adapter_dir, "chat_template.jinja")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(merged_dir, "chat_template.jinja"))
    finally:
        # Free GPU memory before returning.
        try:
            del base, model, merged  # noqa: F821
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    size = _dir_size(merged_dir)
    return {"merged_path": merged_dir, "size_bytes": size,
            "size_human": _human_size(size), "skipped": False, "run": run}
