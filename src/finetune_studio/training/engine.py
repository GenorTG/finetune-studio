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
- Mixed VRAM/RAM: if the model doesn't fit entirely in VRAM, we fall
  back to device_map="auto" which layers between GPU and CPU memory.
  Slower, but lets you train larger models than your GPU could hold.
"""

import json
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


def _human_size(n: int) -> int:
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
    merge_on_save: bool = False
    export_gguf: bool = False  # also export GGUF at <output_dir>/gguf/
    gguf_quants: list = field(default_factory=lambda: ["f16", "q8_0", "q4_k_m", "q5_k_m"])
    data_path: str = ""  # training data path (for auto-suite generation)
    project_id: str = ""  # project ID (for DB recording)
    # Abliteration settings
    abliterate: bool = False  # remove refusals after training
    abliteration_strength: float = 1.0  # 0.0 = no change, 1.0 = full removal
    # System prompt handling
    # "bake" = prepend to every training example (current behavior)
    # "runtime" = don't bake, save separately for inference-time use
    # "none" = no system prompt at all
    system_prompt_mode: str = "bake"
    # Advanced quantization
    export_gptq: bool = False
    gptq_bits: int = 4
    gptq_group_size: int = 128
    export_imatrix: bool = False
    imatrix_calibration: str = ""
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
        self.current_run_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list = []

    def on_update(self, callback):
        self._callbacks.append(callback)

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb(self.state)
            except Exception:
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
            mode = getattr(self.config, 'system_prompt_mode', 'bake')
            if mode == "none":
                system_prompt = ""
            elif mode == "runtime":
                # Don't bake into training data, but save for later use
                bake_prompt = ""
            else:
                # "bake" — prepend to every training example (current behavior)
                bake_prompt = system_prompt
            formatted = format_for_sft(training_data, bake_prompt if mode != "runtime" else "")
            if mode == "runtime" and system_prompt:
                # Save system prompt to a file alongside the output for later use
                import os
                prompt_path = os.path.join(self.config.output_dir, "system_prompt.txt")
                os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
                with open(prompt_path, "w") as f:
                    f.write(system_prompt)
            train_data, _val_data = split_data(formatted)
            self.state.message = f"Training on {len(train_data)} examples..."
            self._notify()
            if self.config.unsloth:
                try:
                    self._train_unsloth(train_data)
                except ImportError:
                    self._train_standard(train_data)
            else:
                self._train_standard(train_data)
            self.state.status = "done"
            self.state.message = "Training complete!"
            self._notify()
            self._persist_run_output()
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self.state.message = f"Error: {e}"
            self._notify()
            # Persist the error to the DB so the UI can surface it on the run row
            try:
                self._persist_run_error(str(e))
            except Exception:
                pass

    def _persist_run_error(self, error_msg: str):
        """Write the failure reason to the training_runs.error column.

        Called from the except branch in start() so failed runs show the
        real exception in the UI instead of a bare `failed` status.
        """
        if not self.current_run_id:
            return
        try:
            from finetune_studio.db.runs import update_run
            # current_run_id is the full 8-char hex like "bdc217b1" — use it as-is
            update_run(self.current_run_id, status="failed", error=error_msg[:2000])
        except Exception:
            pass

    def _persist_run_output(self):
        """Update the DB run record with the output path so the merge
        endpoint can find the adapter after training completes."""
        if not self.current_run_id or not self.config.output_dir:
            return
        try:
            from finetune_studio.db.runs import update_run
            run_id = self.current_run_id.split("-")[-1] if "-" in self.current_run_id else self.current_run_id
            update_run(run_id, output_path=self.config.output_dir, status="done", final_loss=self.state.final_loss)
        except Exception:
            pass

    def _load_model_with_fallback(self, model_path, tokenizer):
        """Load model with mixed VRAM/RAM fallback.

        Strategy:
        1. Try full GPU offload (device_map={"": 0}) — fastest.
        2. If OOM, retry with device_map="auto" — mixes RAM + VRAM, slower.
        3. If still failing, raise the original error.
        """
        from transformers import AutoModelForCausalLM
        import torch
        # Attempt 1: Full GPU
        try:
            self.state.message = "Loading model on GPU..."
            self._notify()
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype="auto", device_map={"": 0},
                trust_remote_code=True,
            )
            return model
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and "CUDA" not in str(e):
                raise
            self.state.message = (
                f"GPU OOM — retrying with mixed RAM+VRAM (slower)..."
            )
            self._notify()
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        # Attempt 2: Mixed device map
        self.state.message = "Loading model with CPU offload (RAM+VRAM mix)..."
        self._notify()
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
            trust_remote_code=True,
        )

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

        from trl import SFTTrainer

        # Monkey-patch Trainer._save to avoid PicklingError when saving
        # training args. Unsloth's class replacement chain breaks pickle.
        import json as _json
        def _patched_save(self_trainer, output_dir, _internal_call=False):
            import os as _os
            if hasattr(self_trainer.model, 'save_pretrained'):
                self_trainer.model.save_pretrained(output_dir)
            if hasattr(self_trainer, 'tokenizer') and self_trainer.tokenizer is not None:
                self_trainer.tokenizer.save_pretrained(output_dir)
            args_path = _os.path.join(output_dir, "training_args.json")
            with open(args_path, "w") as f:
                _json.dump(self_trainer.args.to_dict(), f, indent=2, default=str)
        SFTTrainer._save = _patched_save

        def format_chat(example):
            text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
            return {"text": text}
        dataset = Dataset.from_list(train_data).map(format_chat, remove_columns=list(train_data[0].keys()))
        steps_per_epoch = len(dataset) // (cfg.batch_size * cfg.gradient_accumulation_steps)
        total = steps_per_epoch * cfg.num_epochs
        self.state.total_steps = total
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
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                with open(os.path.join(adapter_dir, "chat_template.jinja"), "w") as f:
                    f.write(tokenizer.chat_template)
            except Exception:
                pass
        if cfg.merge_on_save:
            self._do_merge(model, tokenizer, cfg.output_dir)
        if cfg.export_gguf:
            self._do_export_gguf(cfg.output_dir)
        if cfg.export_gptq:
            self._do_export_gptq(cfg.output_dir)
        if cfg.export_gptq:
            self._do_export_gptq(cfg.output_dir)
        if cfg.export_imatrix:
            self._do_export_imatrix(cfg.output_dir)
        self._auto_generate_suite()
        # Optional: Abliteration (de-censor)
        if getattr(cfg, 'abliterate', False):
            self._do_abliteration()
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
        model = self._load_model_with_fallback(cfg.model_path, tokenizer)
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
            except Exception:
                pass
        if cfg.merge_on_save:
            self._do_merge(model, tokenizer, cfg.output_dir)
        if cfg.export_gguf:
            self._do_export_gguf(cfg.output_dir)
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
        # Copy system prompt file if runtime mode was used
        prompt_src = os.path.join(output_dir, "system_prompt.txt")
        if os.path.exists(prompt_src):
            shutil.copy(prompt_src, os.path.join(merged_dir, "system_prompt.txt"))
        size = _dir_size(merged_dir)
        self.state.message = f"Saved merged model ({_human_size(size)})."
        self._notify()
        return {"merged_path": merged_dir, "size_bytes": size,
                "size_human": _human_size(size), "skipped": False}

    def _do_abliteration(self) -> dict:
        """Optional: Remove refusal direction from the merged model."""
        merged_dir = os.path.join(self.config.output_dir, "merged")
        if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
            return {"skipped": True, "reason": "no merged model"}
        abliterated_dir = os.path.join(self.config.output_dir, "abliterated")
        self.state.message = "Abliterating (de-censoring) model..."
        self._notify()
        try:
            from finetune_studio.training.abliteration import abliterate_model
            result = abliterate_model(
                model_path=merged_dir,
                output_dir=abliterated_dir,
                strength=getattr(self.config, 'abliteration_strength', 1.0),
            )
            self.state.message = f"Abliteration complete (magnitude: {result.get('refusal_magnitude', 0):.4f})."
            self._notify()
            return result
        except Exception as e:
            self.state.message = f"Abliteration failed: {e}"
            self._notify()
            return {"error": str(e)}

    def _do_export_gptq(self, output_dir: str) -> dict:
        """Export the merged model using GPTQ quantization."""
        merged_dir = os.path.join(output_dir, "merged")
        if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
            return {"skipped": True, "reason": "no merged model"}
        gptq_dir = os.path.join(output_dir, "gptq")
        self.state.message = "Exporting GPTQ..."
        self._notify()
        try:
            from finetune_studio.training.advanced_quant import quantize_gptq
            result = quantize_gptq(
                model_path=merged_dir,
                output_dir=gptq_dir,
                bits=getattr(self.config, 'gptq_bits', 4),
                group_size=getattr(self.config, 'gptq_group_size', 128),
            )
            self.state.message = f"GPTQ exported: {result.get('size_human', 'unknown')}."
            self._notify()
            return result
        except Exception as e:
            self.state.message = f"GPTQ export failed: {e}"
            self._notify()
            return {"error": str(e)}

    def _do_export_imatrix(self, output_dir: str) -> dict:
        """Export the merged model using imatrix-based GGUF quantization."""
        merged_dir = os.path.join(output_dir, "merged")
        if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
            return {"skipped": True, "reason": "no merged model"}
        imatrix_dir = os.path.join(output_dir, "imatrix")
        self.state.message = "Exporting imatrix GGUF..."
        self._notify()
        try:
            from finetune_studio.training.advanced_quant import quantize_gguf_imatrix
            result = quantize_gguf_imatrix(
                model_path=merged_dir,
                output_dir=imatrix_dir,
                imatrix_path=getattr(self.config, 'imatrix_calibration', ''),
                quants=getattr(self.config, 'gguf_quants', ['q4_k_m', 'q5_k_m', 'q8_0']),
            )
            self.state.message = f"Imatrix GGUF exported."
            self._notify()
            return result
        except Exception as e:
            self.state.message = f"Imatrix GGUF export failed: {e}"
            self._notify()
            return {"error": str(e)}
        """Auto-generate a benchmark suite from the training data.

        Saves to `<output_dir>/suite_<name>.json` and records in DB.
        """
        data_path = self.config.get("data_path", "") if hasattr(self.config, "data_path") else ""
        if not data_path or not os.path.isfile(data_path):
            self.state.message = "Auto-suite: no training data found, skipping."
            self._notify()
            return {}
        from finetune_studio.testing.generate_suite import generate_suite_from_training_data
        output_dir = self.config.output_dir
        result = generate_suite_from_training_data(data_path, output_dir)
        if result.get("error"):
            self.state.message = f"Auto-suite: {result['error']}"
            self._notify()
            return result
        # Record in DB
        try:
            from finetune_studio.db.connection import cursor, new_id
            from time import time as _time
            suite_id = new_id()
            project_id = self.config.get("project_id", "") if hasattr(self.config, "project_id") else ""
            with cursor() as c:
                c.execute(
                    "INSERT INTO auto_suites (id, run_id, project_id, suite_name, suite_path, case_count, categories_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (suite_id, self.current_run_id, project_id,
                     result.get("suite_name", "auto"), result.get("suite_path", ""),
                     result.get("case_count", 0),
                     json.dumps(result.get("categories", {})), _time()),
                )
        except Exception:
            pass
        self.state.message = f"Auto-suite: {result.get('case_count', 0)} cases saved."
        self._notify()
        return result

    def _auto_generate_suite(self) -> dict:
        """Auto-generate a benchmark suite from training data."""
        data_path = getattr(self.config, 'data_path', '')
        if not data_path or not os.path.isfile(data_path):
            self.state.message = "Auto-suite: no training data found, skipping."
            self._notify()
            return {}
        from finetune_studio.testing.generate_suite import generate_suite_from_training_data
        output_dir = self.config.output_dir
        result = generate_suite_from_training_data(data_path, output_dir)
        if result.get("error"):
            self.state.message = f"Auto-suite: {result['error']}"
            self._notify()
            return result
        try:
            from finetune_studio.db.connection import cursor, new_id
            from time import time as _time
            suite_id = new_id()
            project_id = getattr(self.config, 'project_id', '')
            with cursor() as c:
                c.execute(
                    "INSERT INTO auto_suites (id, run_id, project_id, suite_name, suite_path, case_count, categories_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (suite_id, self.current_run_id, project_id,
                     result.get("suite_name", "auto"), result.get("suite_path", ""),
                     result.get("case_count", 0),
                     json.dumps(result.get("categories", {})), _time()),
                )
        except Exception:
            pass
        self.state.message = f"Auto-suite: {result.get('case_count', 0)} cases saved."
        self._notify()
        return result

    def _do_export_gguf(self, output_dir: str) -> dict:
        """Export the merged model to GGUF format for llama.cpp.

        Saves to `<output_dir>/gguf/` using llama.cpp's convert.py script.
        If llama.cpp is not available, sets a warning message and returns.

        Quantization types supported: f16, bf16, q8_0, q4_k_m, q5_k_m, q4_0,
        q4_1, q5_0, q5_1, q2_k, q3_k_m, q3_k_l, q3_k_s, q4_k_s, q5_k_s,
        q6_k, iq2_xxs, iq2_xs, iq2_s, iq2_m, iq3_xxs, iq3_xs, iq3_s,
        iq3_m, iq4_nl, iq4_xs, q4_0_4_4, q4_0_4_8, q4_0_8_8.
        """
        gguf_dir = os.path.join(output_dir, "gguf")
        merged_dir = os.path.join(output_dir, "merged")
        if not os.path.isdir(merged_dir) or not os.listdir(merged_dir):
            return {"gguf_path": "", "skipped": True,
                    "reason": "no merged model to convert"}
        os.makedirs(gguf_dir, exist_ok=True)
        # Find llama.cpp convert script
        convert_script = None
        candidates = [
            os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py"),
            os.path.expanduser("~/llama.cpp/convert.py"),
            os.path.expanduser("~/llama.cpp/convert-hf-to-gguf.py"),
            "/usr/local/bin/convert-hf-to-gguf.py",
        ]
        for c in candidates:
            if os.path.isfile(c):
                convert_script = c
                break
        if not convert_script:
            self.state.message = "GGUF export: llama.cpp not found. Install llama.cpp to enable GGUF export."
            self._notify()
            return {"gguf_path": gguf_dir, "skipped": True,
                    "reason": "llama.cpp not found"}
        # Find llama.cpp quantize binary
        quant_bin = None
        quant_candidates = [
            os.path.join(os.path.dirname(convert_script), "..", "build", "bin", "llama-quantize"),
            os.path.join(os.path.expanduser("~"), "llama.cpp", "build", "bin", "llama-quantize"),
            os.path.join(os.path.expanduser("~"), "llama.cpp", "build-RPC", "bin", "llama-quantize"),
            "llama-quantize",
        ]
        for q in quant_candidates:
            if os.path.isfile(q):
                quant_bin = q
                break
        try:
            import subprocess
            # Step 1: Convert to F16 GGUF first
            f16_file = os.path.join(gguf_dir, "model-f16.gguf")
            cmd = ["python3", convert_script, merged_dir, "--outfile", f16_file,
                   "--outtype", "f16"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                self.state.message = f"GGUF export failed: {result.stderr[:200]}"
                self._notify()
                return {"gguf_path": gguf_dir, "skipped": True,
                        "reason": result.stderr[:200]}
            # Step 2: Quantize to all requested formats
            quants = self.config.gguf_quants if hasattr(self.config, 'gguf_quants') else ["f16"]
            exported = {}
            for quant in quants:
                quant = quant.lower().replace("-", "_").replace(".", "_")
                if quant in ("f16", "bf16"):
                    # Already exported as f16
                    src = f16_file
                    dst = os.path.join(gguf_dir, f"model-{quant}.gguf")
                    if quant == "f16":
                        exported[quant] = {"path": f16_file, "size": os.path.getsize(f16_file)}
                    else:
                        import shutil
                        shutil.copy(f16_file, dst)
                        exported[quant] = {"path": dst, "size": os.path.getsize(dst)}
                    continue
                # Use llama.cpp quantize binary
                quant_bin = os.path.join(os.path.dirname(convert_script), "quantize")
                if not os.path.isfile(quant_bin):
                    # Try system-wide
                    quant_bin = "quantize"
                out_file = os.path.join(gguf_dir, f"model-{quant}.gguf")
                cmd = [quant_bin, f16_file, out_file, quant]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
                if result.returncode == 0 and os.path.isfile(out_file):
                    exported[quant] = {"path": out_file, "size": os.path.getsize(out_file)}
                else:
                    exported[quant] = {"error": result.stderr[:100]}
            size = _dir_size(gguf_dir)
            # Copy runtime system prompt file if present
            prompt_src = os.path.join(output_dir, "merged", "system_prompt.txt")
            if os.path.exists(prompt_src):
                import shutil
                shutil.copy(prompt_src, os.path.join(gguf_dir, "system_prompt.txt"))
            self.state.message = f"GGUF exported ({_human_size(size)}, {len(exported)} formats)."
            self._notify()
            return {"gguf_path": gguf_dir, "size_bytes": size,
                    "size_human": _human_size(size), "skipped": False,
                    "exported": exported}
        except Exception as e:
            self.state.message = f"GGUF export error: {e}"
            self._notify()
            return {"gguf_path": gguf_dir, "skipped": True, "reason": str(e)}


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
        with open(os.path.join(merged_dir, "SKIPPED_BY_TEST"), "w") as f:
            f.write("FTS_SKIP_MERGE=1\n")
        return {"merged_path": merged_dir, "size_bytes": 0,
                "size_human": "0 B", "skipped": True, "run": run}
    os.makedirs(merged_dir, exist_ok=True)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    # Load base in full precision (strip any quantization config)
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", trust_remote_code=True,
    )
    try:
        model = PeftModel.from_pretrained(base, adapter_dir)
        merged = model.merge_and_unload()
        # Strip quantization config from merged model
        if hasattr(merged, 'config') and hasattr(merged.config, 'quantization_config'):
            merged.config.quantization_config = None
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        src = os.path.join(adapter_dir, "chat_template.jinja")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(merged_dir, "chat_template.jinja"))
        # Also carry runtime system prompt file if it exists
        prompt_src = os.path.join(os.path.dirname(adapter_dir), "system_prompt.txt")
        if os.path.exists(prompt_src):
            shutil.copy(prompt_src, os.path.join(merged_dir, "system_prompt.txt"))
    finally:
        try:
            del base, model, merged
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    size = _dir_size(merged_dir)
    return {"merged_path": merged_dir, "size_bytes": size,
            "size_human": _human_size(size), "skipped": False, "run": run}
