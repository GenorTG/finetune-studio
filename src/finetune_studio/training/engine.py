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
import logging
import math
import os
import shutil

# Training runs inside the threaded uvicorn server with CUDA initialised, so
# any forked Dataset.map / tokenizer pool deadlocks (E2E-27: 8 workers stuck in
# futex/pipe_read, run frozen at "Loading model…"). Unsloth's own escape hatch
# "0" forces in-process tokenization; users can still override via env.
os.environ.setdefault("UNSLOTH_DATASET_NUM_PROC", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def _format_exc(exc: BaseException) -> str:
    """``Type: msg`` without a trailing empty ``: `` when msg is blank."""
    return f"{type(exc).__name__}: {exc}".rstrip(": ")


def _merged_dir_complete(merged_dir: str) -> bool:
    """True when merged/ has weight files (not just a partial config dump)."""
    if not os.path.isdir(merged_dir):
        return False
    for name in os.listdir(merged_dir):
        if name.endswith((".safetensors", ".bin")):
            return True
    return False


def _free_cuda() -> None:
    """Drop refs the caller already deleted and clear the CUDA cache."""
    try:
        import gc
        gc.collect()
    except Exception:  # noqa: BLE001, S110
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001, S110
        pass


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

class _ThreadChild:
    """Duck-typed process handle wrapping a thread (test hook only)."""

    def __init__(self, thread: threading.Thread) -> None:
        self._thread = thread
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)
        if not self._thread.is_alive():
            self.exitcode = 0

    def terminate(self) -> None:
        # Threads cannot be hard-killed; cooperative stop + join is the test path.
        self.exitcode = -15

    def kill(self) -> None:
        self.exitcode = -9


class TrainingEngine:
    def __init__(self):
        self.state = TrainingState()
        self.config = TrainingConfig()
        self.current_run_id: str | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._callbacks: list = []
        # E2E-40: training runs in a spawn child so uvicorn never imports unsloth.
        self._process: object | None = None
        self._mp_stop: object | None = None
        self._out_queue: object | None = None
        self._listener: threading.Thread | None = None

    def on_update(self, callback):
        self._callbacks.append(callback)

    def _notify(self):
        for cb in self._callbacks:
            try:
                cb(self.state)
            except Exception:
                pass

    def start(self, config, training_data, system_prompt="", *, _worker_target=None):
        """Start training in a spawn child process (never in the uvicorn process).

        ``_worker_target`` is a test hook: when set, the target runs in a daemon
        thread (same Queue protocol) so callables need not be picklable. Production
        always uses ``multiprocessing`` spawn + ``training.worker.training_worker``.
        """
        if self.state.status in ("training", "loading", "saving"):
            raise RuntimeError("Training already in progress")
        if self._process is not None and getattr(self._process, "is_alive", lambda: False)():
            raise RuntimeError("Training already in progress")

        import multiprocessing as mp
        import queue as queue_mod

        from finetune_studio.training.worker import config_to_dict, training_worker

        self.config = config
        self._stop_event.clear()
        self.state = TrainingState(status="loading", message="Starting training worker...")
        self._notify()

        cfg_dict = config_to_dict(config)
        if _worker_target is not None:
            # In-process fake child for unit tests (identical message protocol).
            self._out_queue = queue_mod.Queue()
            self._mp_stop = threading.Event()
            target = _worker_target
            thread = threading.Thread(
                target=target,
                args=(cfg_dict, training_data, system_prompt, self._out_queue, self._mp_stop),
                daemon=True,
                name="fts-training-worker-test",
            )
            self._process = _ThreadChild(thread)
            thread.start()
        else:
            ctx = mp.get_context("spawn")
            self._out_queue = ctx.Queue()
            self._mp_stop = ctx.Event()
            self._process = ctx.Process(
                target=training_worker,
                args=(
                    cfg_dict,
                    training_data,
                    system_prompt,
                    self._out_queue,
                    self._mp_stop,
                ),
                daemon=True,
                name="fts-training-worker",
            )
            self._process.start()
        self._listener = threading.Thread(
            target=self._listen_child, daemon=True, name="fts-training-listener",
        )
        self._listener.start()

    def _apply_state_dict(self, payload: dict) -> None:
        """Copy a child state snapshot onto ``self.state`` and notify parents."""
        for key in (
            "status", "current_step", "total_steps", "loss", "learning_rate",
            "epoch", "elapsed", "eta", "message", "error",
        ):
            if key in payload:
                setattr(self.state, key, payload[key])
        if "log_lines" in payload and isinstance(payload["log_lines"], list):
            self.state.log_lines = list(payload["log_lines"])
        self._notify()

    def _listen_child(self) -> None:
        """Drain the child queue until ``op=done`` or the process exits."""
        import queue as queue_mod

        q = self._out_queue
        proc = self._process
        if q is None:
            return
        while True:
            try:
                msg = q.get(timeout=0.5)
            except queue_mod.Empty:
                if proc is not None and not getattr(proc, "is_alive", lambda: False)():
                    break
                continue
            except Exception:  # noqa: BLE001
                if proc is not None and not getattr(proc, "is_alive", lambda: False)():
                    break
                continue
            if not isinstance(msg, dict):
                continue
            op = msg.get("op")
            if op == "state":
                payload = msg.get("state") or {}
                if isinstance(payload, dict):
                    self._apply_state_dict(payload)
            elif op == "done":
                break
        # If the child died without a clean terminal status, surface it.
        if self.state.status in ("loading", "training", "saving", "running"):
            exitcode = getattr(proc, "exitcode", None) if proc is not None else None
            if self._stop_event.is_set() or (
                self._mp_stop is not None and getattr(self._mp_stop, "is_set", lambda: False)()
            ):
                self.state.status = "stopped"
                self.state.message = "Stopped by user"
                self._notify()
            elif exitcode not in (0, None):
                self.state.status = "error"
                self.state.error = self.state.error or f"Training worker exited with code {exitcode}"
                self.state.message = self.state.error
                self._notify()
        self._cleanup_child_handles()

    def _cleanup_child_handles(self) -> None:
        proc = self._process
        if proc is not None:
            try:
                if getattr(proc, "is_alive", lambda: False)():
                    proc.join(timeout=0.1)
            except Exception:  # noqa: BLE001, S110
                pass
        self._process = None
        self._mp_stop = None
        self._out_queue = None
        self._listener = None

    def stop(self) -> None:
        """Request cooperative stop, then terminate the child if it hangs."""
        self._stop_event.set()
        if self._mp_stop is not None:
            try:
                self._mp_stop.set()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S110
                pass
        self.state.message = "Stopping..."
        self._notify()

        proc = self._process
        if proc is None:
            if self.state.status not in ("done", "error", "stopped", "idle"):
                self._mark_stopped()
            return

        # Cooperative window for TrainerCallback / phase gates.
        try:
            proc.join(timeout=3.0)  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001, S110
            pass
        if getattr(proc, "is_alive", lambda: False)():
            try:
                proc.terminate()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                proc.join(timeout=3.0)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S110
                pass
        if getattr(proc, "is_alive", lambda: False)():
            try:
                proc.kill()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S110
                pass
            try:
                proc.join(timeout=1.0)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001, S110
                pass

        if self.state.status not in ("done", "error", "stopped"):
            self._mark_stopped()

    def _stop_requested(self) -> bool:
        if self._stop_event.is_set():
            return True
        if self._mp_stop is not None and getattr(self._mp_stop, "is_set", lambda: False)():
            return True
        return False

    def _mark_stopped(self) -> None:
        """Terminal state when the user hits Stop (adapter may still be valid)."""
        self.state.status = "stopped"
        self.state.message = "Stopped by user"
        self._notify()
        if not self.current_run_id:
            return
        try:
            from finetune_studio.db.runs import update_run
            fields: dict = {
                "status": "stopped",
                "notes": "Stopped by user",
            }
            if self.config.output_dir and os.path.isdir(
                os.path.join(self.config.output_dir, "adapter")
            ):
                fields["output_path"] = self.config.output_dir
            update_run(self.current_run_id, **fields)
        except Exception:  # noqa: BLE001
            pass

    def _maybe_merge(self, model: object, tokenizer: object, output_dir: str) -> None:
        """Run merge; failures are non-fatal (adapter on disk is still valid)."""
        try:
            self._do_merge(model, tokenizer, output_dir)
        except Exception as e:
            merge_msg = _format_exc(e)
            note = f"Training complete — merge failed: {merge_msg}"
            log.exception("Merge failed after training; adapter remains at %s", output_dir)
            self.state.message = note
            self.state.error = note
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
                prompt_path = os.path.join(self.config.output_dir, "system_prompt.txt")
                os.makedirs(os.path.dirname(prompt_path) or ".", exist_ok=True)
                os.makedirs(self.config.output_dir, exist_ok=True)
                with open(prompt_path, "w") as f:
                    f.write(system_prompt)
            train_data, _val_data = split_data(formatted)
            if self._stop_requested():
                self._mark_stopped()
                return
            self.state.message = f"Training on {len(train_data)} examples..."
            self._notify()
            if self.config.unsloth:
                try:
                    self._train_unsloth(train_data)
                except ImportError:
                    self._train_standard(train_data)
            else:
                self._train_standard(train_data)
            if self.state.status == "stopped":
                return
            if self.state.status not in ("error",):
                self.state.status = "done"
                if not (self.state.message or "").startswith("Training complete — merge failed"):
                    self.state.message = "Training complete!"
                self._notify()
                self._persist_run_output()
        except Exception as e:
            msg = _format_exc(e)
            log.exception("Training failed: %s", msg)
            self.state.status = "error"
            self.state.error = msg
            self.state.message = msg
            self._notify()
            # Persist the error to the DB so the UI can surface it on the run row
            try:
                self._persist_run_error(msg)
            except Exception:
                pass

    def _persist_run_error(self, error_msg: str) -> None:
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

    def _persist_run_output(self) -> None:
        """Update the DB run record with the output path so the merge
        endpoint can find the adapter after training completes."""
        if not self.current_run_id or not self.config.output_dir:
            return
        try:
            from finetune_studio.db.runs import update_run
            run_id = self.current_run_id.split("-")[-1] if "-" in self.current_run_id else self.current_run_id
            fields: dict = {
                "output_path": self.config.output_dir,
                "status": "done",
            }
            final_loss = getattr(self.state, "final_loss", None)
            if final_loss is not None:
                fields["final_loss"] = final_loss
            # Merge (or post-train export) failed soft — keep status done, surface note.
            err = (self.state.error or "").strip()
            if err:
                fields["error"] = err[:2000]
                fields["notes"] = err[:2000]
            update_run(run_id, **fields)
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
        from unsloth import FastLanguageModel

        from finetune_studio.training.sft_args import build_sft_args_from_config
        cfg = self.config
        self.state.message = "Loading model with Unsloth..."
        self._notify()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cfg.model_path, max_seq_length=cfg.max_seq_length,
            dtype=None, load_in_4bit=True,
        )
        if self._stop_requested():
            self._mark_stopped()
            return
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
        if self._stop_requested():
            self._mark_stopped()
            return
        denom = max(1, cfg.batch_size * cfg.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(len(dataset) / denom))
        total = steps_per_epoch * cfg.num_epochs
        self.state.total_steps = total
        # SFTConfig (not TrainingArguments): avoids TRL KeyError push_to_hub_token.
        args = build_sft_args_from_config(cfg)
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
        class StopCallback(TrainerCallback):
            def on_step_end(self2, args, state, control, **kwargs):
                if engine._stop_requested():
                    control.should_training_stop = True
                return control
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=args, callbacks=[ProgressCallback(), StopCallback()],
        )
        self.state.status = "training"
        self.state.message = "Training…"
        self._notify()
        trainer.train()
        if self._stop_requested():
            # Persist partial adapter so the work is not lost, then stop.
            try:
                os.makedirs(cfg.output_dir, exist_ok=True)
                adapter_dir = os.path.join(cfg.output_dir, "adapter")
                model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
            except Exception:
                log.exception("Failed to save adapter after stop")
            self._mark_stopped()
            return
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
        if self._stop_requested():
            self._mark_stopped()
            return
        if cfg.merge_on_save:
            self._maybe_merge(model, tokenizer, cfg.output_dir)
        if cfg.export_gguf:
            try:
                self._do_export_gguf(cfg.output_dir)
            except Exception as e:
                log.exception("GGUF export failed (non-fatal)")
                self.state.message = (
                    f"Training complete — GGUF export failed: {_format_exc(e)}"
                )
                self.state.error = self.state.message
                self._notify()
        if cfg.export_gptq:
            self._do_export_gptq(cfg.output_dir)
        if cfg.export_imatrix:
            self._do_export_imatrix(cfg.output_dir)
        try:
            self._auto_generate_suite()
        except Exception as e:
            log.exception("Auto-suite failed (non-fatal)")
            if not (self.state.message or "").startswith("Training complete —"):
                self.state.message = (
                    f"Training complete — auto-suite failed: {_format_exc(e)}"
                )
                self.state.error = self.state.message
                self._notify()
        # Optional: Abliteration (de-censor)
        if getattr(cfg, 'abliterate', False):
            self._do_abliteration()
        self.state.status = "done"
        if not (self.state.message or "").startswith("Training complete —"):
            self.state.message = "Training complete!"
        self._notify()

    def _train_standard(self, train_data):
        import sys
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer

        from finetune_studio.training.sft_args import build_sft_args_from_config
        cfg = self.config
        self.state.message = "Loading model..."
        self._notify()
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = self._load_model_with_fallback(cfg.model_path, tokenizer)
        if self._stop_requested():
            self._mark_stopped()
            return
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
        if self._stop_requested():
            self._mark_stopped()
            return
        denom = max(1, cfg.batch_size * cfg.gradient_accumulation_steps)
        steps_per_epoch = max(1, math.ceil(len(dataset) / denom))
        total = steps_per_epoch * cfg.num_epochs
        self.state.total_steps = total
        # Fix PicklingError: re-patch sys.modules after any unsloth/trl patches
        import trl.trainer.sft_trainer as _sft_trainer_mod
        import trl.trainer.sft_config as _sft_config_mod
        sys.modules["trl.trainer.sft_trainer"].SFTTrainer = _sft_trainer_mod.SFTTrainer
        sys.modules["trl.trainer.sft_config"].SFTConfig = _sft_config_mod.SFTConfig

        from trl import SFTTrainer
        # SFTConfig (not TrainingArguments): avoids TRL KeyError push_to_hub_token.
        args = build_sft_args_from_config(cfg)
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
        class StopCallback(TrainerCallback):
            def on_step_end(self2, args, state, control, **kwargs):
                if engine._stop_requested():
                    control.should_training_stop = True
                return control
        trainer = SFTTrainer(
            model=model, processing_class=tokenizer, train_dataset=dataset,
            args=args, callbacks=[ProgressCallback(), StopCallback()],
        )
        self.state.status = "training"
        self.state.message = "Training…"
        self._notify()
        trainer.train()
        if self._stop_requested():
            try:
                os.makedirs(cfg.output_dir, exist_ok=True)
                adapter_dir = os.path.join(cfg.output_dir, "adapter")
                model.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
            except Exception:
                log.exception("Failed to save adapter after stop")
            self._mark_stopped()
            return
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
        if self._stop_requested():
            self._mark_stopped()
            return
        if cfg.merge_on_save:
            self._maybe_merge(model, tokenizer, cfg.output_dir)
        if cfg.export_gguf:
            try:
                self._do_export_gguf(cfg.output_dir)
            except Exception as e:
                log.exception("GGUF export failed (non-fatal)")
                self.state.message = (
                    f"Training complete — GGUF export failed: {_format_exc(e)}"
                )
                self.state.error = self.state.message
                self._notify()
        self.state.status = "done"
        if not (self.state.message or "").startswith("Training complete —"):
            self.state.message = "Training complete!"
        self._notify()

    def _do_merge(self, model, tokenizer, output_dir: str) -> dict:
        """Merge the PEFT adapter onto a 16-bit base and save to ``merged/``.

        Frees the training model first, resolves a non-quantized base via
        ``resolve_merge_base``, loads it in bfloat16, then
        ``PeftModel.from_pretrained`` + ``merge_and_unload``. Partial
        ``merged/`` dirs are removed on failure.
        """
        merged_dir = os.path.join(output_dir, "merged")
        if _merged_dir_complete(merged_dir):
            size = _dir_size(merged_dir)
            self.state.message = "Merged model already exists; skipping."
            self._notify()
            return {"merged_path": merged_dir, "size_bytes": size,
                    "size_human": _human_size(size), "skipped": True}
        # Stale partial merge (config-only) — wipe before retrying.
        if os.path.isdir(merged_dir):
            shutil.rmtree(merged_dir, ignore_errors=True)
        os.makedirs(merged_dir, exist_ok=True)
        if os.environ.get("FTS_SKIP_MERGE") == "1":
            self.state.message = "FTS_SKIP_MERGE=1 — skipping merge."
            self._notify()
            return {"merged_path": merged_dir, "size_bytes": 0,
                    "size_human": "0 B", "skipped": True}
        self.state.message = "Merging adapter into full model..."
        self._notify()

        # Free the in-memory QLoRA training model before loading a 16-bit base.
        try:
            del model
        except Exception:  # noqa: BLE001, S110
            pass
        _free_cuda()

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from finetune_studio.training.merge_base import resolve_merge_base

        adapter_dir = os.path.join(output_dir, "adapter")
        base_path = resolve_merge_base(self.config.model_path)
        base = None
        peft_model = None
        merged = None
        try:
            base = AutoModelForCausalLM.from_pretrained(
                base_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            peft_model = PeftModel.from_pretrained(base, adapter_dir)
            merged = peft_model.merge_and_unload()
            if hasattr(merged, "config") and hasattr(merged.config, "quantization_config"):
                merged.config.quantization_config = None
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            src = os.path.join(adapter_dir, "chat_template.jinja")
            if os.path.exists(src):
                shutil.copy(src, os.path.join(merged_dir, "chat_template.jinja"))
            prompt_src = os.path.join(output_dir, "system_prompt.txt")
            if os.path.exists(prompt_src):
                shutil.copy(prompt_src, os.path.join(merged_dir, "system_prompt.txt"))
        except Exception:
            if os.path.isdir(merged_dir):
                shutil.rmtree(merged_dir, ignore_errors=True)
            raise
        finally:
            try:
                del base, peft_model, merged
            except Exception:  # noqa: BLE001, S110
                pass
            _free_cuda()

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
    if _merged_dir_complete(merged_dir) and not force:
        size = _dir_size(merged_dir)
        return {"merged_path": merged_dir, "size_bytes": size,
                "size_human": _human_size(size), "skipped": True, "run": run}
    if os.path.isdir(merged_dir) and (force or not _merged_dir_complete(merged_dir)):
        shutil.rmtree(merged_dir, ignore_errors=True)
    if os.environ.get("FTS_SKIP_MERGE") == "1":
        os.makedirs(merged_dir, exist_ok=True)
        with open(os.path.join(merged_dir, "SKIPPED_BY_TEST"), "w") as f:
            f.write("FTS_SKIP_MERGE=1\n")
        return {"merged_path": merged_dir, "size_bytes": 0,
                "size_human": "0 B", "skipped": True, "run": run}
    os.makedirs(merged_dir, exist_ok=True)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetune_studio.training.merge_base import resolve_merge_base

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base_path = resolve_merge_base(base_model)
    _free_cuda()
    base = None
    model = None
    merged = None
    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        merged = model.merge_and_unload()
        # Strip quantization config from merged model
        if hasattr(merged, "config") and hasattr(merged.config, "quantization_config"):
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
    except Exception:
        if os.path.isdir(merged_dir) and not _merged_dir_complete(merged_dir):
            shutil.rmtree(merged_dir, ignore_errors=True)
        raise
    finally:
        try:
            del base, model, merged
        except Exception:  # noqa: BLE001, S110
            pass
        _free_cuda()
    size = _dir_size(merged_dir)
    return {"merged_path": merged_dir, "size_bytes": size,
            "size_human": _human_size(size), "skipped": False, "run": run}
