"""Run inference on a single prompt.

WHAT THIS FILE DOES
==================
A thin wrapper around the inference engine for testing purposes.
Loads a model, sends a prompt, returns the response with metadata.

KEY CONCEPTS
============
- Test isolation: each test loads its own model instance to avoid
  interference between tests.
- Determinism: testing often requires reproducible outputs. Set
  temperature=0 for deterministic generation.
"""

import os
import threading
import time

import torch

# Auto-unload after idle (configurable via FTS_IDLE_TIMEOUT env, default 30 min)
IDLE_TIMEOUT = int(os.environ.get("FTS_IDLE_TIMEOUT", 1800))


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path = None
        self._gguf_template = None
        self._last_used = 0.0
        self._idle_timer = None

    def load(self, model_path, device="auto", n_ctx=4096, n_gpu_layers=99, n_batch=512, mmap=True, mlock=False):
        from pathlib import Path
        self.unload()
        path = Path(model_path)
        if path.is_file() and path.suffix == ".gguf":
            self._load_gguf(str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_batch=n_batch, mmap=mmap, mlock=mlock)
        else:
            self._load_hf(model_path, device)
        self.model_path = model_path
        self._last_used = time.time()
        self._start_idle_timer()

    def _load_hf(self, model_path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
        self.is_gguf = False

    def _load_gguf(self, gguf_path, n_ctx=4096, n_gpu_layers=99, n_batch=512, mmap=True, mlock=False):
        from pathlib import Path
        from llama_cpp import Llama
        self.is_gguf = True
        self.vision = False
        self.mmproj_path = None

        # Auto-detect mmproj in same directory
        gguf_dir = Path(gguf_path).parent
        base_name = Path(gguf_path).stem.replace("-Q4_K_M", "").replace("-Q8_0", "").replace("-F16", "").replace("-BF16", "")
        for candidate in gguf_dir.glob("mmproj*.gguf"):
            self.mmproj_path = str(candidate)
            break
        if not self.mmproj_path:
            # Also check for files matching base model name
            for candidate in gguf_dir.glob(f"*mmproj*{base_name}*.gguf"):
                self.mmproj_path = str(candidate)
                break

        chat_handler = None
        if self.mmproj_path:
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=self.mmproj_path, verbose=False)
                self.vision = True
                print(f"Vision enabled: mmproj={Path(self.mmproj_path).name}")
            except Exception as e:  # noqa: BLE001
                print(f"mmproj load failed ({e}), running text-only")
                self.mmproj_path = None

        self.model = Llama(
            model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            n_batch=n_batch, mmap=mmap, mlock=mlock,
            chat_handler=chat_handler, verbose=False,
        )
        self.tokenizer = None
        # Cache the GGUF's own chat template + tokens so we don't re-extract per call.
        try:
            from finetune_studio.templates.renderer import extract_template_from_gguf
            self._gguf_template = extract_template_from_gguf(gguf_path)
        except Exception:
            self._gguf_template = None

    def _start_idle_timer(self):
        """Start/restart the idle unload timer."""
        if self._idle_timer:
            self._idle_timer.cancel()
        if IDLE_TIMEOUT > 0 and self.model is not None:
            self._idle_timer = threading.Timer(IDLE_TIMEOUT, self._auto_unload)
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _auto_unload(self):
        """Unload model if idle too long."""
        if self.model is None:
            return
        elapsed = time.time() - self._last_used
        if elapsed >= IDLE_TIMEOUT:
            print(f"[inference] Auto-unloading {self.model_path} (idle {int(elapsed)}s)")
            self.unload()

    def unload(self):
        if self._idle_timer:
            self._idle_timer.cancel()
            self._idle_timer = None
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path = None
        self._gguf_template = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def idle_seconds(self):
        """Seconds since last use, or 0 if no model loaded."""
        if self.model is None or self._last_used == 0:
            return 0
        return int(time.time() - self._last_used)

    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, stop=None):
        if self.model is None:
            raise RuntimeError("No model loaded")
        self._last_used = time.time()
        self._start_idle_timer()
        if self.is_gguf:
            return self._generate_gguf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop)
        return self._generate_hf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop)

    def _generate_hf(self, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
                top_p=top_p, top_k=top_k, repetition_penalty=repeat_penalty,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_gguf(self, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop):
        if self.vision and self.mmproj_path:
            # Use chat_handler which supports image content
            result = self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )
            return result["choices"][0]["message"]["content"].strip()
        # Text-only: respect the GGUF's built-in tokenizer.chat_template via the existing
        # finetune_studio.templates.renderer. Mixing templates across models is fragile —
        # we never fall back to a hardcoded prompt; the renderer itself falls
        # back to ChatML only when the GGUF has no template at all.
        from finetune_studio.templates.renderer import render_chat
        tmpl = self._gguf_template or {}
        prompt = render_chat(
            template_str=tmpl.get("chat_template", ""),
            messages=messages,
            tools=None,
            bos_token=tmpl.get("bos_token", "<bos>"),
            eos_token=tmpl.get("eos_token", "<eos>"),
            add_generation_prompt=True,
        )
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=max(temperature, 0.01),
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )
        return output["choices"][0]["text"].strip()
