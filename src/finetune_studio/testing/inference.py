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
from collections import OrderedDict

import torch

# Module-level cache for GGUF metadata reads (fast header-only parser is still
# cheap, but for 16 GB files we don't want to walk the header twice).
_GGUF_META_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_GGUF_META_CACHE_MAX = 32

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

    def load(self, model_path, device="auto", n_ctx=4096, n_gpu_layers=99, n_batch=512, mmap=True, mlock=False,
              n_threads=None, flash_attn=True, seed=None, rope_freq_base=0.0, rope_freq_scale=0.0):
        from pathlib import Path
        self.unload()
        path = Path(model_path)
        if path.is_file() and path.suffix == ".gguf":
            self._load_gguf(str(path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_batch=n_batch,
                            mmap=mmap, mlock=mlock, n_threads=n_threads, flash_attn=flash_attn,
                            seed=seed, rope_freq_base=rope_freq_base, rope_freq_scale=rope_freq_scale)
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

    def _load_gguf(self, gguf_path, n_ctx=4096, n_gpu_layers=99, n_batch=512, mmap=True, mlock=False,
                   n_threads=None, flash_attn=True, seed=None, rope_freq_base=0.0, rope_freq_scale=0.0):
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

        import multiprocessing
        if n_threads is None or n_threads <= 0:
            n_threads = multiprocessing.cpu_count()

        kwargs = dict(
            model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
            n_batch=n_batch, mmap=mmap, mlock=mlock,
            chat_handler=chat_handler, verbose=False,
            n_threads=n_threads,
        )
        if flash_attn:
            kwargs["flash_attn"] = True
        if seed is not None and seed >= 0:
            kwargs["seed"] = seed
        if rope_freq_base > 0:
            kwargs["rope_freq_base"] = rope_freq_base
        if rope_freq_scale > 0:
            kwargs["rope_freq_scale"] = rope_freq_scale
        self.model = Llama(**kwargs)
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
        """Unload model and free VRAM/RAM immediately.

        Previous implementation just set `self.model = None` and relied on
        Python GC to eventually call `Llama.__del__` (which frees the native
        llama.cpp context). Under load this could take seconds to minutes,
        leaving VRAM occupied and the model appearing "still loaded" to the
        next inference request or to the dashboard's VRAM chip. This version:

        1. Cancels the idle timer.
        2. Drops every reference to the model + tokenizer + gguf metadata.
        3. Explicitly `del`s the object and forces an immediate GC pass so
           `Llama.__del__` runs while we're still on the calling thread
           (the native free is deterministic this way).
        4. Calls `torch.cuda.empty_cache()` if available so PyTorch's
           caching allocator hands memory back to the driver.

        After unload, `self.model is None` AND the Llama native context is
        gone — verified via nvidia-smi drop in VRAM within a second.
        """
        import gc

        if self._idle_timer is not None:
            try:
                self._idle_timer.cancel()
            except Exception:
                pass
        self._idle_timer = None

        # Drop references first.
        _model = self.model
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self.vision = False
        self.mmproj_path = None
        self._gguf_template = None
        self._last_used = 0.0

        # Explicit del + GC pass so Llama.__del__ runs now, not "later".
        # Without this, a busy process can hold the model in VRAM for
        # many seconds after unload returns.
        if _model is not None:
            try:
                del _model
            except Exception:
                pass
        try:
            gc.collect()
        except Exception:
            pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @property
    def idle_seconds(self):
        """Seconds since last use, or 0 if no model loaded."""
        if self.model is None or self._last_used == 0:
            return 0
        return int(time.time() - self._last_used)

    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, stop=None, think=False):
        if self.model is None:
            raise RuntimeError("No model loaded")
        self._last_used = time.time()
        self._start_idle_timer()
        if self.is_gguf:
            return self._generate_gguf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop)
        return self._generate_hf(messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop, think=think)

    def _generate_hf(self, messages, max_tokens, temperature, top_p, top_k, repeat_penalty, stop, think=False):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
                top_p=top_p, top_k=top_k, repetition_penalty=repeat_penalty,
                do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Strip Qwen3 thinking traces: <think>...</think> blocks
        import re
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        # Also strip the trailing \n\n that follows </think>
        response = response.lstrip("\n")
        return response

    @staticmethod
    def estimate_memory(model_path, n_ctx=4096, n_gpu_layers=99):
        """Estimate VRAM/RAM usage for a model. Returns dict with estimates in GB."""
        from pathlib import Path
        path = Path(model_path)
        result = {"weights_gb": 0.0, "kv_cache_gb": 0.0, "total_vram_gb": 0.0, "total_ram_gb": 0.0, "total_layers": 0}

        if path.is_file() and path.suffix == ".gguf":
            size_gb = path.stat().st_size / (1024**3)
            # Read GGUF metadata via fast header parser (no model load)
            meta_info = InferenceEngine.read_model_metadata(str(path))
            total_layers = meta_info["total_layers"]
            num_kv_heads = meta_info["num_kv_heads"]
            head_dim = meta_info["head_dim"]

            if total_layers == 0:
                total_layers = 32  # fallback
            if num_kv_heads == 0:
                num_kv_heads = 8
            if head_dim == 0:
                head_dim = 128

            # Weight distribution
            gpu_frac = min(n_gpu_layers / total_layers, 1.0)
            weights_vram = size_gb * gpu_frac
            weights_ram = size_gb * (1.0 - gpu_frac)

            # KV cache: n_ctx * 2 * n_layers * n_kv_heads * head_dim * 2 bytes (fp16)
            kv_bytes = n_ctx * 2 * total_layers * num_kv_heads * head_dim * 2
            kv_gb = kv_bytes / (1024**3)

            result.update({
                "weights_gb": round(weights_vram, 2),
                "kv_cache_gb": round(kv_gb, 2),
                "total_vram_gb": round(weights_vram + kv_gb, 2),
                "total_ram_gb": round(weights_ram, 2),
                "total_layers": total_layers,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            })
        else:
            # Safetensors — approximate from file size
            if path.is_dir():
                total = sum(
                    f.stat().st_size for f in path.rglob("*")
                    if f.suffix in (".safetensors", ".bin", ".pt")
                ) / (1024**3)
            else:
                total = path.stat().st_size / (1024**3) if path.exists() else 0
            result["weights_gb"] = round(total, 2)
            result["total_vram_gb"] = round(total, 2)
            result["total_ram_gb"] = 0.0
        return result

    @staticmethod
    def read_model_metadata(model_path):
        """Read model metadata (layer count, etc.) without loading the full model.

        Uses an in-process cache keyed on (path, mtime) so repeated calls on the
        same file are O(1). Reads only the GGUF header (first few KB) by hand
        so we never walk the whole file just to peek at layer counts.
        """
        from pathlib import Path
        import json
        import time as _time
        path = Path(model_path)
        try:
            st = path.stat()
            cache_key = (str(path), st.st_mtime, st.st_size)
        except OSError:
            cache_key = (str(path), 0, 0)
        cached = _GGUF_META_CACHE.get(cache_key)
        if cached is not None:
            return cached
        result = {"total_layers": 0, "num_kv_heads": 0, "head_dim": 0, "n_ctx_default": 4096}

        if path.is_file() and path.suffix == ".gguf":
            try:
                # Fast path: parse GGUF header in-process (first few KB).
                # GGUF files are little-endian; header has magic 'GGUF' (0x46554747).
                # Format (v2/v3):
                #   magic: 4 bytes ('GGUF')
                #   version: uint32
                #   tensor_count: uint64
                #   kv_count: uint64
                #   For each KV:
                #     key_length: uint64
                #     key: utf-8 bytes
                #     value_type: uint32 (0..=10 are scalar; >10 are arrays)
                #     value: depends on type
                import struct
                with open(path, "rb") as f:
                    magic = f.read(4)
                    if magic != b"GGUF":
                        raise ValueError("not a GGUF file")
                    f.read(4)  # version (v2 or v3)
                    f.read(8)  # tensor_count
                    f.read(8)  # kv_count (we don't need exact — bounded by safety cap)
                    # The actual key format is "{arch}.X" (e.g. "qwen35.block_count"),
                    # but we don't know `arch` until we find general.architecture.
                    # Collect every KV (cheap — small header), match the arch-prefixed
                    # ones after we know the architecture.
                    found_arch = None
                    found = {}
                    for _ in range(500):  # safety cap
                        raw = f.read(8)
                        if len(raw) < 8:
                            break
                        klen = struct.unpack("<Q", raw)[0]
                        if klen == 0 or klen > 1024:
                            break
                        key = f.read(klen).decode("utf-8", errors="ignore").strip("\x00")
                        raw_t = f.read(4)
                        if len(raw_t) < 4:
                            break
                        gtype = struct.unpack("<I", raw_t)[0]
                        if gtype == 4:  # UINT32
                            v = struct.unpack("<I", f.read(4))[0]
                        elif gtype == 5:  # INT32
                            v = struct.unpack("<i", f.read(4))[0]
                        elif gtype == 8:  # STRING
                            slen = struct.unpack("<Q", f.read(8))[0]
                            v = f.read(slen).decode("utf-8", errors="ignore").strip("\x00")
                        elif gtype == 10:  # BOOL
                            v = bool(f.read(1)[0])
                        elif gtype == 0:  # UINT8
                            v = struct.unpack("<B", f.read(1))[0]
                        elif gtype == 6:  # FLOAT32
                            f.read(4); v = None
                        elif gtype == 7:  # FLOAT64
                            f.read(8); v = None
                        else:
                            # ARRAY (type 9): [4-byte elem_type][8-byte count][elements...]
                            etype = struct.unpack("<I", f.read(4))[0]
                            acount = struct.unpack("<Q", f.read(8))[0]
                            if etype == 8:  # string array: variable-size elements
                                for _ in range(acount):
                                    slen = struct.unpack("<Q", f.read(8))[0]
                                    f.read(slen)
                            else:
                                sizes = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:8, 10:1}
                                esize = sizes.get(etype, 1)
                                f.read(acount * esize)
                            v = None
                        if key == "general.architecture" and isinstance(v, str):
                            found_arch = v
                            result["total_layers"] = 0  # reset so we know real values vs defaults
                            continue  # we'll re-fill below once we know arch
                        # Once we know the arch, accept arch-prefixed scalars.
                        if found_arch and key.startswith(found_arch + "."):
                            short = key[len(found_arch) + 1:]
                            if short == "block_count" and isinstance(v, int):
                                result["total_layers"] = int(v)
                            elif short == "attention.head_count_kv" and isinstance(v, int):
                                result["num_kv_heads"] = int(v)
                            elif short == "attention.key_length" and isinstance(v, int):
                                result["head_dim"] = int(v)
                            elif short == "context_length" and isinstance(v, int):
                                result["n_ctx_default"] = int(v)
                            if (result["total_layers"] > 0 and result["num_kv_heads"] > 0
                                    and result["head_dim"] > 0 and result["n_ctx_default"] != 4096):
                                # Got everything we need
                                break
                    # If arch-prefixed keys were never found, fall back to full reader.
                    if result["total_layers"] == 0 or result["num_kv_heads"] == 0 or result["head_dim"] == 0:
                        try:
                            from gguf.gguf_reader import GGUFReader
                            reader = GGUFReader(str(path))
                            arch_field = reader.fields.get("general.architecture")
                            if arch_field:
                                arch = bytes(arch_field.parts[arch_field.data[0]]).decode("utf-8")
                                for key_suffix, result_key, fallback in [
                                    (".block_count", "total_layers", 0),
                                    (".attention.head_count_kv", "num_kv_heads", 0),
                                    (".attention.key_length", "head_dim", 0),
                                    (".context_length", "n_ctx_default", 4096),
                                ]:
                                    field = reader.fields.get(f"{arch}{key_suffix}")
                                    if field:
                                        val = field.parts[field.data[0]]
                                        result[result_key] = int(val[0]) if len(val) else fallback
                            del reader
                        except Exception:
                            pass
            except Exception:
                pass


        elif path.is_dir() and (path / "config.json").exists():
            try:
                with open(path / "config.json") as f:
                    cfg = json.load(f)
                result["total_layers"] = cfg.get("num_hidden_layers", 0)
                result["num_kv_heads"] = cfg.get("num_key_value_heads", cfg.get("num_attention_heads", 0))
                result["head_dim"] = cfg.get("hidden_size", 0) // max(cfg.get("num_attention_heads", 1), 1)
                result["n_ctx_default"] = cfg.get("max_position_embeddings", 4096)
            except Exception:
                pass
        # Cache the result (even partial) so next call is O(1)
        _GGUF_META_CACHE[cache_key] = result
        while len(_GGUF_META_CACHE) > _GGUF_META_CACHE_MAX:
            _GGUF_META_CACHE.popitem(last=False)
        return result

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
