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
        """Read model metadata (layer count, etc.) without loading the full model."""
        from pathlib import Path
        import struct, mmap, json
        path = Path(model_path)
        result = {"total_layers": 0, "num_kv_heads": 0, "head_dim": 0, "n_ctx_default": 4096}

        if path.is_file() and path.suffix == ".gguf":
            try:
                with open(path, "rb") as f:
                    h = f.read(262144)  # 256KB covers all metadata before tokenizer vocab
                off = 24  # magic(4) + version(4) + n_tensors(8) + n_kv(8)
                n_kv = struct.unpack_from("<Q", h, 16)[0]
                for _ in range(int(n_kv)):
                    if off + 9 > len(h): break
                    klen = struct.unpack_from("<Q", h, off)[0]; off += 8
                    if klen > 200 or off + klen + 1 > len(h): break
                    key = h[off:off+klen].decode("utf-8"); off += klen
                    vtype = h[off]; off += 1
                    if vtype == 9:  # ARRAY — skip (tokenizer vocab is huge)
                        alen = struct.unpack_from("<Q", h, off)[0]; off += 8
                        atype = h[off]; off += 1
                        can_skip = True
                        for _ in range(alen):
                            if atype == 8: el = struct.unpack_from("<Q", h, off)[0]; off += 8 + el
                            elif atype in (0,1): off += 1
                            elif atype in (2,3): off += 2
                            elif atype in (4,5): off += 4
                            elif atype == 6: off += 4
                            elif atype == 7: off += 1
                            else: can_skip = False; break
                            if off > len(h): can_skip = False; break
                        if not can_skip: break
                        continue
                    if vtype == 0: val = h[off]; off += 1
                    elif vtype == 1: val = struct.unpack_from("<b", h, off)[0]; off += 1
                    elif vtype == 2: val = struct.unpack_from("<H", h, off)[0]; off += 2
                    elif vtype == 3: val = struct.unpack_from("<h", h, off)[0]; off += 2
                    elif vtype == 4: val = struct.unpack_from("<I", h, off)[0]; off += 4
                    elif vtype == 5: val = struct.unpack_from("<i", h, off)[0]; off += 4
                    elif vtype == 6: val = struct.unpack_from("<f", h, off)[0]; off += 4
                    elif vtype == 7: val = h[off] != 0; off += 1
                    elif vtype == 8:
                        slen = struct.unpack_from("<Q", h, off)[0]; off += 8
                        val = h[off:off+slen].decode("utf-8", errors="replace"); off += slen
                    else: break
                    # Extract what we need
                    if "block_count" in key and result["total_layers"] == 0:
                        result["total_layers"] = int(val) if isinstance(val, (int, float)) else 0
                    elif "head_count_kv" in key and result["num_kv_heads"] == 0:
                        result["num_kv_heads"] = int(val) if isinstance(val, (int, float)) else 0
                    elif "head_count" in key and "kv" not in key and result["num_kv_heads"] == 0:
                        result["num_kv_heads"] = int(val) if isinstance(val, (int, float)) else 0
                    elif "rope.dimension_count" in key:
                        result["head_dim"] = int(val) if isinstance(val, (int, float)) else 0
                    elif "embedding_length" in key and result["head_dim"] == 0:
                        embd = int(val) if isinstance(val, (int, float)) else 0
                        if embd and result["num_kv_heads"]:
                            result["head_dim"] = embd // result["num_kv_heads"]
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
