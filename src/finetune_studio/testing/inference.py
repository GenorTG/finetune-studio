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

import torch


class InferenceEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False

    def load(self, model_path, device="auto"):
        from pathlib import Path
        self.unload()
        path = Path(model_path)
        if path.is_file() and path.suffix == ".gguf":
            self._load_gguf(str(path))
        else:
            self._load_hf(model_path, device)
        self.model_path = model_path

    def _load_hf(self, model_path, device):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
        self.is_gguf = False

    def _load_gguf(self, gguf_path):
        from llama_cpp import Llama
        self.model = Llama(model_path=gguf_path, n_ctx=4096, n_gpu_layers=99, verbose=False)
        self.tokenizer = None
        self.is_gguf = True
        # Cache the GGUF's own chat template + tokens so we don't re-extract per call.
        try:
            from finetune_studio.templates.renderer import extract_template_from_gguf
            self._gguf_template = extract_template_from_gguf(gguf_path)
        except Exception:
            self._gguf_template = None

    def unload(self):
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.is_gguf = False
        self._gguf_template = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(self, messages, max_tokens=512, temperature=0.7, top_p=0.9, stop=None):
        if self.model is None:
            raise RuntimeError("No model loaded")
        if self.is_gguf:
            return self._generate_gguf(messages, max_tokens, temperature, top_p, stop)
        return self._generate_hf(messages, max_tokens, temperature, top_p, stop)

    def _generate_hf(self, messages, max_tokens, temperature, top_p, stop):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=max(temperature, 0.01),
                top_p=top_p, do_sample=temperature > 0, pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_gguf(self, messages, max_tokens, temperature, top_p, stop):
        # Respect the GGUF's built-in tokenizer.chat_template via the existing
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
            stop=stop,
        )
        return output["choices"][0]["text"].strip()
