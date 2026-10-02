# Formatter parity audit (2026-10-02)
Train: training/data.format_for_sft (bakes system prompt unless mode none/runtime) -> engine/profile apply_chat_template(add_generation_prompt=False).
Eval: testing/suite.run_suite sent only the user turn (no system prompt) -> inference._generate_hf apply_chat_template(add_generation_prompt=True, enable_thinking=think). GGUF uses templates.renderer.
Divergences: (1) baked system prompt absent at eval (real); (2) 4 duplicated apply_chat_template call sites; (3) enable_thinking only at inference (template-specific, left as is). EOS/truncation: SFTTrainer owns max_seq_length; eval has none (not unified).
Fix: training/formatting.py (render_chat_text, with_system_prompt); run_suite(system_prompt=""). Callers not yet passing system_prompt (routes owned elsewhere).
RAG coverage filename-based: not addressed (in rag.py/project_rag.py, out of scope).
