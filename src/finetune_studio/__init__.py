"""finetune_studio — training, evaluation, and deployment for fine-tuned LLMs.

This package is the APPLICATION LAYER for our AI workflow:
  - Train models on custom data (training/)
  - Evaluate models on industry benchmarks (benchmarks/)
  - Compare models side-by-side (compare/)
  - Serve models via a web UI (webui/)
  - Manage RAG documents (rag/)
  - Inspect and validate training data (data/)

It depends on the inference-server package for:
  - Jinja2 template rendering (canonical source)
  - Tool-call parsing
  - Built-in tools
"""

__version__ = "0.1.0"
