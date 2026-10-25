"""Templates subpackage — exports the template management API.

This subpackage provides:
- `render_chat()`: render messages using a Jinja2 template
- `extract_template_from_gguf()`: read template + tokens from a GGUF file
- `render_with_model_template()`: convenience one-shot function

The canonical renderer (renderer.py) is the SINGLE SOURCE OF TRUTH
for Jinja template rendering. Always import from here, NOT from a copy.
"""
from .renderer import (
    extract_template_from_gguf,
    render_chat,
    render_with_model_template,
)

__all__ = [
    "extract_template_from_gguf",
    "render_chat",
    "render_with_model_template",
]
