"""Templates subpackage — exports the template management API.

This subpackage provides:
- `render_chat()`: render messages using a Jinja2 template
- `extract_template_from_gguf()`: read template + tokens from a GGUF file
- `render_with_model_template()`: convenience one-shot function
- `TemplateManager`: class that caches templates per model
- `ChatTemplate`: dataclass representation of a model's template
- `DEFAULT_TOOLS`: standard tool definitions for testing

The canonical renderer (renderer.py) is the SINGLE SOURCE OF TRUTH
for Jinja template rendering. Always import from here, NOT from a copy.
"""

"""Templates package - Jinja2 chat template rendering and management."""
from .manager import DEFAULT_TOOLS, ChatTemplate, TemplateManager
from .renderer import (
    extract_template_from_gguf,
    render_chat,
    render_with_model_template,
)

__all__ = [
    "DEFAULT_TOOLS",
    "ChatTemplate",
    "TemplateManager",
    "extract_template_from_gguf",
    "render_chat",
    "render_with_model_template",
]
