"""An API connection (Settings → AI assistant) as the reader of a Smart Brain test run.

It answers the same grounded prompt the local model gets (``testing.brain_suite.run_brain_case``), through the
connection's own client: nothing is loaded on the GPU and nothing is downloaded, so a user without a GPU can test
the package's answers. Only the Smart Brain mode takes it (a closed-book quiz of a remote model says nothing about
this project's training).
"""
from __future__ import annotations

from typing import Any

#: The prompt budget for an API reader: its window is not known here; 32k tokens is what current API models take.
API_READER_N_CTX = 32768
API_READER_TIMEOUT_S = 180.0


class ApiReader:
    """Duck-types the inference engine surface ``run_brain_suite`` uses: ``generate``, ``model_path``, ``n_ctx``,
    ``model``."""

    def __init__(self, choice: str) -> None:
        from finetune_studio.models.assistant import connection_client
        from finetune_studio.webui.testing_models import API_READER_PREFIX

        self.connection_id = choice[len(API_READER_PREFIX):]
        self.client = connection_client(self.connection_id)    # ValueError: not an API connection
        self.model_path = choice
        self.model = self.client
        self.n_ctx = API_READER_N_CTX
        self.last_error = ""

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int = 512, temperature: float = 0.3,
                 think: bool = False) -> str:
        del think   # the connection's own settings decide (e.g. reasoning_effort in its request body)
        return self.client.chat(messages, max_tokens=max_tokens, temperature=temperature,
                                timeout=API_READER_TIMEOUT_S)
