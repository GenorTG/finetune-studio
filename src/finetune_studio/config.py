"""Application settings and constants.

WHAT THIS FILE DOES
==================
Centralizes all the configuration values used by finetune-studio:
  - File paths (data directories, model directories)
  - Default hyperparameters (learning rate, batch size, epochs)
  - Server settings (host, port)
  - Debug flags

KEY CONCEPTS
============
- Pydantic BaseModel: a way to define a typed settings object.
  Gives you type checking, default values, and validation.
- Environment variables: settings can be overridden by env vars
  (useful for production deployments).
- Frozen dataclass: immutable config that can't be accidentally modified.
"""

import os
from dataclasses import dataclass, field


@dataclass
class RAGSettings:
    """Typed configuration for RAG (Retrieval-Augmented Generation).

    Used as Settings.rag so callers can access .store_path, .embedding_model,
    .min_score with full type information (instead of dict[str, Any]).
    """
    store_path: str = "data/rag_store"
    enabled: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_score: float = 0.3
    documents_path: str = "data/rag_documents"
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class Settings:
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    model_dirs: list = field(default_factory=lambda: [
        os.path.expanduser("~/1TB-SAMSUNG/Comfy/models"),
        os.path.expanduser("~/gemma-finetune"),
        os.path.expanduser("~/gemma-training"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ])
    default_lora_rank: int = 64
    default_lr: float = 8e-5
    default_epochs: int = 4
    default_batch_size: int = 2
    default_max_seq_length: int = 2048
    data_dir: str = "data"
    db_path: str = "data/finetune_studio.db"
    rag_store_path: str = "data/rag_store"
    rag_embedding_model: str = "all-MiniLM-L6-v2"
    rag: RAGSettings = field(default_factory=RAGSettings)

settings = Settings()

