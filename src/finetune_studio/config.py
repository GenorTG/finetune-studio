"""Mutable dataclass defaults for the app and its nested RAG settings.

This module defines ``Settings`` and ``RAGSettings`` plus the process-wide
``settings`` singleton. It reads no environment itself (the HF cache path comes from ``hf_env``), validate
values with Pydantic, or freeze the dataclasses; integrations that read their
own environment or settings files do so at their call sites.
"""

from dataclasses import dataclass, field

from finetune_studio.hf_env import hf_hub_cache


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
        "models",           # project-local models/ directory
        "output",           # training output directory
    ])
    model_dirs_extra: list = field(default_factory=lambda: [
        str(hf_hub_cache()),  # HF hub cache (honors HF_HUB_CACHE / HF_HOME)
        "~/.finetune-studio/hf_models",  # HF Explorer Pull destination
        "~/.finetune-studio/shared_models",  # embedder/reranker cache
    ])  # user-added via env or config
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
