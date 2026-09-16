"""Local SentenceTransformer directory helpers.

The shared model store historically copied only top-level files from
``model.save()``, dropping ``1_Pooling`` / ``2_Normalize``. Loading those
incomplete dirs with sentence-transformers ≥5.x raises::

    Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'

These helpers (1) copy ST trees with module subdirs intact and (2) repair or
normalize incomplete local dirs before ``SentenceTransformer(path)`` so the
*same* model identity loads — never a silent switch to a different embedder.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def copy_sentence_transformer_tree(src: Path, dest: Path) -> None:
    """Copy a sentence-transformers save directory, preserving module subdirs.

    Skips ``META.json`` (shared-store metadata written separately).
    """
    src = Path(src)
    dest = Path(dest)
    if not src.is_dir():
        raise FileNotFoundError(f"SentenceTransformer source dir missing: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name == "META.json":
            continue
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        elif item.is_file():
            shutil.copy2(item, target)


def move_sentence_transformer_tree(src: Path, dest: Path) -> None:
    """Move staged ST save contents into ``dest``, preserving module subdirs."""
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        if item.name == "META.json":
            continue
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))


def is_complete_sentence_transformer_dir(model_dir: Path) -> bool:
    """True when ``modules.json`` module paths exist with required configs."""
    model_dir = Path(model_dir)
    modules_path = model_dir / "modules.json"
    if not modules_path.is_file():
        return False
    try:
        modules: list[dict[str, Any]] = json.loads(
            modules_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    for mod in modules:
        rel = str(mod.get("path") or "")
        if not rel:
            continue
        mod_dir = model_dir / rel
        if not mod_dir.is_dir():
            return False
        mod_type = str(mod.get("type") or "")
        if _is_pooling_module(mod_type):
            cfg = mod_dir / "config.json"
            if not cfg.is_file():
                return False
            try:
                data = json.loads(cfg.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return False
            if "embedding_dimension" not in data and "word_embedding_dimension" not in data:
                return False
    return True


def prepare_local_sentence_transformer_dir(model_dir: Path) -> Path:
    """Repair/normalize a local ST dir so current sentence-transformers can load it.

    - Creates missing module subdirs referenced by ``modules.json``
    - Writes a minimal Pooling ``config.json`` when absent (infers dim)
    - Renames legacy ``word_embedding_dimension`` → ``embedding_dimension``

    Raises ``RuntimeError`` with an actionable message when the dir cannot be
    made loadable (no silent switch to another embedder).
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local SentenceTransformer dir not found: {model_dir}")

    modules_path = model_dir / "modules.json"
    if not modules_path.is_file():
        # HF hub-style cache dirs / plain transformers checkpoints are loaded
        # by SentenceTransformer without modules.json — leave untouched.
        return model_dir

    try:
        modules: list[dict[str, Any]] = json.loads(
            modules_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Corrupt modules.json in local SentenceTransformer dir {model_dir}: {e}"
        ) from e

    dim = _infer_embedding_dimension(model_dir)
    for mod in modules:
        rel = str(mod.get("path") or "")
        if not rel:
            continue
        mod_dir = model_dir / rel
        mod_type = str(mod.get("type") or "")
        mod_dir.mkdir(parents=True, exist_ok=True)
        if _is_pooling_module(mod_type):
            _ensure_pooling_config(mod_dir, dim, model_dir=model_dir)
        else:
            cfg = mod_dir / "config.json"
            if cfg.is_file():
                _normalize_pooling_keys_in_place(cfg)

    return model_dir


def _is_pooling_module(mod_type: str) -> bool:
    leaf = mod_type.rsplit(".", 1)[-1]
    return leaf == "Pooling" or mod_type.endswith(".pooling.Pooling")


def _infer_embedding_dimension(model_dir: Path) -> int:
    """Best-effort dim from existing pooling config or transformer configs."""
    for candidate in (
        model_dir / "1_Pooling" / "config.json",
        model_dir / "sentence_bert_config.json",
        model_dir / "config.json",
    ):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in (
            "embedding_dimension",
            "word_embedding_dimension",
            "hidden_size",
        ):
            val = data.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return 0


def _ensure_pooling_config(mod_dir: Path, dim: int, *, model_dir: Path) -> None:
    cfg_path = mod_dir / "config.json"
    if cfg_path.is_file():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Corrupt Pooling config at {cfg_path}: {e}. "
                f"Re-register the embedder in the shared model store."
            ) from e
        changed = _apply_pooling_compat(data, dim)
        if "embedding_dimension" not in data:
            raise RuntimeError(
                f"Incomplete SentenceTransformer dir at {model_dir}: "
                f"{cfg_path.relative_to(model_dir)} has no embedding_dimension "
                f"(and dimension could not be inferred). "
                f"Re-register the embedder or rebuild the corpus. "
                f"Original symptom: Pooling.__init__() missing embedding_dimension."
            )
        if changed:
            cfg_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
        return

    if dim <= 0:
        raise RuntimeError(
            f"Incomplete SentenceTransformer dir at {model_dir}: "
            f"missing {mod_dir.name}/config.json and cannot infer "
            f"embedding_dimension from config.json. "
            f"Re-register the embedder in the shared model store "
            f"(shared copies must keep 1_Pooling/). "
            f"Original symptom: Pooling.__init__() missing embedding_dimension."
        )
    data = {
        "embedding_dimension": dim,
        "pooling_mode": "mean",
        "include_prompt": True,
    }
    cfg_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")


def _apply_pooling_compat(data: dict[str, Any], fallback_dim: int) -> bool:
    """Normalize legacy pooling keys. Returns True if ``data`` was mutated."""
    changed = False
    if "embedding_dimension" not in data and "word_embedding_dimension" in data:
        data["embedding_dimension"] = data["word_embedding_dimension"]
        changed = True
    if "embedding_dimension" not in data and fallback_dim > 0:
        data["embedding_dimension"] = fallback_dim
        changed = True
    if "pooling_mode" not in data:
        # Legacy boolean flags → single mode string (ST ≥5 expects pooling_mode).
        mode = "mean"
        if data.get("pooling_mode_cls_token"):
            mode = "cls"
        elif data.get("pooling_mode_max_tokens"):
            mode = "max"
        elif data.get("pooling_mode_mean_tokens") is False and data.get(
            "pooling_mode_cls_token"
        ):
            mode = "cls"
        data["pooling_mode"] = mode
        changed = True
    return changed


def _normalize_pooling_keys_in_place(cfg_path: Path) -> None:
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if _apply_pooling_compat(data, 0):
        cfg_path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
