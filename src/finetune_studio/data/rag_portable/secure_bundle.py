"""Encrypted studio-to-studio corpus bundle (``.ftsrag``).

Replaces the legacy plaintext tar/zip bundle.  Every corpus file is encrypted
frame by frame (AES-256-GCM, scrypt key from a passphrase) straight into the
output file; no plaintext is staged anywhere.  The key is never stored.

Model weights are public and are NOT bundled: the manifest is rewritten
in memory to ``shared:`` refs so the importing studio resolves/downloads them.
Use the Portable RAG package export when a self-contained offline pack is needed.
"""
from __future__ import annotations

import json
import logging
import secrets
import shutil
from pathlib import Path

from .io import relabel_imported_sources
from .rag_container import MAGIC as _MAGIC
from .rag_container import ContainerReader, ContainerWriter
from .schema import Manifest

log = logging.getLogger(__name__)

BUNDLE_EXT = ".ftsrag"
MIN_PASSPHRASE_LEN = 8
_MODEL_DIRS = frozenset({"embedder", "reranker"})


def is_secure_bundle(path: str | Path) -> bool:
    try:
        with open(path, "rb") as fp:
            return fp.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def generate_passphrase() -> str:
    return secrets.token_urlsafe(18)


def export_secure_bundle(corpus_dir: str | Path, out_path: str | Path,
                         passphrase: str | None = None, *,
                         log_n: int = 17) -> tuple[Path, str | None]:
    """Write the corpus at ``corpus_dir`` to ``out_path`` (encrypted).

    Returns ``(path, generated_passphrase)``; the second item is set only when
    the passphrase was generated here (show it once, never persist it).
    """
    from .store import PortableRAG

    src = Path(corpus_dir)
    if not (src / "manifest.json").is_file():
        raise FileNotFoundError(f"no corpus at {src}")
    generated = None
    if passphrase is None:
        generated = passphrase = generate_passphrase()
    elif len(passphrase) < MIN_PASSPHRASE_LEN:
        raise ValueError(f"passphrase too short (minimum {MIN_PASSPHRASE_LEN} characters)")

    rag = PortableRAG(src)
    manifest = Manifest.from_json(json.loads((src / "manifest.json").read_text(encoding="utf-8")))
    shared = dict((manifest.extra or {}).get("shared_model_paths") or {})
    rag._ensure_shared_refs_in_manifest(manifest, shared)  # in-memory only
    manifest_bytes = json.dumps(manifest.to_json(), indent=2, ensure_ascii=False).encode("utf-8")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    try:
        with open(part, "wb") as fp:
            cw = ContainerWriter(fp, passphrase, log_n=log_n)
            cw.add("manifest.json", manifest_bytes)
            for f in sorted(src.rglob("*")):
                rel = f.relative_to(src)
                if (not f.is_file() or f.is_symlink() or rel.parts[0] in _MODEL_DIRS
                        or rel.as_posix() == "manifest.json"):
                    continue
                with open(f, "rb") as rf:
                    cw.add(rel.as_posix(), rf)
            cw.close()
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)
    return out, generated


def import_secure_bundle(archive: str | Path, passphrase: str,
                         dest_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Decrypt ``archive`` into ``dest_dir`` (the project's live corpus dir).

    Raises ``WrongPassphraseOrTampered`` / ``ContainerError`` (nothing is
    written) on a bad passphrase or corrupt file; entry names are validated so
    a crafted bundle cannot write outside ``dest_dir``.
    """
    dest = Path(dest_dir)
    if dest.exists() and any(dest.iterdir()) and not overwrite:
        raise FileExistsError(f"Corpus already exists at {dest}; pass overwrite=True to replace.")
    with ContainerReader(archive, passphrase) as cr:
        names = cr.names()
        if "manifest.json" not in names:
            raise ValueError("No manifest.json found in bundle")
        root = dest.resolve()
        for n in names:  # validate everything before touching disk
            target = (root / n).resolve()
            if n.startswith("/") or not target.is_relative_to(root) or target == root:
                raise ValueError(f"Bundle contains unsafe path: {n}")
        staging = dest.with_name(dest.name + ".importing")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            for n in names:
                t = staging / n
                t.parent.mkdir(parents=True, exist_ok=True)
                t.write_bytes(cr.read(n))  # authenticates each frame
            if dest.exists():
                shutil.rmtree(dest)
            relabel_imported_sources(staging)
            staging.replace(dest)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return dest
