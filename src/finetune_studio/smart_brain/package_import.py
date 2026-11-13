"""Open a Smart Brain package to edit it: its index becomes a project's starting knowledge, its files come back.

A standalone server package (``build_package``; encrypted with a passphrase, or plaintext) is unpacked inside the
project (ciphertext only for an encrypted one), its corpus is restored as the project's index (passages, vectors,
word index, parsed sources, identity), and its original files, when it ships them, go back into the file library in
their folders, ready to parse. Its version record joins the project's package history, so the next export can be
"a new version" of it: same package id, the next version, a list of what changed.
"""
from __future__ import annotations

import contextlib
import json
import re
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

from finetune_studio.data.fs.paths import brain_index_dir, project_dir

ENC_NAME = "corpus.enc"


def _names(archive: Path) -> list[str]:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            return zf.namelist()
    with tarfile.open(archive) as tf:
        return tf.getnames()


def check_archive(archive: str | Path, shown_name: str = "") -> None:
    """Read the whole archive's table of contents once: a truncated download or a file that is not a tar/zip
    archive raises ``ValueError`` with a reason the user can act on (it raised EOFError / BadZipFile -> HTTP 500)."""
    name = shown_name or Path(archive).name
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                bad = zf.testzip()
            if bad:
                raise ValueError(f"{name} is damaged ({bad} does not read back): download it again")
            return
        with tarfile.open(archive) as tf:
            for _ in tf:       # walks to the end: a cut-off file fails here
                pass
    except ValueError:
        raise
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile) as e:
        raise ValueError(f"{name} is not a Smart Brain package or index, or it is damaged (cut off or not a .tar.gz "
                         f"/ .zip archive): download it again ({type(e).__name__})") from e


def has_version_record(archive: str | Path) -> bool:
    """A package with a ``package.json`` beside its corpus (exported with package versions)."""
    try:
        return any(n.endswith("/package.json") or n == "package.json" for n in _names(Path(archive)))
    except (OSError, EOFError, tarfile.TarError, zipfile.BadZipFile):
        return False


def is_server_package(archive: str | Path) -> bool:
    """A standalone server package (server.py beside corpus/), not a legacy index bundle."""
    try:
        names = _names(Path(archive))
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return any(n.endswith("/server.py") or n == "server.py" for n in names) and any("/corpus/" in n for n in names)


class _Package:
    """Reads members of an unpacked package's corpus (decrypted in memory for an encrypted one)."""

    def __init__(self, root: Path, passphrase: str | None):
        self.root = root
        enc = root / "corpus" / ENC_NAME
        self._reader = None
        if enc.is_file():
            if not passphrase:
                raise ValueError("this package is encrypted: a passphrase is required")
            from .brain_container import ContainerReader
            self._reader = ContainerReader(enc, passphrase)

    def has(self, name: str) -> bool:
        return name in self._reader if self._reader is not None else (self.root / "corpus" / name).is_file()

    def read(self, name: str) -> bytes:
        return self._reader.read(name) if self._reader is not None else (self.root / "corpus" / name).read_bytes()

    def names(self) -> list[str]:
        if self._reader is not None:
            return self._reader.names()
        base = self.root / "corpus"
        return [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()]

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()


def _unpack(archive: Path, dest: Path) -> Path:
    def safe(name: str) -> bool:
        parts = Path(name).parts
        return bool(parts) and not name.startswith("/") and ".." not in parts
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest, [n for n in zf.namelist() if safe(n) and "/embedder/" not in n and "/reranker/" not in n])
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, members=[m for m in tf.getmembers() if safe(m.name) and (m.isfile() or m.isdir())
                                         and "/embedder/" not in m.name and "/reranker/" not in m.name], filter="data")
    server = sorted(dest.rglob("server.py"), key=lambda p: len(p.parts))
    if not server:
        raise ValueError("not a Smart Brain package (no server.py inside)")
    return server[0].parent


def _portable_embedder(manifest: dict) -> None:
    """A package built with models names its embedder and reranker by short id; the studio's index wants
    ``shared:<kind>:<id>``. A reranker left as a bare short id made the next export with models fail ("could not be
    bundled": it was looked up as a Hugging Face repo id)."""
    settings = manifest.get("rag_settings") or {}
    for holder, key, kind in ((manifest.get("embedding_model") or {}, "name", "embedder"),
                              (settings, "embedder", "embedder"), (settings, "reranker", "reranker")):
        ref = str(holder.get(key) or "")
        if ref and not ref.startswith(("shared:", "embedder_local:", "reranker_local:")):
            from finetune_studio.data import shared_models as sm
            with contextlib.suppress(Exception):   # not in this studio's store: the name stays as it was
                if Path(sm.resolve(ref, kind)).is_dir():
                    holder[key] = f"shared:{kind}:{ref}"


def _restore_index(pkg: _Package, target: Path) -> dict[str, Any]:
    import pandas as pd

    from .bm25 import BM25Index

    manifest = json.loads(pkg.read("manifest.json").decode("utf-8"))
    rows = [json.loads(line) for line in pkg.read("chunks.jsonl").decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("the package holds no passages")
    if target.exists():
        shutil.rmtree(target)
    (target / "sources").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(target / "chunks.parquet", index=False)
    vectors = pkg.has("vectors.npy") and pkg.has("vectors.idx.json")
    if vectors:
        (target / "vectors.npy").write_bytes(pkg.read("vectors.npy"))
        (target / "vectors.idx.json").write_bytes(pkg.read("vectors.idx.json"))
    if pkg.has("bm25.json"):
        (target / "bm25.json").write_bytes(pkg.read("bm25.json"))
    else:
        (target / "bm25.json").write_text(json.dumps(BM25Index.build([str(r.get("text") or "") for r in rows]).to_dict()))
    for name in pkg.names():
        if name.startswith("sources/") and ".." not in Path(name).parts:
            dest = target / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(pkg.read(name))
    record = manifest.pop("package", None) or {}
    settings = _settings_from(manifest, pkg)
    extra = manifest.setdefault("extra", {})
    extra["package"] = record
    if manifest.get("scope"):
        extra["package_scope"] = manifest.pop("scope")
    identity = manifest.pop("identity", None)
    if identity:
        (target / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    for key in ("search_layers", "original_files"):
        manifest.pop(key, None)
    _portable_embedder(manifest)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"record": record, "settings": settings, "vectors": vectors,
            "documents": int(manifest.get("documents") or 0), "chunks": len(rows)}


def _settings_from(manifest: dict, pkg: _Package) -> dict[str, Any]:
    """What the package was exported with (its scope, originals, hidden names, audience, encryption): the history
    keeps them, so "a new version of" an opened package keeps them too."""
    scope = manifest.get("scope") or {}
    record = manifest.get("package") or {}
    policy = record.get("policy") or {}
    out: dict[str, Any] = {"encrypted": pkg._reader is not None,
                           "include_originals": bool(manifest.get("original_files")),
                           "originals": int(manifest.get("original_files") or 0)}
    if scope:
        out["scope"] = scope.get("label") or "all documents"
        out["scope_spec"] = ({k: list(scope.get(k) or []) for k in ("include", "exclude", "tags", "document_ids")}
                             if out["scope"] != "all documents" else {})
    hide = policy.get("hide_file_names")
    if hide is None:
        try:
            hide = bool(json.loads((pkg.root / "rag.config.json").read_text(encoding="utf-8")).get("hide_file_names"))
        except (OSError, ValueError):
            hide = None
    if hide is not None:
        out["hide_file_names"] = bool(hide)
    if policy.get("audience") or record.get("audience"):
        out["audience"] = policy.get("audience") or record.get("audience")
    return out


def _folder_id(pid: str, name: str, cache: dict[str, str]) -> str:
    from finetune_studio.data.fs import file_library as fl

    if name not in cache:
        cache[name] = fl.ensure_folder(pid, name)["id"]     # a path (public/faq): the folders above it too
    return cache[name]


def _restore_files(pid: str, pkg: _Package, record: dict) -> list[dict]:
    """The package's original files back into the library (in their folders), staged for parsing."""
    from finetune_studio.data.fs import file_library as fl
    from finetune_studio.knowledge import sources

    if not pkg.has("originals.json"):
        return []
    paths = {d.get("document_id"): d.get("path") or "" for d in record.get("documents") or []}
    folders: dict[str, str] = {}
    out = []
    for row in json.loads(pkg.read("originals.json").decode("utf-8")):
        did, name = str(row.get("document_id") or ""), str(row.get("filename") or "")
        if not did or not name or "/" in name or ".." in name:
            continue
        data = pkg.read(f"originals/{did}")
        existing = fl.find_existing_hash(pid, fl.sha256_of_bytes(data))
        if existing:
            out.append({"filename": name, "status": "already here", "file_id": existing["id"]})
            continue
        meta = fl.write_uploaded_file(pid, data, name, mime_hint=row.get("mime_type") or None, uploaded_by="package")
        folder = paths.get(did, "").rsplit("/", 1)[0] if "/" in paths.get(did, "") else ""
        if folder:
            fl.move_file_to_folder(pid, meta.file_id, _folder_id(pid, folder, folders))
        if sources.is_parseable(name):
            sources.stage(pid, meta.file_id)
        out.append({"filename": name, "status": "restored", "file_id": meta.file_id, "folder": folder})
    return out


def _display_name(record: dict, file_name: str) -> str:
    """The package's own name (inside its manifest since 0.5), else the file name the user chose (never the staged
    temp name ``upload-<time>.tar.gz``), without the archive extension and the export's own suffix."""
    if record.get("name"):
        return str(record["name"])
    stem = re.sub(r"(\.tar\.gz|\.tgz|\.tar|\.zip)$", "", Path(file_name).name, flags=re.IGNORECASE)
    stem = re.sub(r"-smart-brain-[0-9a-f]{8}-v[\w.+-]+$", "", stem)
    return stem or Path(file_name).name


def import_package(pid: str, archive: str | Path, passphrase: str | None, *, overwrite: bool = False,
                   display_name: str | None = None) -> dict[str, Any]:
    """Open a server package as ``pid``'s starting knowledge. The caller registers the index afterwards
    (``knowledge.index.register_import``, ``identity.adopt_imported``). ``display_name`` is the file name the user
    uploaded (the archive itself may be a staged temp copy)."""
    from . import package_meta

    archive = Path(archive)
    target = brain_index_dir(pid)
    if (target / "manifest.json").is_file() and not overwrite:
        raise FileExistsError(f"this project already has an index at {target}; replace it to open the package")
    staging = project_dir(pid) / "brain-import-staging" / f"pkg-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        root = _unpack(archive, staging)
        pkg = _Package(root, passphrase)
        try:
            restored = _restore_index(pkg, target)
            files = _restore_files(pid, pkg, restored["record"])
            changes_md = pkg.read("CHANGES.md").decode("utf-8") if pkg.has("CHANGES.md") else ""
        finally:
            pkg.close()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    record = restored["record"]
    if record.get("package_id"):
        shown = Path(display_name).name if display_name else archive.name
        package_meta.add_history(package_meta.package_dir(pid), {
            **record, **restored["settings"], "origin": "opened", "filename": shown,
            "name": _display_name(record, shown), "changes_md": changes_md})
    return {"package_id": record.get("package_id"), "version": record.get("version"),
            "documents": restored["documents"], "chunks": restored["chunks"], "vectors": restored["vectors"],
            "files": files, "restored_files": sum(1 for f in files if f["status"] == "restored"),
            "note": ("" if restored["vectors"] else
                     "the package has no meaning-search vectors: Build / Update the index to embed it")}


def read_record(archive: str | Path, passphrase: str | None, workdir: Path) -> dict[str, Any]:
    """A package's version record (with its document list), for "new version of" an uploaded archive."""
    staging = workdir / f"pkg-{time.time_ns()}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        pkg = _Package(_unpack(Path(archive), staging), passphrase)
        try:
            manifest = json.loads(pkg.read("manifest.json").decode("utf-8"))
            record = manifest.get("package") or {}
            if record.get("package_id"):
                record = {**record, **_settings_from(manifest, pkg)}
        finally:
            pkg.close()
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not record.get("package_id"):
        raise ValueError("this package has no version record (exported before package versions)")
    return record

