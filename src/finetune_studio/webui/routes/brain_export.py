"""Smart Brain export: the encrypted studio bundle (``.ftsrag``), import, and the standalone server package.

Mounted under ``/api/projects``:

  POST /{pid}/brain/bundle               -- export the index as an encrypted .ftsrag bundle
  GET  /{pid}/brain/bundle/download      -- download it
  POST /{pid}/brain/import               -- upload a bundle (+passphrase)
  GET  /{pid}/brain/export-config        -- standalone server defaults (+ this project's override)
  PUT  /{pid}/brain/export-config        -- save them
  POST /{pid}/brain/standalone           -- build the standalone Smart Brain server package
  GET  /{pid}/brain/standalone/download  -- download it
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio.data.fs.paths import brain_index_dir, project_dir

log = logging.getLogger(__name__)
router = APIRouter()


def _corpus_dir(pid: str) -> Path:
    return brain_index_dir(pid)


def _project_404(pid: str) -> JSONResponse | None:
    """Return a 404 response when the project does not exist, else None.

    This module answers errors with ``JSONResponse`` (see the corpus 404s
    below), so the guard matches that style. Called before any corpus
    directory is touched — otherwise a bad pid gets a real ``corpus_dir``
    path back with a 200.
    """
    from finetune_studio import db
    if not db.get_project(pid):
        return JSONResponse({"error": "project not found"}, status_code=404)
    return None


class BundleRequest(BaseModel):
    name: str | None = None
    passphrase: str | None = None  # blank -> generated, returned once, never stored


_SAFE_BUNDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.ftsrag$")


def _bundle_dir(pid: str) -> Path:
    return project_dir(pid) / "brain-bundles"


@router.post("/{pid}/brain/bundle")
async def brain_bundle(pid: str, body: BundleRequest | None = None):
    """Export the corpus as an encrypted ``.ftsrag`` file inside the project dir.

    AES-256-GCM, key derived from a passphrase that is returned once (when
    generated here) and never stored. Download it with ``GET …/brain/bundle/download``.
    """
    from finetune_studio.smart_brain import secure_bundle as sb

    missing = _project_404(pid)
    if missing is not None:
        return missing
    req = body or BundleRequest()
    corpus = _corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        return JSONResponse({"error": "no corpus to bundle"}, status_code=404)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", req.name or "").strip(".-")[:48] or f"{pid}-corpus"
    out = _bundle_dir(pid) / f"{stem}{sb.BUNDLE_EXT}"
    try:
        path, generated = await asyncio.to_thread(
            sb.export_secure_bundle, corpus, out, req.passphrase or None)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        log.exception("bundle export failed")  # never includes the passphrase
        return JSONResponse({"error": f"bundle export failed: {type(e).__name__}"}, status_code=500)
    return JSONResponse(
        {"filename": path.name, "size": path.stat().st_size, "encrypted": True,
         "passphrase": generated,
         "download_url": f"/api/projects/{pid}/brain/bundle/download?file={path.name}"},
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{pid}/brain/bundle/download")
async def brain_bundle_download(pid: str, file: str):
    from fastapi.responses import FileResponse

    missing = _project_404(pid)
    if missing is not None:
        return missing
    if not _SAFE_BUNDLE.match(file):
        return JSONResponse({"error": "invalid bundle filename"}, status_code=400)
    path = _bundle_dir(pid) / file
    if not path.is_file():
        return JSONResponse({"error": "bundle not found — export again"}, status_code=404)
    return FileResponse(path=str(path), media_type="application/octet-stream",
                        filename=file, headers={"Cache-Control": "no-store"})


@router.post("/{pid}/brain/import")
async def brain_import(pid: str, file: UploadFile, overwrite: bool = False,
                     passphrase: str = Form("")):
    """Import a corpus bundle: encrypted ``.ftsrag`` (needs ``passphrase``) or a
    legacy ``.tar/.tar.gz/.zip``.

    The upload is staged inside the project directory (never the system temp
    dir) and removed afterwards. The corpus is replaced only when
    ``overwrite=true`` or none exists. Returns the same stats as
    :func:`brain_status`.
    """
    from finetune_studio.smart_brain import BrainIndex
    from finetune_studio.smart_brain import secure_bundle as sb
    from finetune_studio.smart_brain.brain_container import ContainerError

    missing = _project_404(pid)
    if missing is not None:
        return missing
    filename = file.filename or "bundle"
    name_lower = filename.lower()
    # Path.suffix only sees the last dot, so "x.tar.gz" reports ".gz" —
    # match compound suffixes explicitly.
    if name_lower.endswith(sb.BUNDLE_EXT):
        suffix = sb.BUNDLE_EXT
    elif name_lower.endswith((".tar.gz", ".tgz")):
        suffix = ".tar.gz"
    elif name_lower.endswith(".tar"):
        suffix = ".tar"
    elif name_lower.endswith(".zip"):
        suffix = ".zip"
    else:
        return JSONResponse(
            {"error": f"unsupported archive format: {Path(filename).suffix.lower()}"},
            status_code=400,
        )

    staging = project_dir(pid) / "brain-import-staging"
    staging.mkdir(parents=True, exist_ok=True)
    tmp_path = staging / f"upload-{time.time_ns()}{suffix}"

    def _stage() -> None:
        with open(tmp_path, "wb") as tmp:
            shutil.copyfileobj(file.file, tmp)

    await asyncio.to_thread(_stage)

    index = BrainIndex(_corpus_dir(pid))
    try:
        if suffix == sb.BUNDLE_EXT:
            if not passphrase:
                return JSONResponse({"error": "this bundle is encrypted; a passphrase is required"},
                                    status_code=400)
            if index.exists() and not overwrite:
                raise FileExistsError(
                    f"Corpus already exists at {index.dir}; pass overwrite=True to replace.")
            await asyncio.to_thread(sb.import_secure_bundle, tmp_path, passphrase,
                                    index.dir, overwrite=True)
            stats = index.load().manifest.to_json()
            stats = {"documents": stats.get("documents", 0), "chunks": stats.get("chunks", 0)}
        else:
            stats = index.import_bundle(tmp_path, overwrite=overwrite)
    except FileNotFoundError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except FileExistsError as e:
        return JSONResponse({"error": str(e)}, status_code=409)
    except ContainerError as e:  # wrong passphrase / tampered / not a bundle
        return JSONResponse({"error": str(e)}, status_code=400)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        log.exception("bundle import failed")
        return JSONResponse({"error": f"bundle import failed: {type(e).__name__}"}, status_code=500)
    finally:
        tmp_path.unlink(missing_ok=True)

    # Persist a fresh activity-feed row so the user sees this in the drawer.
    try:
        from finetune_studio import db
        db.record_activity_event(
            kind="brain_build",
            operation=f"POST /api/projects/{pid}/brain/import",
            method="POST",
            path=f"/api/projects/{pid}/brain/import",
            project_id=pid,
            status="done",
            http_status=200,
            message=(
                f"Imported RAG bundle: {stats.get('documents', 0)} docs, "
                f"{stats.get('chunks', 0)} chunks"
            ),
        )
    except Exception:
        log.exception("failed to record import activity event")

    return stats


class ExportConfigPatch(BaseModel):
    """Partial export settings (all optional; see ``BrainExportConfig``)."""

    archive_format: str | None = None
    include_models: bool | None = None
    include_reranker: bool | None = None
    reranker_enabled: bool | None = None
    device: str | None = None
    host: str | None = None
    port: int | None = None
    top_k: int | None = None
    encrypt: bool | None = None
    kdf_log_n: int | None = None


class ExportConfigSave(BaseModel):
    scope: str = "studio"            # "studio" defaults | "project" override
    values: ExportConfigPatch = Field(default_factory=ExportConfigPatch)
    clear: bool = False              # project scope only: drop the override


class StandaloneRequest(ExportConfigPatch):
    """Export request: optional overrides + the (never stored) passphrase."""

    name: str | None = None
    passphrase: str | None = Field(default=None, repr=False)


def _patch_values(p: ExportConfigPatch) -> dict[str, Any]:
    return {k: v for k, v in p.model_dump().items() if v is not None}


def _require_project(pid: str) -> dict:
    from finetune_studio import db
    proj = db.get_project(pid)
    if not proj:
        raise HTTPException(status_code=404, detail="project not found")
    return proj


def _export_config_payload(pid: str) -> dict[str, Any]:
    from finetune_studio.smart_brain import export_config as ec
    return {
        "defaults": ec.load_defaults().model_dump(),
        "project_override": ec.load_project_override(pid),
        "effective": ec.effective_config(pid).model_dump(),
    }


@router.get("/{pid}/brain/export-config")
async def brain_export_config(pid: str) -> dict[str, Any]:
    """Studio-wide export defaults, this project's override, and the merge."""
    _require_project(pid)
    return _export_config_payload(pid)


@router.put("/{pid}/brain/export-config")
async def brain_export_config_save(pid: str, body: ExportConfigSave) -> dict[str, Any]:
    """Save studio defaults (``scope=studio``) or a project override
    (``scope=project``; ``clear=true`` removes it). Never stores a passphrase."""
    from finetune_studio.smart_brain import export_config as ec
    _require_project(pid)
    try:
        if body.scope == "studio":
            ec.save_defaults(_patch_values(body.values))
        elif body.scope == "project":
            ec.save_project_override(pid, None if body.clear else _patch_values(body.values))
        else:
            raise HTTPException(status_code=400, detail="scope must be 'studio' or 'project'")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _export_config_payload(pid)


def _package_dir(pid: str) -> Path:
    return Path("output") / "projects" / pid / "brain-packages"


_SAFE_PKG_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.(tar\.gz|tar|zip)$")


@router.post("/{pid}/brain/standalone")
async def brain_standalone(pid: str, body: StandaloneRequest | None = None):
    """Build a hostable, self-installing RAG package and return its metadata.

    Options come from studio defaults < project override < this request.
    Encryption (AES-256-GCM, key from a passphrase) is ON unless the request
    or saved settings turn it off. A blank ``passphrase`` generates one; it is
    returned **once** in this response and never stored. Download the file via
    ``GET …/brain/standalone/download?file=<filename>``.
    """
    from fastapi.responses import JSONResponse

    from finetune_studio.smart_brain import export_config as ec
    from finetune_studio.smart_brain.server_package import build_package

    proj = _require_project(pid)
    req = body or StandaloneRequest()
    corpus = _corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        raise HTTPException(
            status_code=404,
            detail="no corpus yet — build one first (section 2 on this page)",
        )
    try:
        cfg = ec.effective_config(pid, _patch_values(req))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    title = req.name or proj["name"] or pid
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip(".-")[:48] or "project"
    out_dir = _package_dir(pid)
    suffix = "-with-models" if cfg.include_models else ""
    plain = "" if cfg.encrypt else "-PLAINTEXT"
    ext = "zip" if cfg.archive_format == "zip" else cfg.archive_format
    out_path = out_dir / f"{safe}-smart-brain{suffix}{plain}.{ext}"
    try:
        result = await asyncio.to_thread(
            build_package, corpus, out_path, name=title, config=cfg,
            passphrase=(req.passphrase or None))
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        log.exception("rag package build failed")  # never includes the passphrase
        raise HTTPException(status_code=500, detail=f"package build failed: {type(e).__name__}") from e
    return JSONResponse(
        {
            "filename": result.path.name,
            "download_url": f"/api/projects/{pid}/brain/standalone/download?file={result.path.name}",
            "size": result.size,
            "encrypted": result.encrypted,
            # Shown once; only set when generated here.
            "passphrase": result.passphrase,
            "config": cfg.model_dump(),
        },
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{pid}/brain/standalone/download")
async def brain_standalone_download(pid: str, file: str):
    """Download a previously built package (filename validated; no paths)."""
    from fastapi.responses import FileResponse

    _require_project(pid)
    if not _SAFE_PKG_NAME.match(file):
        raise HTTPException(status_code=400, detail="invalid package filename")
    path = _package_dir(pid) / file
    if not path.is_file():
        raise HTTPException(status_code=404, detail="package not found — export again")
    media = "application/zip" if file.endswith(".zip") else "application/gzip"
    return FileResponse(path=str(path), media_type=media, filename=file,
                        headers={"Cache-Control": "no-store"})


