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
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from finetune_studio.data.fs.paths import brain_index_dir, project_dir
from finetune_studio.smart_brain.scope import DocumentScope
from finetune_studio.webui.errors import ApiError

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
        return JSONResponse({"detail": "project not found"}, status_code=404)
    return None


class BundleRequest(BaseModel):
    name: str | None = None
    passphrase: str | None = None  # blank -> generated, returned once, never stored


_SAFE_BUNDLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.ftsrag$")


def _identity_load(pid: str) -> dict | None:
    from finetune_studio.smart_brain import identity as brain_identity

    saved = brain_identity.load(pid)
    return saved if brain_identity.is_set(saved) else None


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
        return JSONResponse({"detail": "no corpus to bundle"}, status_code=404)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", req.name or "").strip(".-")[:48] or f"{pid}-corpus"
    out = _bundle_dir(pid) / f"{stem}{sb.BUNDLE_EXT}"
    try:
        path, generated = await asyncio.to_thread(
            sb.export_secure_bundle, corpus, out, req.passphrase or None, identity=_identity_load(pid))
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=422)
    except Exception as e:
        log.exception("bundle export failed")  # never includes the passphrase
        return JSONResponse({"detail": f"bundle export failed: {type(e).__name__}"}, status_code=500)
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
        return JSONResponse({"detail": "invalid bundle filename"}, status_code=400)
    path = _bundle_dir(pid) / file
    if not path.is_file():
        return JSONResponse({"detail": "bundle not found — export again"}, status_code=404)
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
            {"detail": f"unsupported archive format: {Path(filename).suffix.lower()}"},
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
    from finetune_studio.smart_brain import package_import

    opened: dict | None = None
    try:
        # The file is checked before anything else (and before the "replace the index?" question): a damaged or
        # wrong file is a 400 with a plain reason, not a 500 after the user agreed to replace their index.
        if suffix != sb.BUNDLE_EXT:
            await asyncio.to_thread(package_import.check_archive, tmp_path, filename)
        if suffix != sb.BUNDLE_EXT and await asyncio.to_thread(package_import.is_server_package, tmp_path):
            if (index.exists() or (index.dir / "manifest.json").is_file()) and not overwrite:   # keyword-only too
                record = await asyncio.to_thread(package_import.read_record, tmp_path, passphrase or None,
                                                 project_dir(pid) / "brain-import-staging") \
                    if await asyncio.to_thread(package_import.has_version_record, tmp_path) else {}
                return JSONResponse({"detail": "this project already has an index: opening the package replaces it",
                                     "code": "index_exists",
                                     "package": {k: record.get(k) for k in ("package_id", "version", "name")}},
                                    status_code=409)
            opened = await asyncio.to_thread(package_import.import_package, pid, tmp_path, passphrase or None,
                                             overwrite=True, display_name=filename)
        elif suffix == sb.BUNDLE_EXT:
            if not passphrase:
                return JSONResponse({"detail": "this bundle is encrypted; a passphrase is required"},
                                    status_code=400)
            if index.exists() and not overwrite:
                raise FileExistsError(
                    f"Corpus already exists at {index.dir}; pass overwrite=True to replace.")
            await asyncio.to_thread(sb.import_secure_bundle, tmp_path, passphrase,
                                    index.dir, overwrite=True)
        else:
            await asyncio.to_thread(index.import_bundle, tmp_path, overwrite=overwrite)
        from finetune_studio.knowledge import index as knowledge_index

        stats = await asyncio.to_thread(knowledge_index.register_import, pid, bundle=filename)
        if opened is not None:
            stats = {**stats, "package": opened}
        from finetune_studio.smart_brain import identity as brain_identity
        await asyncio.to_thread(brain_identity.adopt_imported, pid, index.dir)
    except FileNotFoundError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except FileExistsError as e:
        return JSONResponse({"detail": str(e)}, status_code=409)
    except ContainerError as e:  # wrong passphrase / tampered / not a bundle
        return JSONResponse({"detail": str(e)}, status_code=400)
    except ValueError as e:
        return JSONResponse({"detail": str(e)}, status_code=400)
    except (EOFError, OSError, tarfile.TarError, zipfile.BadZipFile) as e:   # damaged past the first check
        return JSONResponse({"detail": f"{filename} could not be read: it is damaged or not a Smart Brain package "
                                       f"({type(e).__name__})"}, status_code=400)
    except Exception as e:
        log.exception("bundle import failed")
        return JSONResponse({"detail": f"bundle import failed: {type(e).__name__}"}, status_code=500)
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
    semantic_search: bool | None = None
    keyword_search: bool | None = None
    answer_enabled: bool | None = None
    llm_provider: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_headers: dict[str, str] | None = None
    llm_body: dict[str, Any] | None = None
    include_originals: bool | None = None
    originals_inline_max_mb: int | None = None
    hide_file_names: bool | None = None
    image_search: bool | None = None
    answer_images: bool | None = None
    llm_connection: str | None = None   # copy endpoint, model, headers and body from a Settings connection (no key)


class ExportConfigSave(BaseModel):
    scope: str = "studio"            # "studio" defaults | "project" override
    values: ExportConfigPatch = Field(default_factory=ExportConfigPatch)
    clear: bool = False              # project scope only: drop the override


class StandaloneRequest(ExportConfigPatch):
    """Export request: optional overrides + the (never stored) passphrase."""

    name: str | None = Field(default=None, max_length=120)   # the package's name (shown in its versions; file name)
    audience: Literal["internal", "public", "other"] | None = None   # a label: who uses it (staff / customers)
    passphrase: str | None = Field(default=None, repr=False)
    scope: DocumentScope | None = None    # which documents go in (None: all, or the base version's scope; {}: all)
    new_version_of: str | None = None     # a package id from this project's package history: keep it, bump the version
    version: str | None = None            # override the bumped version
    allow_widen: bool = False             # confirmed: this version may open up more than its base (see package_versions)
    new_passphrase: bool = False          # confirmed: a new version of an encrypted package gets another passphrase


def _patch_values(p: ExportConfigPatch) -> dict[str, Any]:
    values = {k: v for k, v in p.model_dump().items()
              if v is not None and k not in ("scope", "name", "audience", "passphrase", "new_version_of", "version", "allow_widen",
                        "new_passphrase")}
    conn = values.pop("llm_connection", None)
    if conn:
        values = {**connection_llm_settings(conn), **values}   # a typed endpoint or model wins
    return values


def connection_llm_settings(connection_id: str) -> dict[str, Any]:
    """Answer-model settings of a Settings → AI assistant API connection: endpoint, model, headers, body.

    The API key is never copied (a package is handed to other people): it is ``RAG_LLM_API_KEY`` where it runs.
    """
    from finetune_studio.models.manager import get_manager

    row = get_manager().get_provider(connection_id)
    if row is None or row.get("kind") != "openai_compat":
        raise HTTPException(status_code=422, detail=f"{connection_id!r} is not an API connection (Settings → AI assistant)")
    extra = row.get("extra") or {}
    return {"llm_base_url": (row.get("base_url") or "").rstrip("/"), "llm_model": row.get("model_id") or "",
            "llm_headers": dict(extra.get("headers") or {}), "llm_body": dict(extra.get("body") or {})}


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


@router.get("/{pid}/brain/packages")
def brain_packages(pid: str) -> dict[str, Any]:
    """This project's package history: every package id with its versions (newest first), and the latest
    version's change report (``CHANGES.md``)."""
    from finetune_studio.smart_brain import package_meta as pm

    _require_project(pid)
    rows = pm.read_history(_package_dir(pid))
    last = max(rows, key=lambda r: r.get("created") or "", default=None)
    return {"packages": pm.packages(_package_dir(pid)),
            "latest": ({k: last.get(k) for k in ("package_id", "version", "parent_version", "created", "filename",
                                                  "origin", "changes")} | {"changes_md": last.get("changes_md") or ""})
            if last else None}


@router.post("/{pid}/brain/package-base")
async def brain_package_base(pid: str, file: UploadFile, passphrase: str = Form("")):
    """Add an exported package archive (+ its passphrase when encrypted) to the history, as the base for
    "New version of an existing package". Only its version record and document list are kept, never its content."""
    from finetune_studio.smart_brain import package_import
    from finetune_studio.smart_brain import package_meta as pm

    _require_project(pid)
    staging = project_dir(pid) / "brain-import-staging"
    staging.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "package.tar.gz").name
    tmp = staging / f"base-{time.time_ns()}-{re.sub(r'[^A-Za-z0-9._-]+', '-', name)}"

    def _stage() -> None:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(file.file, out)

    await asyncio.to_thread(_stage)
    try:
        record = await asyncio.to_thread(package_import.read_record, tmp, passphrase or None, staging)
    except Exception as e:  # wrong passphrase, not a package, damaged archive: the reason is what the user needs
        raise HTTPException(status_code=400, detail=f"cannot read that package: {e}") from e
    finally:
        tmp.unlink(missing_ok=True)
    await asyncio.to_thread(pm.add_history, _package_dir(pid), {**record, "origin": "uploaded", "filename": name,
                                                                "name": record.get("name") or name})
    return {k: record.get(k) for k in ("package_id", "version", "parent_version", "created")} | {
        "documents": len(record.get("documents") or [])}


@router.get("/{pid}/brain/package-documents")
def brain_package_documents(pid: str) -> dict[str, Any]:
    """The index's documents with their folder path and tags, folders and tags with counts: the Export card's
    picker for a package's audience scope."""
    from finetune_studio.smart_brain import scope as brain_scope

    _require_project(pid)
    corpus = _corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        # Not built yet: nothing to pick from. An empty overview, not a 404 the Smart Brain page logged as an error
        # on every visit before the first build.
        return {**brain_scope.overview([]), "indexed": False}
    return {**brain_scope.overview(brain_scope.documents(corpus, pid)), "indexed": True}


def _package_dir(pid: str) -> Path:
    from finetune_studio.smart_brain.package_meta import package_dir
    return package_dir(pid)


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
    from finetune_studio.smart_brain import identity as brain_identity
    from finetune_studio.smart_brain.server_package import build_package

    proj = _require_project(pid)
    req = body or StandaloneRequest()
    corpus = _corpus_dir(pid)
    if not (corpus / "manifest.json").is_file():
        raise HTTPException(
            status_code=404,
            detail="no corpus yet — build one first (section 2 on this page)",
        )
    from finetune_studio.knowledge.lint import SUPERSEDED_HINT, lint, superseded_files

    filenames = superseded_files(await asyncio.to_thread(lint, pid))
    if filenames:
        raise HTTPException(
            status_code=409,
            detail="Review superseded documents before packaging: " + ", ".join(filenames) + ". " + SUPERSEDED_HINT,
        )
    from finetune_studio.smart_brain import package_meta as pm
    from finetune_studio.smart_brain import package_versions as pv

    out_dir = _package_dir(pid)
    base = None
    if req.new_version_of:
        base = pm.latest(out_dir, req.new_version_of)
        if base is None:
            raise HTTPException(status_code=404, detail=f"no package {req.new_version_of!r} in this project's history")
    patch = _patch_values(req)
    if base:   # a new version keeps what its base was for, unless the request says otherwise
        was = pv.base_settings(base)
        patch.setdefault("include_originals", was["include_originals"])
        patch.setdefault("hide_file_names", was["hide_file_names"])
    try:
        cfg = ec.effective_config(pid, patch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    title = (req.name or "").strip() or (base or {}).get("name") or proj["name"] or pid
    audience = req.audience or (base or {}).get("audience") or ""
    from finetune_studio.smart_brain import originals as brain_originals
    from finetune_studio.smart_brain import scope as brain_scope

    docs = await asyncio.to_thread(brain_scope.documents, corpus, pid)
    scope_req = pv.inherit_scope(req.scope, base)
    try:
        keep, scope_record = brain_scope.resolve(scope_req, docs)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    widens = pv.widening(base, docs, keep, include_originals=cfg.include_originals,
                         hide_file_names=cfg.hide_file_names, audience=audience, encrypt=cfg.encrypt) if base else []
    if widens and not req.allow_widen:
        raise ApiError(409, f"Version {pm.bump_minor(base.get('version'))} would open up more than version "
                            f"{base.get('version')} of {base.get('name') or base['package_id']}: "
                            + "; ".join(w["text"] for w in widens)
                            + ". Confirm it (allow_widen), or export a new package instead.",
                       code="widens_package", widens=widens)
    problem = pv.passphrase_problem(base, encrypt=cfg.encrypt, passphrase=req.passphrase,
                                    new_passphrase=req.new_passphrase)
    if problem:
        raise ApiError(422, problem, code="passphrase_required")
    if base and cfg.encrypt and req.passphrase and not req.new_passphrase:
        same = await asyncio.to_thread(pv.passphrase_matches_base, base, out_dir, req.passphrase)
        if same is False:
            raise ApiError(422, f"That is not the passphrase of version {base.get('version')}: running copies "
                                "could not open the new version. Enter the old passphrase, or choose a new one "
                                "on purpose.", code="passphrase_mismatch")
    probes = (await asyncio.to_thread(brain_scope.probe_queries, corpus, {d.document_id for d in docs} - keep)
              if keep is not None else {})
    originals = await asyncio.to_thread(brain_originals.collect, corpus, pid) if cfg.include_originals else []
    doc_rows = await asyncio.to_thread(pm.document_manifest, corpus, pid, keep)
    try:
        record = pm.make_record(base=base, documents=doc_rows, version=req.version)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    diff = pm.changes(base["documents"] if base else None, doc_rows)
    record["changes"] = pm.summary(diff)
    shipped_originals = sum(1 for o in originals if keep is None or o.document_id in keep)
    record["policy"] = pv.policy(scope_record, originals=shipped_originals, hide_file_names=cfg.hide_file_names,
                                 audience=audience, encrypted=cfg.encrypt)
    record["widens"] = widens
    record["passphrase_changed"] = bool(base and cfg.encrypt and req.new_passphrase)
    pm.name_record(record, base, name=title, project=proj.get("name") or pid, taken=pm.taken_roots(out_dir))
    record["audience"] = audience
    changes_md = pm.changes_markdown(record, diff) + pv.widening_markdown(record["parent_version"], widens)
    changes_md_public = (pm.changes_markdown(record, diff, names=False)
                         + pv.widening_markdown(record["parent_version"], widens, names=False))
    ext = "zip" if cfg.archive_format == "zip" else cfg.archive_format
    out_path = out_dir / pm.package_filename(record, ext=ext, models=cfg.include_models, encrypted=cfg.encrypt)
    try:
        result = await asyncio.to_thread(
            build_package, corpus, out_path, name=title, config=cfg,
            passphrase=(req.passphrase or None), identity=brain_identity.package_record(pid), originals=originals,
            documents=keep, scope=scope_record, package=record, changes_md=changes_md,
            changes_md_public=changes_md_public)
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        log.exception("rag package build failed")  # never includes the passphrase
        raise HTTPException(status_code=500, detail=f"package build failed: {type(e).__name__}") from e
    from finetune_studio.knowledge.readiness import record_package

    # The passphrase is only in memory here: the check gets it through the child's environment, never on disk.
    from finetune_studio.smart_brain.server_package import (
        studio_models_for,
        verify_package,
    )

    verified = await asyncio.to_thread(verify_package, result.path, result.passphrase or req.passphrase or None,
                                       studio_models=studio_models_for(corpus),
                                       expected_originals={o.document_id: o.sha256 for o in originals},
                                       out_of_scope=probes)
    await asyncio.to_thread(pm.add_history, out_dir, {
        **record, "origin": "exported", "filename": result.path.name, "name": title, "scope": scope_record["label"],
        "scope_spec": scope_req.model_dump() if scope_req and not scope_req.is_all() else {},
        "audience": audience, "originals": shipped_originals, "include_originals": bool(cfg.include_originals),
        "hide_file_names": bool(cfg.hide_file_names), "image_search": bool(cfg.image_search),
        "encrypted": result.encrypted, "changes_md": changes_md})
    await asyncio.to_thread(record_package, pid, filename=result.path.name, size=result.size, encrypted=result.encrypted,
                            identity=brain_identity.is_set(brain_identity.load(pid)), verified=verified)
    return JSONResponse(
        {
            "verified": verified,
            "filename": result.path.name,
            "download_url": f"/api/projects/{pid}/brain/standalone/download?file={result.path.name}",
            "size": result.size,
            "encrypted": result.encrypted,
            # Shown once; only set when generated here.
            "passphrase": result.passphrase,
            "config": cfg.model_dump(),
            "layers": result.layers,
            "runtime_search": result.runtime_search,
            "original_files": shipped_originals,
            "scope": scope_record,
            "widens": widens,
            "package": {k: record.get(k) for k in ("package_id", "version", "parent_version", "created", "name",
                                                   "slug", "root")},
            "changes": record["changes"],
            "identity": brain_identity.is_set(brain_identity.load(pid)),
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


