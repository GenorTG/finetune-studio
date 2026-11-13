"""A new version of a package keeps what the package is for: its documents, its original files, hidden file names,
its audience and its encryption.

The danger it guards against: a public bot's package (``public/`` only, no originals, names hidden) re-exported as
"a new version" with the export form's defaults (all documents, originals on) became the internal package, kept the
public package's id, and ``update.sh`` installed it into the running public bot. So:

- ``inherit`` fills every setting the request leaves open from the base version (scope, originals, hidden names,
  audience);
- ``widening`` lists what the new version opens up compared to its base (documents outside the base's scope,
  originals turned on, file names shown, a public audience changed, encryption turned off); the export refuses a
  widening unless it is confirmed (``allow_widen``), and the confirmed list goes into ``CHANGES.md`` and
  ``package.json`` (``widens``), where ``update.sh`` reads it;
- ``policy`` is the package's name-free record of the same settings (``package.json``), so ``update.sh`` can compare
  the installed version with the new one.
"""
from __future__ import annotations

from typing import Any

from .scope import DocRef, DocumentScope

AUDIENCE_LABEL = {"public": "Customers (public bot)", "internal": "Your team (internal assistant)", "other": "Other"}


def _folder(path: str) -> str:
    return path.rsplit("/", 1)[0] + "/" if "/" in path else path


def base_scope(base: dict) -> DocumentScope | None:
    """The base version's document choice. Versions exported since 0.5 record it (``scope_spec``); for older ones
    it is the folders their documents were in (all documents when the base said so)."""
    spec = base.get("scope_spec")
    if isinstance(spec, dict):
        scope = DocumentScope(**{k: spec.get(k) or [] for k in ("include", "exclude", "tags", "document_ids")})
        return None if scope.is_all() else scope
    if str(base.get("scope") or "") == "all documents":
        return None
    paths = [str(d.get("path") or "") for d in base.get("documents") or [] if d.get("path")]
    if not paths:
        return None
    return DocumentScope(include=sorted({_folder(p) for p in paths}))


def base_settings(base: dict) -> dict[str, Any]:
    """What a new version inherits: ``scope`` (a DocumentScope dump, None = all), ``include_originals``,
    ``hide_file_names``, ``audience``, ``encrypted``."""
    scope = base_scope(base)
    originals = base.get("include_originals")
    if originals is None:
        originals = bool(base.get("originals") or (base.get("policy") or {}).get("originals"))
    hide = base.get("hide_file_names")
    if hide is None:
        hide = (base.get("policy") or {}).get("hide_file_names")
    return {"scope": scope.model_dump() if scope else None, "include_originals": bool(originals),
            "hide_file_names": bool(hide), "audience": base.get("audience") or "",
            "encrypted": base.get("encrypted") is not False}


def inherit_scope(requested: DocumentScope | None, base: dict | None) -> DocumentScope | None:
    """The request's scope, or the base's when the request names none."""
    if requested is not None or not base:
        return requested
    return base_scope(base)


def widening(base: dict, docs: list[DocRef], keep: set[str] | None, *, include_originals: bool,
             hide_file_names: bool, audience: str | None, encrypt: bool) -> list[dict[str, str]]:
    """``[{"code", "text"}]``: what this version opens up compared to ``base`` (empty: nothing)."""
    out: list[dict[str, str]] = []
    was = base_settings(base)
    old_scope = base_scope(base)
    old = {d.document_id for d in docs if old_scope is None or old_scope.selects(d)}
    new = keep if keep is not None else {d.document_id for d in docs}
    extra = [d for d in docs if d.document_id in new - old]
    if extra:
        folders = sorted({_folder(d.path) for d in extra})
        shown = ", ".join(folders[:6]) + (f" (+{len(folders) - 6} more)" if len(folders) > 6 else "")
        out.append({"code": "scope", "text": f"{len(extra)} document(s) outside version {base.get('version')}'s "
                                             f"scope ({shown})"})
    if include_originals and not was["include_originals"]:
        out.append({"code": "originals", "text": "original files: off → on (anyone who reaches the server can "
                                                 "download them)"})
    if was["hide_file_names"] and not hide_file_names:
        out.append({"code": "file_names", "text": "file names in answers: hidden → shown"})
    if was["audience"] == "public" and audience and audience != "public":
        out.append({"code": "audience", "text": f"used by: {AUDIENCE_LABEL['public']} → "
                                                f"{AUDIENCE_LABEL.get(audience, audience)}"})
    if was["encrypted"] and not encrypt:
        out.append({"code": "encryption", "text": "encryption: on → off (plaintext package)"})
    return out


def widening_markdown(version: str | None, widens: list[dict[str, str]], *, names: bool = True) -> str:
    """The CHANGES.md section for a confirmed widening (``names=False``: the codes only)."""
    if not widens:
        return ""
    lines = "".join(f"- {w['text'] if names else w['code']}\n" for w in widens)
    return (f"\n## Wider than version {version}\n\nConfirmed at export: this version opens up more than the one "
            f"before it. `update.sh` refuses it on a running copy unless `--allow-widen` is given.\n\n{lines}")


def policy(scope_record: dict, *, originals: int, hide_file_names: bool, audience: str | None,
           encrypted: bool) -> dict[str, Any]:
    """The package's settings without names, for ``package.json``."""
    return {"scope": "all" if scope_record.get("label", "all documents") == "all documents" else "part",
            "documents": scope_record.get("documents"), "of": scope_record.get("of"), "originals": int(originals),
            "hide_file_names": bool(hide_file_names), "audience": audience or "", "encrypted": bool(encrypted)}


def passphrase_problem(base: dict | None, *, encrypt: bool, passphrase: str | None, new_passphrase: bool) -> str | None:
    """Why the passphrase choice would break running copies of the base (None: fine)."""
    if not base or not encrypt or new_passphrase or passphrase:
        return None
    if base.get("encrypted") is False:
        return None
    return (f"This is a new version of an encrypted package: enter the passphrase of version {base.get('version')} "
            "so running copies keep working after update.sh, or choose a new passphrase on purpose (then every "
            "running copy needs the new one in its keyfile).")


def passphrase_matches_base(base: dict, package_dir, passphrase: str) -> bool | None:
    """Whether ``passphrase`` opens the base version's archive (None: that archive is not here to check)."""
    from pathlib import Path

    from . import package_import

    name = str(base.get("filename") or "")
    archive = Path(package_dir) / name
    if not name or "/" in name or not archive.is_file() or base.get("encrypted") is False:
        return None
    try:
        package_import.read_record(archive, passphrase, Path(package_dir))
    except ValueError as e:
        if "no version record" in str(e):
            return None
        return False
    except Exception:  # noqa: BLE001 - a wrong passphrase raises the container's own error type
        return False
    return True
