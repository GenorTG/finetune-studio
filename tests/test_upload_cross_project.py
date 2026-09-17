"""Cross-project file uploads must not collide on project_files.id.

The same content uploaded to two different projects used to derive both
``file_id`` from ``raw_hash[:16]``. ``project_files.id`` is a global
PRIMARY KEY, so the second project's insert raised
``UNIQUE constraint failed: project_files.id`` and OCR/image ingestion
silently failed for that upload.
"""

from __future__ import annotations

from finetune_studio.data.fs import file_library as fl


def test_write_uploaded_file_namespaces_id_by_project(mock_settings) -> None:
    """Identical bytes uploaded to two different projects must yield two
    distinct project_files rows (and so two distinct file_ids)."""
    from finetune_studio import db

    pid_a = db.create_project(name="cross-project-a")["id"]
    pid_b = db.create_project(name="cross-project-b")["id"]

    fl.ensure_dirs(pid_a)
    fl.ensure_dirs(pid_b)

    data = b"identical-image-bytes-png"

    meta_a = fl.write_uploaded_file(pid_a, data, "same.png", mime_hint="image/png")
    meta_b = fl.write_uploaded_file(pid_b, data, "same.png", mime_hint="image/png")

    assert meta_a.file_id != meta_b.file_id, (
        "identical content uploaded to different projects must yield different file_ids "
        f"(got {meta_a.file_id!r} for both; project_files.id PRIMARY KEY collision would "
        "have failed on the second insert)"
    )
    assert meta_a.file_id.startswith(pid_a[:8])
    assert meta_b.file_id.startswith(pid_b[:8])
    assert meta_a.raw_hash == meta_b.raw_hash  # content is identical


def test_write_uploaded_file_includes_image_png_metadata(mock_settings) -> None:
    """The OCR ingestion path must round-trip an image upload without raising."""
    from finetune_studio import db

    pid = db.create_project(name="cross-project-image")["id"]
    fl.ensure_dirs(pid)

    # A minimal valid PNG: 8-byte header + IHDR + IDAT + IEND (zero-length).
    # The OCR pipeline never opens the bytes — only tesseract on fan-dragon
    # does — but the file_library must still accept and persist them.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
        b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
        b"\r\n-\xb4"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    meta = fl.write_uploaded_file(pid, png_bytes, "tiny.png", mime_hint="image/png")
    assert meta.file_id.startswith(pid[:8])
    assert meta.auto_kind == "imgs"  # OCR ingestion path
    assert len(meta.raw_hash) == 64  # sha256 hex
    assert meta.size_bytes == len(png_bytes)
    assert meta.mime_type == "image/png"
