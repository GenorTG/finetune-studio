"""Authenticated-encrypted container for the exported Portable RAG package.

STANDALONE: this file is shipped verbatim next to ``server.py`` in every
export (like ``standalone_server.py`` it must not import ``finetune_studio``).
The only third-party dependency is ``cryptography`` (AES-256-GCM + scrypt).

Layout (all integers big-endian)::

    MAGIC "FTSRAGE1"                      8 bytes   plaintext
    u32 header_len + header JSON                    plaintext, minimal:
        {"v": 2, "kdf": "scrypt", "n": .., "r": .., "p": .., "salt": b64,
         "frame": 65536, "nonce": "prefix8+ctr32", "prefix": b64}
    data frames ...                                 AES-256-GCM, 64 KiB each
    index frame                                     AES-256-GCM (JSON: entry
                                                    names, offsets, sizes)
    trailer: u64 index_offset + u32 total_frames    plaintext

Everything content-bearing (document names, text, vectors, BM25) lives in the
entries; the plaintext part carries only format version, KDF parameters, salt
and the nonce scheme.  Each frame is its own AEAD message so a source file can
be read lazily without decrypting the rest.  Nonce = 8 random prefix bytes +
a 32-bit global frame counter (never reused for one key).  The AAD binds every
frame to the header hash, its counter and its kind (data vs index), so
reordering, splicing between containers and header tampering all fail the
tag check.  Truncation is caught by the trailer/index consistency checks.

The key is NEVER stored in the container: it is scrypt(passphrase, salt).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import struct
import threading
from typing import BinaryIO

MAGIC = b"FTSRAGE1"
FORMAT_VERSION = 2
FRAME = 64 * 1024
TAG = 16
DEFAULT_LOG_N = 17          # scrypt N = 2**17 (128 MiB at r=8) -- OWASP floor
_MAX_LOG_N = 20             # refuse hostile headers asking for >1 GiB
_MAX_HEADER = 4096
_TRAILER = struct.Struct(">QI")
_INDEX_KIND = b"I"
_DATA_KIND = b"D"


class ContainerError(Exception):
    """Base class: the container cannot be read."""


class WrongPassphraseOrTampered(ContainerError):
    """Authentication failed: wrong passphrase, or the data was modified."""


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # pragma: no cover - dependency hint
        raise ContainerError(
            "the 'cryptography' package is required to open this encrypted "
            "package (run install.sh, or: pip install cryptography)") from e
    return AESGCM(key)


def _derive_key(passphrase: bytes, salt: bytes, n: int, r: int, p: int) -> bytes:
    try:
        from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    except ImportError as e:  # pragma: no cover
        raise ContainerError("the 'cryptography' package is required") from e
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase)


def _nonce(prefix: bytes, counter: int) -> bytes:
    return prefix + struct.pack(">I", counter)


def _aad(header_hash: bytes, kind: bytes, counter: int) -> bytes:
    return header_hash + kind + struct.pack(">I", counter)


class ContainerWriter:
    """Streaming writer. ``fp`` is a binary, seekable-or-not file object that
    receives ONLY ciphertext (and the minimal plaintext header/trailer)."""

    def __init__(self, fp: BinaryIO, passphrase: str | bytes, *,
                 log_n: int = DEFAULT_LOG_N, r: int = 8, p: int = 1) -> None:
        if not 4 <= log_n <= _MAX_LOG_N:
            raise ValueError(f"log_n out of range: {log_n}")
        pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
        if not pw:
            raise ValueError("empty passphrase")
        self._fp = fp
        salt = secrets.token_bytes(16)
        self._prefix = secrets.token_bytes(8)
        header = {
            "v": FORMAT_VERSION, "kdf": "scrypt", "n": 2 ** log_n, "r": r, "p": p,
            "salt": base64.b64encode(salt).decode(), "frame": FRAME,
            "nonce": "prefix8+ctr32",
            "prefix": base64.b64encode(self._prefix).decode(),
        }
        hb = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        self._header_bytes = MAGIC + struct.pack(">I", len(hb)) + hb
        self._hash = hashlib.sha256(self._header_bytes).digest()
        self._aes = _aesgcm(_derive_key(pw, salt, 2 ** log_n, r, p))
        fp.write(self._header_bytes)
        self._pos = len(self._header_bytes)
        self._frames = 0
        self._entries: dict[str, dict] = {}
        self._closed = False

    def _write_frame(self, chunk: bytes) -> None:
        ct = self._aes.encrypt(_nonce(self._prefix, self._frames), chunk,
                               _aad(self._hash, _DATA_KIND, self._frames))
        self._fp.write(ct)
        self._pos += len(ct)
        self._frames += 1

    def add(self, name: str, data: bytes | BinaryIO) -> None:
        """Add one entry (bytes, or a binary file object read in frames)."""
        if self._closed:
            raise ValueError("container already closed")
        if name in self._entries:
            raise ValueError(f"duplicate entry: {name}")
        meta = {"o": self._pos, "f": self._frames, "s": 0, "c": 0}
        if isinstance(data, (bytes, bytearray, memoryview)):
            mv = memoryview(data)
            for i in range(0, len(mv), FRAME):
                self._write_frame(bytes(mv[i:i + FRAME]))
                meta["s"] += min(FRAME, len(mv) - i)
        else:
            while True:
                buf = data.read(FRAME)
                while buf and len(buf) < FRAME:  # raw readers may short-read
                    more = data.read(FRAME - len(buf))
                    if not more:
                        break
                    buf += more
                if not buf:
                    break
                self._write_frame(buf)
                meta["s"] += len(buf)
                if len(buf) < FRAME:
                    break
        meta["c"] = self._frames - meta["f"]
        self._entries[name] = meta

    def close(self) -> None:
        if self._closed:
            return
        index = json.dumps({"entries": self._entries, "frames": self._frames},
                           separators=(",", ":")).encode()
        index_offset = self._pos
        ct = self._aes.encrypt(_nonce(self._prefix, self._frames), index,
                               _aad(self._hash, _INDEX_KIND, self._frames))
        self._fp.write(ct)
        self._fp.write(_TRAILER.pack(index_offset, self._frames))
        self._closed = True


class ContainerReader:
    """Reads entries in memory only; never writes anything to disk."""

    def __init__(self, path: str | os.PathLike, passphrase: str | bytes) -> None:
        pw = passphrase.encode("utf-8") if isinstance(passphrase, str) else passphrase
        self._lock = threading.Lock()
        self._fp = open(path, "rb")  # noqa: SIM115 - held for lazy reads
        try:
            self._open(pw)
        except BaseException:
            self._fp.close()
            raise

    def close(self) -> None:
        self._fp.close()

    def __enter__(self) -> ContainerReader:  # noqa: PYI034 - py3.10 compatible
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _open(self, pw: bytes) -> None:
        fp = self._fp
        if fp.read(len(MAGIC)) != MAGIC:
            raise ContainerError("not an encrypted RAG package (bad magic)")
        raw_len = fp.read(4)
        if len(raw_len) != 4:
            raise ContainerError("truncated header")
        (hlen,) = struct.unpack(">I", raw_len)
        if not 2 <= hlen <= _MAX_HEADER:
            raise ContainerError("corrupt header")
        hb = fp.read(hlen)
        if len(hb) != hlen:
            raise ContainerError("truncated header")
        try:
            h = json.loads(hb)
            n, r, p = int(h["n"]), int(h["r"]), int(h["p"])
            salt = base64.b64decode(h["salt"])
            self._prefix = base64.b64decode(h["prefix"])
            version = int(h["v"])
        except (ValueError, KeyError, TypeError) as e:
            raise ContainerError(f"corrupt header: {e}") from e
        if version != FORMAT_VERSION or h.get("kdf") != "scrypt" \
                or h.get("nonce") != "prefix8+ctr32" or int(h.get("frame", 0)) != FRAME:
            raise ContainerError(f"unsupported container format (v={version})")
        if not (2 ** 4 <= n <= 2 ** _MAX_LOG_N) or n & (n - 1) or not 1 <= r <= 32 \
                or not 1 <= p <= 16 or len(self._prefix) != 8 or len(salt) < 8:
            raise ContainerError("corrupt header: bad KDF parameters")
        self._hash = hashlib.sha256(MAGIC + raw_len + hb).digest()
        data_start = len(MAGIC) + 4 + hlen
        self._aes = _aesgcm(_derive_key(pw, salt, n, r, p))

        fp.seek(0, os.SEEK_END)
        size = fp.tell()
        if size < data_start + _TRAILER.size + TAG:
            raise ContainerError("truncated package")
        fp.seek(size - _TRAILER.size)
        index_offset, total = _TRAILER.unpack(fp.read(_TRAILER.size))
        index_end = size - _TRAILER.size
        if not data_start <= index_offset <= index_end - TAG:
            raise WrongPassphraseOrTampered(
                "package is truncated or tampered with")
        fp.seek(index_offset)
        ct = fp.read(index_end - index_offset)
        try:
            index = json.loads(self._aes.decrypt(
                _nonce(self._prefix, total), ct,
                _aad(self._hash, _INDEX_KIND, total)))
        except Exception as e:  # InvalidTag / bad JSON -> fail closed
            raise WrongPassphraseOrTampered(
                "wrong passphrase, or the package is corrupted/tampered") from e
        self._entries: dict[str, dict] = index["entries"]
        if index.get("frames") != total:
            raise WrongPassphraseOrTampered("package index mismatch")

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def size(self, name: str) -> int:
        return int(self._entries[name]["s"])

    def read(self, name: str) -> bytes:
        """Decrypt one entry fully into memory (lazy: only its frames)."""
        try:
            m = self._entries[name]
        except KeyError:
            raise KeyError(f"no such entry in package: {name}") from None
        out = bytearray()
        remaining = int(m["s"])
        for k in range(int(m["c"])):
            plain = min(FRAME, remaining)
            with self._lock:
                self._fp.seek(int(m["o"]) + k * (FRAME + TAG))
                ct = self._fp.read(plain + TAG)
            counter = int(m["f"]) + k
            try:
                out += self._aes.decrypt(_nonce(self._prefix, counter), ct,
                                         _aad(self._hash, _DATA_KIND, counter))
            except Exception as e:
                raise WrongPassphraseOrTampered(
                    f"package data failed authentication ({name!r}): "
                    "corrupted or tampered") from e
            remaining -= plain
        if remaining:
            raise WrongPassphraseOrTampered("package entry size mismatch")
        return bytes(out)
