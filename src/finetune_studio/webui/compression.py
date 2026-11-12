"""Gzip for pages, styles, scripts and JSON: never for streams, downloads or already-compressed files.

The app shipped every page and asset uncompressed (app.css 186 KB, the Smart Brain page 240 KB of HTML); gzip makes
text 4-6x smaller. Only an allow-list of text types is compressed, and only whole responses: a streamed body
(``more_body`` on its first chunk and no Content-Length: SSE, NDJSON progress, log tails, chat tokens) and anything sent as an attachment
(a package, an export, a ZIP) pass through untouched, so nothing a client reads incrementally is ever buffered.
"""
from __future__ import annotations

from starlette.datastructures import Headers
from starlette.middleware.gzip import GZipResponder, IdentityResponder
from starlette.types import ASGIApp, Message, Receive, Scope, Send

COMPRESSED_TYPES = frozenset({
    "text/html", "text/css", "text/javascript", "application/javascript", "application/json",
    "image/svg+xml", "text/markdown",
})
MINIMUM_SIZE = 1024
LEVEL = 6   # 9 costs ~3x the CPU for ~2 % fewer bytes


def compressible(headers: Headers) -> bool:
    media = headers.get("content-type", "").partition(";")[0].strip().lower()
    disposition = headers.get("content-disposition", "").lower()
    return media in COMPRESSED_TYPES and "attachment" not in disposition


class _Responder(GZipResponder):
    """Starlette's gzip responder, limited to whole responses of the allow-listed types."""

    async def send_with_compression(self, message: Message) -> None:
        kind = message["type"]
        if kind == "http.response.start":
            if not compressible(Headers(raw=message.get("headers") or [])):
                self.content_type_is_excluded = True
                await self.send(message)
                return
        elif kind == "http.response.body" and not self.started and message.get("more_body", False) \
                and not (self.content_type_is_excluded or self.content_encoding_set or self.partial_response) \
                and "content-length" not in Headers(raw=self.initial_message.get("headers") or []):
            # A streamed body of unknown length (a stream, not a file sent in chunks): pass it through as it comes.
            self.content_type_is_excluded = True
            await self.send(self.initial_message)
            await self.send(message)
            return
        await super().send_with_compression(message)


class CompressionMiddleware:
    def __init__(self, app: ASGIApp, minimum_size: int = MINIMUM_SIZE) -> None:
        self.app = app
        self.minimum_size = minimum_size

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or "gzip" not in Headers(scope=scope).get("accept-encoding", ""):
            await self.app(scope, receive, send)
            return
        responder: IdentityResponder = _Responder(self.app, self.minimum_size, compresslevel=LEVEL)
        await responder(scope, receive, send)
